"""Plain-assert tests for yee.integrations.tensorfold. Run: python tests/test_tensorfold.py

Round trip our disk record -> TensorFold chunk files -> our record, byte checks against recall_store.zig's layout,
multi-chunk files (small chunk size), refusals (vocab, bad header, existing files), detect() against fake binaries
and status/clear against a local fake /v1/recall server.
"""
from __future__ import annotations

import http.server
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "yee")):
    sys.path.insert(0, ROOT)

from yee import core as rc  # noqa: E402
from yee.integrations import tensorfold as tf  # noqa: E402

RESULTS = []
VOCAB = 129280
DIGEST = "ab" * 8 + "cd" * 24  # 64 hex
IDENT = {"tokenizer_digest": "f" * 64, "vocab_size": VOCAB, "model": "m"}


def test(fn):
    RESULTS.append(fn)
    return fn


def segs(n, seed, lo=40, hi=400):
    g = np.random.default_rng(seed)
    return [g.integers(10, VOCAB, size=int(g.integers(lo, hi))).astype(np.int64) for _ in range(n)]


def write_ours(d, segments, chunk):
    """Our chunk-%08d.ids layout directly (int32 LE, -1 between segments), as _DiskStore writes it."""
    os.makedirs(d, exist_ok=True)
    stream = []
    for s in segments:
        stream.extend(int(x) for x in s)
        stream.append(-1)
    arr = np.array(stream, dtype="<i4")
    for k in range(0, arr.size, chunk):
        with open(os.path.join(d, "chunk-%08d.ids" % (k // chunk)), "wb") as f:
            f.write(arr[k:k + chunk].tobytes())
    with open(os.path.join(d, "identity"), "w") as f:
        json.dump(IDENT, f)


class Tmp:
    def __enter__(self):
        self.d = tempfile.mkdtemp(prefix="ra_tf_")
        return self.d

    def __exit__(self, *a):
        shutil.rmtree(self.d, ignore_errors=True)


@test
def test_export_bytes_and_roundtrip():
    with Tmp() as d:
        ours, tfd, back = os.path.join(d, "ours"), os.path.join(d, "tf"), os.path.join(d, "back")
        S = segs(30, 1)
        write_ours(ours, S, 4096)
        r = tf.export_to_tf(ours, tfd, DIGEST, VOCAB, "model-x", chunk_ids=4096)
        assert r["segments"] == 30 and r["dropped_segments"] == 0, r
        names = sorted(n for n in os.listdir(tfd) if n.endswith(".ids"))
        assert names[0] == "%016x.ids" % 0 and all(len(n) == 20 for n in names), names
        assert len(names) == r["files"] > 1
        blob = b""
        for i, n in enumerate(names):
            data = open(os.path.join(tfd, n), "rb").read()
            assert data[:8] == b"TFRECALL"
            assert struct.unpack("<II", data[8:16]) == (1, VOCAB)
            assert data[16:32] == bytes.fromhex(DIGEST)[:16]
            body = data[32:]
            assert len(body) % 4 == 0
            if i + 1 < len(names):
                assert len(body) == 4 * 4096, "every file but the last is full"
            blob += body
        u = np.frombuffer(blob, dtype="<u4")
        expect = np.concatenate([np.concatenate([s.astype(np.uint32), [0xFFFFFFFF]]) for s in S]).astype(np.uint32)
        assert np.array_equal(u, expect)
        ident = open(os.path.join(tfd, "identity")).read()
        assert ident == "tensorfold recall 1\ndigest %s\nvocab %d\nname model-x\n" % (DIGEST, VOCAB), ident
        r2 = tf.import_from_tf(tfd, back, IDENT, chunk_ids=4096)
        assert r2["segments"] == 30 and r2["ids"] == sum(int(s.size) for s in S)
        got = tf.read_our_segments(back, 4096)
        assert len(got) == 30 and all(np.array_equal(a, b) for a, b in zip(got, S))
        for n in os.listdir(back):
            if n.startswith("chunk-"):
                assert open(os.path.join(ours, n), "rb").read() == open(os.path.join(back, n), "rb").read(), n


@test
def test_import_into_core_record_and_lookup():
    """The imported record loads in core.RecallRecord (disk backend) and continues every segment exactly."""
    with Tmp() as d:
        tfd, ours = os.path.join(d, "tf"), os.path.join(d, "ours")
        S = segs(12, 2, 300, 600)
        src = os.path.join(d, "src")
        write_ours(src, S, rc.CHUNK_IDS)
        tf.export_to_tf(src, tfd, DIGEST, VOCAB, "m")
        tf.import_from_tf(tfd, ours, IDENT)
        rec = rc.RecallRecord(directory=ours, identity=IDENT, gib=0.01, backend="disk", min_free_gib=0.0)
        try:
            st = rec.stats()
            assert st["ids"] >= sum(int(s.size) for s in S), st
            eng = rc.RecallEngine(rec, gamma=5)
            hits = 0
            for i, s in enumerate(S):
                rid = "r%d" % i
                cut = len(s) // 2
                eng.begin_stream(rid, [int(x) for x in s[:cut]])
                p = eng.propose(rid, n=8)
                if p is not None:
                    want = [int(x) for x in s[cut:cut + 8]]
                    assert [int(x) for x in p.tokens[:len(want)]][:int(p.real)] == want[:int(p.real)], (i, p)
                    hits += 1
                eng.end_stream(rid, [], finished=False)
            assert hits >= 10, hits
        finally:
            close = getattr(rec, "close", None)
            if close:
                close()


@test
def test_identity_file_from_tensorfold_wins():
    with Tmp() as d:
        ours, tfd = os.path.join(d, "ours"), os.path.join(d, "tf")
        write_ours(ours, segs(3, 3), 4096)
        os.makedirs(tfd)
        open(os.path.join(tfd, "identity"), "w").write(tf.tf_identity_text(DIGEST, VOCAB, "tf-model"))
        r = tf.export_to_tf(ours, tfd, chunk_ids=4096)
        assert r["files"] == 1
        assert open(os.path.join(tfd, "0000000000000000.ids"), "rb").read()[16:32] == bytes.fromhex(DIGEST)[:16]
        try:
            tf.export_to_tf(ours, tfd, chunk_ids=4096)
            raise AssertionError("second export into a filled dir must be refused")
        except tf.TensorFoldError as e:
            assert "already holds" in str(e)


@test
def test_refusals():
    with Tmp() as d:
        ours, tfd = os.path.join(d, "ours"), os.path.join(d, "tf")
        write_ours(ours, segs(3, 4), 4096)
        for args, word in (((None, None), "64 hex"), ((DIGEST[:16], VOCAB), "64 hex"), ((DIGEST, VOCAB + 1), "vocab")):
            try:
                tf.export_to_tf(ours, tfd + word.replace(" ", ""), args[0], args[1], chunk_ids=4096)
                raise AssertionError("must refuse %s" % (args,))
            except tf.TensorFoldError as e:
                assert word in str(e), (args, e)
        tf.export_to_tf(ours, tfd, DIGEST, VOCAB, chunk_ids=4096)
        try:
            tf.import_from_tf(tfd, os.path.join(d, "x"), dict(IDENT, vocab_size=VOCAB - 1), chunk_ids=4096)
            raise AssertionError("import with another vocab must be refused")
        except tf.TensorFoldError as e:
            assert "vocab" in str(e)
        p = os.path.join(tfd, "0000000000000000.ids")
        data = bytearray(open(p, "rb").read())
        data[0:8] = b"XXRECALL"
        open(p, "wb").write(bytes(data))
        try:
            tf.read_tf_segments(tfd, 4096)
            raise AssertionError("bad header must be refused")
        except tf.TensorFoldError as e:
            assert "header" in str(e)


@test
def test_out_of_vocab_segments_dropped():
    with Tmp() as d:
        ours, tfd = os.path.join(d, "ours"), os.path.join(d, "tf")
        S = segs(4, 5)
        S[1] = S[1].copy()
        S[1][3] = VOCAB + 5
        write_ours(ours, S, 4096)
        r = tf.export_to_tf(ours, tfd, DIGEST, VOCAB, chunk_ids=4096)
        assert r["segments"] == 3 and r["dropped_segments"] == 1
        got, vocab, d16 = tf.read_tf_segments(tfd, 4096)
        assert vocab == VOCAB and d16 == bytes.fromhex(DIGEST)[:16] and len(got) == 3


def _fake_bin(d, with_recall, version="1.0.12"):
    p = os.path.join(d, "fake_tf_%s.py" % ("r" if with_recall else "n"))
    help_text = "usage: tensorfold serve [--model M] [--port P]%s [--dspark]" % (
        " [--recall] [--recall-dir D] [--recall-gib X] [--recall-read-only]" if with_recall else "")
    open(p, "w").write("import sys\na=sys.argv[1:]\n"
                       "if a[:1]==['--version']: print('tensorfold %s')\n"
                       "elif a[:1]==['serve']: print(%r)\n"
                       "else: sys.exit(2)\n" % (version, help_text))
    return [sys.executable, p]


@test
def test_detect_and_args():
    with Tmp() as d:
        yes = tf.detect(argv_prefix=_fake_bin(d, True))
        assert yes["found"] and yes["recall_supported"] and yes["version"] == "1.0.12" and yes["heads"]["dspark"], yes
        assert tf.launch_args(0.5, "/r", True, yes) == ["--recall-gib", "0.5", "--recall-dir", "/r", "--recall-read-only"]
        assert tf.launch_args(None, None, False, yes) == ["--recall"]
        no = tf.detect(argv_prefix=_fake_bin(d, False, "1.0.2"))
        assert no["found"] and not no["recall_supported"] and "1.0.12" in no["reason"], no
        try:
            tf.launch_args(0.25, detected=no)
            raise AssertionError("no Recall: must raise")
        except tf.TensorFoldError as e:
            assert "1.0.12" in str(e)
        try:
            tf.launch_args(0.001)
            raise AssertionError("below 1/128 GiB must raise")
        except tf.TensorFoldError:
            pass
        missing = tf.detect(binary=os.path.join(d, "nope"))
        assert not missing["found"]


@test
def test_status_and_clear_http():
    calls = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            calls.append(("GET", self.path))
            self._reply(200 if self.path == "/v1/recall" else 404, {"ids": 1234, "identity": "abababababababab"})

        def do_DELETE(self):
            calls.append(("DELETE", self.path))
            self._reply(200, {"emptied": True})

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        url = "http://127.0.0.1:%d" % srv.server_address[1]
        assert tf.status(url)["ids"] == 1234
        assert tf.clear(url + "/")["emptied"] is True
        assert calls == [("GET", "/v1/recall"), ("DELETE", "/v1/recall")], calls
    finally:
        srv.shutdown()
    try:
        tf.status("http://127.0.0.1:9", timeout=1.0)
        raise AssertionError("unreachable must raise")
    except tf.TensorFoldError:
        pass


@test
def test_cli_args():
    with Tmp() as d:
        ours, tfd = os.path.join(d, "ours"), os.path.join(d, "tf")
        write_ours(ours, segs(2, 9), 4096)
        assert tf.main(["args", "--no-detect", "--gib", "0.5"]) == 0
        assert tf.main(["export", "--from", ours, "--to", tfd]) == 2  # no digest, no identity file
        assert tf.main(["export", "--from", ours, "--to", tfd, "--digest", DIGEST, "--vocab", str(VOCAB)]) == 0
        ij = os.path.join(d, "ident.json")
        json.dump(IDENT, open(ij, "w"))
        assert tf.main(["import", "--from", tfd, "--to", os.path.join(d, "b"), "--identity-json", ij]) == 0


def main():
    ok = fail = skip = 0
    for fn in RESULTS:
        try:
            res = fn()
            if isinstance(res, str) and res.startswith("SKIP"):
                skip += 1
                print("SKIP %s: %s" % (fn.__name__, res))
            else:
                ok += 1
                print("PASS %s" % fn.__name__)
        except Exception:
            fail += 1
            print("FAIL %s" % fn.__name__)
            traceback.print_exc()
    if fail:
        print("FAILED %d / %d (skip %d)" % (fail, ok + fail, skip))
        sys.exit(1)
    print("ALL PASS n=%d skip=%d" % (ok, skip))


if __name__ == "__main__":
    main()
