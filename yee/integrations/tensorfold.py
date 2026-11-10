"""TensorFold end of recall-accel: TensorFold (ashhart/TensorFold, a Zig binary) has Recall built in since 1.0.12
and no Python plugin interface, so this module does not hook it. It

  * detects the binary and whether its ``serve`` takes ``--recall``          (``detect``)
  * builds the launch flags for its native Recall                             (``launch_args``)
  * reads / empties its record over HTTP ``/v1/recall``                       (``status`` / ``clear``)
  * converts records both ways, so a record collected under SGLang / vLLM warms TensorFold and back
                                                                              (``export_to_tf`` / ``import_from_tf``)

TensorFold's record format, from zig/src/core/lanes/recall_store.zig and recall_record.zig (main branch):
  * directory ``$HOME/.cache/tensorfold/recall/<digest[:8] hex>`` (``defaultDir``), made 0700
  * chunk files ``{seq:016x}.ids`` (``chunkPath``; ``listChunks`` takes names of 20 chars ending ".ids"),
    numbered consecutively; ``load`` stops at a gap and at the first short file
  * 32-byte header ``"TFRECALL"`` + u32 LE version 1 + u32 LE vocab + identity digest[:16] (``header``); ``readChunk``
    refuses a file whose header differs from the server's own byte for byte
  * payload u32 LE ids, segment end ``mark = 0xFFFFFFFF``, ``chunk_tokens = 1 << 20`` ids per file;
    an id >= vocab stops the load (``error.RecallRead``)
  * ``identity``: ``"tensorfold recall 1\\ndigest <64 hex>\\nvocab <n>\\nname <model>\\n"`` (``checkIdentity``);
    the digest is TensorFold's own (tokenizer/identity.zig: sha256 over its loaded vocabulary, merges and
    added tokens) and differs from ours, so the export takes it from a TensorFold identity file or ``--digest``.

Our format (core.py ``_DiskStore``): ``chunk-%08d.ids``, no header, int32 LE, segment end -1, CHUNK_IDS = 1 << 20,
JSON ``identity`` {tokenizer_digest, vocab_size, model}. -1 as int32 and 0xFFFFFFFF as u32 are the same bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import urllib.error
import urllib.request

import numpy as np

MAGIC = b"TFRECALL"
VERSION = 1
HEAD_BYTES = 32
MARK = 0xFFFFFFFF
TF_CHUNK = 1 << 20
MIN_GIB = 1.0 / 128.0
MIN_TF_VERSION = (1, 0, 12)
_TF_NAME = re.compile(r"^([0-9a-f]{16})\.ids$")
_OUR_NAME = re.compile(r"^chunk-(\d{8})\.ids$")


class TensorFoldError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------- detection
def _run(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return -1, repr(e)


def _parse_version(text):
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(x) for x in m.groups()) if m else None


def find_binary(binary="tensorfold"):
    if binary and os.path.sep in binary or (binary and os.path.isfile(binary)):
        return binary if os.path.isfile(binary) else None
    hit = shutil.which(binary or "tensorfold")
    if hit:
        return hit
    for p in ("/usr/local/bin/tensorfold", "/opt/tensorfold/bin/tensorfold", os.path.expanduser("~/.local/bin/tensorfold"),
              os.path.expanduser("~/TensorFold/zig-out/bin/tensorfold")):
        if os.path.isfile(p):
            return p
    return None


def detect(binary="tensorfold", argv_prefix=None) -> dict:
    """{found, path, version, recall_supported, heads, reason}. ``argv_prefix`` runs it some other way
    (e.g. ``["python", "fake.py"]`` in tests or ``["docker", "run", "--rm", img]``)."""
    out = {"found": False, "path": None, "version": None, "recall_supported": False, "heads": {}, "reason": ""}
    if argv_prefix:
        base = list(argv_prefix)
        out["path"] = " ".join(base)
    else:
        path = find_binary(binary)
        if not path:
            out["reason"] = "tensorfold binary not found (PATH, /usr/local/bin, /opt/tensorfold/bin)"
            return out
        base = [path]
        out["path"] = path
    rc_v, vtext = _run(base + ["--version"])
    rc_h, help_text = _run(base + ["serve", "--help"])
    if rc_h == -1 and rc_v == -1:
        out["reason"] = "cannot run %s: %s" % (out["path"], help_text.strip()[:200])
        return out
    out["found"] = True
    ver = _parse_version(vtext) or _parse_version(help_text)
    out["version"] = ".".join(map(str, ver)) if ver else None
    rc_c, caps = _run(base + ["capabilities", "--json"])
    heads = {}
    if rc_c == 0:
        try:
            js = json.loads(caps)
            for k in ("dspark", "mtp", "dflash", "dflash2", "eagle", "recall"):
                if k in js:
                    heads[k] = js[k]
            if js.get("recall"):
                out["recall_supported"] = True
        except ValueError:
            pass
    low = help_text.lower()
    for k in ("dspark", "mtp", "dflash2", "dflash", "eagle"):
        heads.setdefault(k, k in low)
    out["heads"] = heads
    if "--recall" in help_text:
        out["recall_supported"] = True
    if out["recall_supported"]:
        out["reason"] = "native Recall available (serve --recall)"
    elif ver and ver < MIN_TF_VERSION:
        out["reason"] = "TensorFold %s has no Recall: upgrade to >= 1.0.12" % out["version"]
    else:
        out["reason"] = "this TensorFold's serve has no --recall flag: upgrade to >= 1.0.12"
    return out


def launch_args(gib=0.25, recall_dir=None, read_only=False, detected=None) -> list:
    """Flags to append to ``tensorfold serve``. Raises TensorFoldError when the binary has no Recall."""
    if detected is not None and not detected.get("recall_supported"):
        raise TensorFoldError(detected.get("reason") or "TensorFold without Recall")
    args = []
    if gib is not None:
        g = float(gib)
        if g < MIN_GIB:
            raise TensorFoldError("--recall-gib must be >= 1/128 (%.6f), got %s" % (MIN_GIB, gib))
        args += ["--recall-gib", ("%g" % g)]
    if recall_dir:
        args += ["--recall-dir", str(recall_dir)]
    if read_only:
        args.append("--recall-read-only")
    return args or ["--recall"]


# Yee patch tensorfold/0001-recall-after-priced.patch: with this set, a priced proposer (DSv4.1 lookup plan,
# Llama drafter, Qwen4 device lanes) that offers no draft falls back to Recall; unset = upstream behaviour.
RECALL_AFTER_PRICED_ENV = "TF_RECALL_AFTER_PRICED"


def launch_env() -> dict:
    """Environment for ``tensorfold serve`` built with the Yee patch (``yee patch apply tensorfold``)."""
    return {RECALL_AFTER_PRICED_ENV: "1"}


# --------------------------------------------------------------------------------------------- HTTP
def _http(method, base_url, timeout=5.0):
    url = base_url.rstrip("/") + "/v1/recall"
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            code = r.status
    except urllib.error.HTTPError as e:
        body, code = e.read().decode("utf-8", "replace"), e.code
    except (urllib.error.URLError, OSError) as e:
        raise TensorFoldError("cannot reach %s: %s" % (url, e))
    try:
        data = json.loads(body) if body.strip() else {}
    except ValueError:
        data = {"text": body}
    return code, data


def status(base_url, timeout=5.0) -> dict:
    code, data = _http("GET", base_url, timeout)
    if code != 200:
        raise TensorFoldError("GET /v1/recall -> %d %s" % (code, data))
    return data


def clear(base_url, timeout=5.0) -> dict:
    code, data = _http("DELETE", base_url, timeout)
    if code not in (200, 204):
        raise TensorFoldError("DELETE /v1/recall -> %d %s" % (code, data))
    return data


# --------------------------------------------------------------------------------------------- formats
def tf_header(vocab: int, digest: bytes) -> bytes:
    if len(digest) < 16:
        raise TensorFoldError("TensorFold digest needs at least 16 bytes, got %d" % len(digest))
    return MAGIC + struct.pack("<II", VERSION, int(vocab)) + bytes(digest[:16])


def read_tf_identity(tf_dir) -> dict:
    p = os.path.join(tf_dir, "identity")
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
    out = {}
    lines = text.split("\n")
    if not lines or not lines[0].startswith("tensorfold recall "):
        raise TensorFoldError("%s is not a TensorFold identity file" % p)
    for ln in lines[1:]:
        if ln.startswith("digest "):
            out["digest"] = ln[7:].strip()
        elif ln.startswith("vocab "):
            out["vocab"] = int(ln[6:].strip())
        elif ln.startswith("name "):
            out["name"] = ln[5:]
    if len(out.get("digest", "")) != 64:
        raise TensorFoldError("%s has no 64-hex digest line" % p)
    return out


def tf_identity_text(digest_hex: str, vocab: int, name: str) -> str:
    return "tensorfold recall %d\ndigest %s\nvocab %d\nname %s\n" % (VERSION, digest_hex.lower(), int(vocab), name)


def _our_run(our_dir):
    seqs = sorted(int(m.group(1)) for m in (_OUR_NAME.match(n) for n in os.listdir(our_dir)) if m)
    if not seqs:
        return []
    run = [seqs[-1]]
    for s in reversed(seqs[:-1]):
        if s != run[0] - 1:
            break
        run.insert(0, s)
    return run


def read_our_segments(our_dir, chunk_ids=TF_CHUNK):
    """Our record as segments (lists of ids), oldest first; the same reading as core._DiskStore._load."""
    run = _our_run(our_dir)
    parts = []
    for k, s in enumerate(run):
        with open(os.path.join(our_dir, "chunk-%08d.ids" % s), "rb") as f:
            data = f.read()
        arr = np.frombuffer(data[: len(data) // 4 * 4], dtype="<i4")[:chunk_ids]
        if k + 1 < len(run) and arr.size < chunk_ids:  # an interior short file reads as sentinels after its end
            arr = np.concatenate([arr, np.full(chunk_ids - arr.size, -1, dtype="<i4")])
        parts.append(arr)
    if not parts:
        return []
    stream = np.concatenate(parts).astype(np.int64)
    return _split(stream, -1)


def _split(stream, mark):
    cut = np.flatnonzero(stream == mark)
    segs, a = [], 0
    for c in cut:
        if c > a:
            segs.append(stream[a:c])
        a = c + 1
    if a < stream.size:
        segs.append(stream[a:])
    return segs


def _chunks_of(stream, size):
    for i in range(0, stream.size, size):
        yield i // size, stream[i:i + size]


def export_to_tf(our_dir, tf_dir, digest_hex=None, vocab=None, name=None, chunk_ids=TF_CHUNK) -> dict:
    """Write our record as TensorFold chunk files into ``tf_dir`` (which must hold no chunk files yet).

    digest/vocab: ``tf_dir/identity`` when it exists (written by a TensorFold started once with that
    --recall-dir), else ``digest_hex`` (64 hex: TensorFold's full identity digest) and ``vocab``.
    """
    ours = _read_our_identity(our_dir)
    if os.path.exists(os.path.join(tf_dir, "identity")):
        tid = read_tf_identity(tf_dir)
        if digest_hex and digest_hex.lower() != tid["digest"].lower():
            raise TensorFoldError("--digest differs from %s/identity" % tf_dir)
        digest_hex, vocab, name = tid["digest"], tid["vocab"], name or tid.get("name", "")
        write_identity = False
    else:
        if not digest_hex or len(digest_hex) != 64 or vocab is None:
            raise TensorFoldError("need TensorFold's full digest (64 hex) and vocab: start TensorFold once with "
                                  "--recall-dir %s (it writes identity), or pass --digest/--vocab" % tf_dir)
        write_identity = True
    vocab = int(vocab)
    ov = ours.get("vocab_size")
    if ov is not None and int(ov) != vocab:
        raise TensorFoldError("vocab mismatch: our record %s, TensorFold %d" % (ov, vocab))
    os.makedirs(tf_dir, mode=0o700, exist_ok=True)
    if any(_TF_NAME.match(n) for n in os.listdir(tf_dir)):
        raise TensorFoldError("%s already holds TensorFold chunk files: empty it first (DELETE /v1/recall)" % tf_dir)
    digest = bytes.fromhex(digest_hex)
    head = tf_header(vocab, digest)
    segs = read_our_segments(our_dir, chunk_ids)
    kept, dropped, ids = [], 0, 0
    for s in segs:
        if s.size and int(s.min()) >= 0 and int(s.max()) < vocab:
            kept.append(s)
            ids += int(s.size)
        else:
            dropped += 1
    pieces = []
    for s in kept:
        pieces.append(s.astype(np.uint32))
        pieces.append(np.array([MARK], dtype=np.uint32))
    stream = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.uint32)
    files = 0
    for seq, part in _chunks_of(stream, chunk_ids):
        p = os.path.join(tf_dir, "%016x.ids" % seq)
        with open(p, "wb") as f:
            f.write(head)
            f.write(part.astype("<u4").tobytes())
        os.chmod(p, 0o600)
        files += 1
    if write_identity:
        with open(os.path.join(tf_dir, "identity"), "w", encoding="utf-8", newline="\n") as f:
            f.write(tf_identity_text(digest_hex, vocab, name or ours.get("model") or ""))
    return {"segments": len(kept), "dropped_segments": dropped, "ids": ids, "files": files, "dir": tf_dir,
            "stream_ids": int(stream.size)}


def read_tf_segments(tf_dir, chunk_ids=TF_CHUNK):
    """TensorFold's record as segments, read like recall_store.zig ``load``. Returns (segments, vocab, digest16)."""
    seqs = sorted(int(m.group(1), 16) for m in (_TF_NAME.match(n) for n in os.listdir(tf_dir)) if m)
    head0 = None
    parts = []
    for k, seq in enumerate(seqs):
        if seq != seqs[0] + k:
            break  # a gap: the files past it are not read
        with open(os.path.join(tf_dir, "%016x.ids" % seq), "rb") as f:
            data = f.read()
        if len(data) < HEAD_BYTES or data[:8] != MAGIC:
            raise TensorFoldError("%016x.ids: bad header" % seq)
        ver, voc = struct.unpack("<II", data[8:16])
        if ver != VERSION:
            raise TensorFoldError("%016x.ids: version %d, expected %d" % (seq, ver, VERSION))
        if head0 is None:
            head0 = data[:HEAD_BYTES]
        elif data[:HEAD_BYTES] != head0:
            raise TensorFoldError("%016x.ids: header differs from the first file's" % seq)
        body = data[HEAD_BYTES:]
        arr = np.frombuffer(body[: len(body) // 4 * 4], dtype="<u4")
        if arr.size > chunk_ids:
            raise TensorFoldError("%016x.ids: more than %d ids" % (seq, chunk_ids))
        parts.append(arr)
        if arr.size < chunk_ids:
            break  # a short file is the last
    if head0 is None:
        return [], None, None
    vocab = struct.unpack("<I", head0[12:16])[0]
    stream = np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, dtype=np.int64)
    segs = _split(stream, MARK)
    bad = [s for s in segs if s.size and int(s.max()) >= vocab]
    if bad:
        raise TensorFoldError("TensorFold record holds ids >= vocab %d" % vocab)
    return segs, vocab, head0[16:32]


def _read_our_identity(our_dir):
    p = os.path.join(our_dir, "identity")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def import_from_tf(tf_dir, our_dir, identity: dict, chunk_ids=TF_CHUNK) -> dict:
    """Write TensorFold's record as our chunk files into ``our_dir`` (no chunk files there yet).

    ``identity`` is ours ({tokenizer_digest, vocab_size, model}, e.g. core.tokenizer_identity(...)); its
    vocab_size must equal TensorFold's header vocab.
    """
    segs, vocab, _ = read_tf_segments(tf_dir, chunk_ids)
    if vocab is None:
        raise TensorFoldError("%s holds no TensorFold chunk files" % tf_dir)
    if identity.get("vocab_size") is None or int(identity["vocab_size"]) != int(vocab):
        raise TensorFoldError("vocab mismatch: TensorFold %d, ours %s" % (vocab, identity.get("vocab_size")))
    os.makedirs(our_dir, mode=0o700, exist_ok=True)
    if any(_OUR_NAME.match(n) for n in os.listdir(our_dir)):
        raise TensorFoldError("%s already holds chunk files" % our_dir)
    old = _read_our_identity(our_dir)
    if old and (old.get("tokenizer_digest") != identity.get("tokenizer_digest")
                or int(old.get("vocab_size", -1)) != int(identity["vocab_size"])):
        raise TensorFoldError("%s/identity names another tokenizer" % our_dir)
    pieces = []
    ids = 0
    for s in segs:
        pieces.append(s.astype(np.int32))
        pieces.append(np.array([-1], dtype=np.int32))
        ids += int(s.size)
    stream = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int32)
    files = 0
    for seq, part in _chunks_of(stream, chunk_ids):
        p = os.path.join(our_dir, "chunk-%08d.ids" % seq)
        with open(p, "wb") as f:
            f.write(part.astype("<i4").tobytes())
        files += 1
    if not old:
        with open(os.path.join(our_dir, "identity"), "w", encoding="utf-8") as f:
            f.write(json.dumps(dict(identity), ensure_ascii=False, sort_keys=True) + "\n")
    return {"segments": len(segs), "ids": ids, "files": files, "dir": our_dir, "vocab": int(vocab)}


# --------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="yee tensorfold")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("detect")
    p.add_argument("--binary", default="tensorfold")
    p = sub.add_parser("args", help="flags for `tensorfold serve` that turn native Recall on")
    p.add_argument("--binary", default="tensorfold")
    p.add_argument("--gib", type=float, default=0.25)
    p.add_argument("--dir", default=None)
    p.add_argument("--read-only", action="store_true")
    p.add_argument("--no-detect", action="store_true", help="skip the binary check")
    for name in ("status", "clear"):
        p = sub.add_parser(name)
        p.add_argument("--url", default="http://127.0.0.1:8080")
    p = sub.add_parser("export", help="our record -> TensorFold chunk files")
    p.add_argument("--from", dest="src", required=True)
    p.add_argument("--to", dest="dst", required=True)
    p.add_argument("--digest", default=None)
    p.add_argument("--vocab", type=int, default=None)
    p.add_argument("--name", default=None)
    p = sub.add_parser("import", help="TensorFold chunk files -> our record")
    p.add_argument("--from", dest="src", required=True)
    p.add_argument("--to", dest="dst", required=True)
    p.add_argument("--tokenizer-dir", default=None, help="our identity from this tokenizer directory")
    p.add_argument("--identity-json", default=None, help="our identity as a JSON file")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "detect":
            print(json.dumps(detect(a.binary), indent=1, ensure_ascii=False))
        elif a.cmd == "args":
            det = None if a.no_detect else detect(a.binary)
            print(" ".join(launch_args(a.gib, a.dir, a.read_only, det)))
            # stdout stays the bare flag list ($(yee tensorfold args) keeps working); the env line goes to stderr
            for k, v in sorted(launch_env().items()):
                print("export %s=%s" % (k, v), file=sys.stderr)
        elif a.cmd == "status":
            print(json.dumps(status(a.url), indent=1, ensure_ascii=False))
        elif a.cmd == "clear":
            print(json.dumps(clear(a.url), indent=1, ensure_ascii=False))
        elif a.cmd == "export":
            print(json.dumps(export_to_tf(a.src, a.dst, a.digest, a.vocab, a.name), indent=1))
        elif a.cmd == "import":
            if a.identity_json:
                with open(a.identity_json, "r", encoding="utf-8") as f:
                    ident = json.load(f)
            elif a.tokenizer_dir:
                from .sglang_dspark import _digest_from_dir

                dg, voc, found = _digest_from_dir(a.tokenizer_dir)
                if not found:
                    raise TensorFoldError("no tokenizer files under %s" % a.tokenizer_dir)
                ident = {"tokenizer_digest": dg, "vocab_size": voc, "model": os.path.basename(a.tokenizer_dir.rstrip("/"))}
            else:
                raise TensorFoldError("import needs --tokenizer-dir or --identity-json")
            print(json.dumps(import_from_tf(a.src, a.dst, ident), indent=1))
    except TensorFoldError as e:
        print("tensorfold: %s" % (e,), file=sys.stderr)
        return 2
    return 0
