"""Plain-assert tests for yee.core. Run: python tests/test_recall_core.py

The record tests run against every backend that is available on this machine (disk always, memory always,
cuda only when torch and a CUDA device are present).
"""
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from yee import core as rc  # noqa: E402

IDENT = {"tokenizer_digest": "a" * 64, "vocab_size": 129280, "model": "test"}
RESULTS = []


def rnd(n, seed, lo=10, hi=120000):
    return np.random.default_rng(seed).integers(lo, hi, size=n, dtype=np.int64)


class TmpDir:
    def __enter__(self):
        self.d = tempfile.mkdtemp(prefix="recall_t_")
        return os.path.join(self.d, "rec")

    def __exit__(self, *a):
        shutil.rmtree(self.d, ignore_errors=True)


def cuda_ok():
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


CUDA_OK = cuda_ok()
# the backends the record tests run on
BACKENDS = ["disk", "memory"] + (["cuda"] if CUDA_OK else [])


def backends():
    """Every runnable backend; cuda is skipped when torch / CUDA are missing."""
    out = []
    for b in BACKENDS:
        if b == "cuda" and not CUDA_OK:
            out.append("SKIP (no torch/CUDA on this machine)")
            break
        out.append(b)
    return out


def make_record(backend, directory=None, **kw):
    if backend == "disk":
        return rc.RecallRecord(directory, IDENT, **kw)
    return rc.RecallRecord(None, IDENT, backend=backend, **kw)


def test(fn):
    RESULTS.append(fn)
    return fn
test.__test__ = False  # decorator, not a test (pytest)


# ------------------------------------------------------------------ record tests, per backend
@test
def basic_add_lookup():
    for b in backends():
        if isinstance(b, str) and b.startswith("SKIP"):
            continue
        with TmpDir() as d:
            r = make_record(b, d, chunk_ids=4096)
            s = rnd(2000, 1)
            assert r.add(s) == 2000, b
            for i in (16, 40, 777, 1500, 1994):
                tail = s[max(0, i - 40):i]
                hit = r.lookup(tail)
                assert hit is not None, (b, i)
                p, L = hit
                assert L == min(40, i), (b, i, L)
                c = r.continuation(p, 5, min_n=1)
                assert np.array_equal(c, s[i:i + 5]), (b, i)
            assert r.lookup(rnd(40, 99)) is None, b
            r.close()


@test
def sentinel_boundary():
    for b in backends():
        if isinstance(b, str) and b.startswith("SKIP"):
            continue
        with TmpDir() as d:
            r = make_record(b, d, chunk_ids=4096)
            a, bb = rnd(100, 2), rnd(100, 3)
            r.add(a)
            r.add(bb)
            assert r.lookup(a[-16:]) is None  # nothing follows it: not indexed (MIN_TAIL)
            hit = r.lookup(a[-25:-5])
            assert hit is not None
            assert np.array_equal(r.continuation(hit[0], 5, min_n=1), a[-5:])
            assert r.continuation(hit[0], 6) is None  # would run into the sentinel
            seam = r.lookup(np.concatenate((a[-8:], bb[:8])))
            if rc.SHORT_WINDOWS:  # the 8-id fallback may find bb's own start, never a window across the seam
                assert seam is None or np.array_equal(r.continuation(seam[0], 5), bb[8:13])
            else:
                assert seam is None
            hit = r.lookup(bb[:20])
            assert hit is not None and np.array_equal(r.continuation(hit[0], 5), bb[20:25])
            r.close()


@test
def dedup_same_twice():
    for b in backends():
        if isinstance(b, str) and b.startswith("SKIP"):
            continue
        with TmpDir() as d:
            r = make_record(b, d, chunk_ids=4096)
            s = rnd(1000, 4)
            assert r.add(s) == 1000, b
            end = r.end_pos
            assert r.add(s) == 0, b
            assert r.end_pos == end, b
            r.close()


@test
def partial_overlap():
    for b in backends():
        if isinstance(b, str) and b.startswith("SKIP"):
            continue
        with TmpDir() as d:
            r = make_record(b, d, chunk_ids=4096)
            x, z = rnd(1000, 5), rnd(300, 6)
            r.add(x)
            y = np.concatenate((x[200:700], z))
            added = r.add(y)
            assert added == 300 + rc.OVERLAP, (b, added)
            hit = r.lookup(y[480:510])  # x[680:700] + z[:10]
            assert hit is not None, b
            assert np.array_equal(r.continuation(hit[0], 5), z[10:15]), b
            r.close()


@test
def chunk_rotation():
    for b in backends():
        if isinstance(b, str) and b.startswith("SKIP"):
            continue
        with TmpDir() as d:
            r = make_record(b, d, chunk_ids=256)
            s = rnd(1000, 7)
            r.add(s)
            if b == "disk":
                files = sorted(f for f in os.listdir(d) if f.startswith("chunk-"))
                assert len(files) == 4, files
            for i in (250, 256, 260, 511, 515, 900):
                hit = r.lookup(s[i - 30:i])
                assert hit is not None, (b, i)
                assert np.array_equal(r.continuation(hit[0], 5, min_n=1), s[i:i + 5]), (b, i)
            r.close()


@test
def capacity_eviction():
    for b in backends():
        if isinstance(b, str) and b.startswith("SKIP"):
            continue
        with TmpDir() as d:
            gib = 512 * 4 / float(1 << 30)  # 512 ids
            r = make_record(b, d, gib=gib, chunk_ids=256)
            segs = [rnd(300, 100 + k) for k in range(10)]
            for sg in segs:
                r.add(sg)
            assert r.end_pos - r.base_pos <= 512 + 256, (b, r.end_pos, r.base_pos)
            if b == "disk":
                files = [f for f in os.listdir(d) if f.startswith("chunk-")]
                assert len(files) <= 3, files
            assert r.lookup(segs[0][:40]) is None, b
            hit = r.lookup(segs[-1][200:240])
            assert hit is not None and np.array_equal(r.continuation(hit[0], 5), segs[-1][240:245]), b
            assert r.evicted_chunks > 0, b
            r.close()


@test
def reopen_persists():
    for b in backends():
        if isinstance(b, str) and b.startswith("SKIP"):
            continue
        if b != "disk":
            continue  # memory/cuda have no on-disk reopen; the snapshot path is tested below
        with TmpDir() as d:
            s = rnd(3000, 8)
            r = make_record(b, d, chunk_ids=1024)
            r.add(s)
            r.close()
            r2 = make_record(b, d, chunk_ids=1024)
            hit = r2.lookup(s[1000:1050])
            assert hit is not None and hit[1] == 50
            assert np.array_equal(r2.continuation(hit[0], 5), s[1050:1055])
            assert r2.add(s) == 0  # still deduplicated after reload
            r2.close()


@test
def identity_mismatch():
    with TmpDir() as d:
        rc.RecallRecord(d, IDENT, chunk_ids=1024).close()
        other = dict(IDENT, tokenizer_digest="b" * 64)
        try:
            rc.RecallRecord(d, other, chunk_ids=1024)
        except rc.RecallIdentityError as e:
            assert "aaaaaaaaaaaaaaaa" in str(e)
        else:
            raise AssertionError("expected RecallIdentityError")
        rc.RecallRecord(d, dict(IDENT, model="another"), chunk_ids=1024).close()  # same tokenizer: shared


@test
def second_writer_reads_only():
    if rc.fcntl is None:
        return "SKIP (no fcntl on this OS)"
    with TmpDir() as d:
        r1 = rc.RecallRecord(d, IDENT, chunk_ids=1024)
        s = rnd(500, 9)
        r1.add(s)
        r2 = rc.RecallRecord(d, IDENT, chunk_ids=1024)
        assert r1.writer and not r2.writer
        assert r2.add(rnd(500, 10)) == 0
        assert r2.lookup(s[100:140]) is not None
        try:
            r2.clear()
        except PermissionError:
            pass
        else:
            raise AssertionError("read-only clear must refuse")
        r2.close()
        r1.close()


@test
def clear_deletes():
    for b in backends():
        if isinstance(b, str) and b.startswith("SKIP"):
            continue
        with TmpDir() as d:
            r = make_record(b, d, chunk_ids=256)
            s = rnd(1000, 11)
            r.add(s)
            r.clear()
            if b == "disk":
                assert not [f for f in os.listdir(d) if f.startswith("chunk-")]
                assert os.path.exists(os.path.join(d, "identity"))
            assert r.lookup(s[100:140]) is None, b
            assert r.add(s) == 1000, b
            assert r.lookup(s[100:140]) is not None, b
            r.close()


@test
def low_disk_skips():
    with TmpDir() as d:
        r = rc.RecallRecord(d, IDENT, chunk_ids=1024, min_free_gib=4.0)
        real = rc.shutil.disk_usage

        class U:
            total, used, free = 100 << 30, 99 << 30, 1 << 30

        rc.shutil.disk_usage = lambda p: U()
        try:
            assert r.add(rnd(500, 12)) == 0
            assert r.skipped_low_disk == 1
        finally:
            rc.shutil.disk_usage = real
        assert r.add(rnd(500, 12)) == 500
        r.close()


@test
def read_only_flag():
    with TmpDir() as d:
        s = rnd(600, 13)
        r = rc.RecallRecord(d, IDENT, chunk_ids=1024)
        r.add(s)
        r.close()
        ro = rc.RecallRecord(d, IDENT, chunk_ids=1024, read_only=True)
        assert not ro.writer and ro.add(rnd(600, 14)) == 0
        assert ro.lookup(s[200:240]) is not None
        try:
            ro.clear()
        except PermissionError:
            pass
        else:
            raise AssertionError("expected PermissionError")
        ro.close()


@test
def memory_snapshot_roundtrip():
    """memory / cuda backends: save() then load() keeps lookup and continuation."""
    for b in backends():
        if isinstance(b, str) and b.startswith("SKIP"):
            continue
        if b == "disk":
            continue
        with TmpDir() as d:
            s = rnd(1200, 41)
            r = make_record(b, None, chunk_ids=256)
            r.add(s)
            snap = os.path.join(d, "snap.npz")
            r.save(snap)
            r.close()
            r2 = rc.RecallRecord.load(snap, IDENT, backend=b)
            hit = r2.lookup(s[500:540])
            assert hit is not None, b
            assert np.array_equal(r2.continuation(hit[0], 5), s[540:545]), b
            assert r2.add(s) == 0, b  # still deduplicated after the reload
            r2.close()


@test
def memory_snapshot_identity_guard():
    with TmpDir() as d:
        r = rc.RecallRecord(None, IDENT, backend="memory", chunk_ids=256)
        r.add(rnd(300, 42))
        snap = os.path.join(d, "snap.npz")
        r.save(snap)
        r.close()
        other = dict(IDENT, tokenizer_digest="b" * 64)
        try:
            rc.RecallRecord.load(snap, other, backend="memory")
        except rc.RecallIdentityError:
            pass
        else:
            raise AssertionError("expected RecallIdentityError")


@test
def cuda_backend_missing_is_explicit():
    if CUDA_OK:
        return "SKIP (CUDA present)"
    try:
        rc.RecallRecord(None, IDENT, backend="cuda")
    except rc.RecallBackendError as e:
        assert "cuda" in str(e).lower(), str(e)
    else:
        raise AssertionError("expected RecallBackendError")


# ------------------------------------------------------------------ engine tests
def _decode(eng, rid, prompt, reply, step_tokens=3, own_accept=2.0, verify=True):
    """Simulate a DSpark + Recall decode loop; returns (rounds, copy_rounds, landed)."""
    eng.begin_stream(rid, prompt)
    pos, rounds, copies, landed = 0, 0, 0, 0
    g = eng.gamma
    while pos < len(reply):
        rounds += 1
        prop = eng.propose(rid)
        if prop is not None:
            truth = reply[pos:pos + g]
            k = 0
            while k < min(len(truth), prop.real) and prop.tokens[k] == truth[k]:
                k += 1
            if verify and prop.real >= 1 and len(truth) == g:
                pass
            eng.feedback(rid, k)
            copies += 1
            landed += k
            adv = k + 1
        else:
            eng.observe_own(rid, own_accept)
            adv = step_tokens
        eng.extend_stream(rid, reply[pos:pos + adv])
        pos += adv
    eng.end_stream(rid, reply)
    return rounds, copies, landed


@test
def engine_second_pass_copies():
    for b in backends():
        if isinstance(b, str) and b.startswith("SKIP"):
            continue
        with TmpDir() as d:
            r = make_record(b, d, chunk_ids=1 << 14)
            eng = rc.RecallEngine(r, gamma=5, async_add=False)
            prompt, reply = rnd(200, 20), rnd(300, 21)
            r1, c1, l1 = _decode(eng, "a", prompt, reply)
            assert c1 == 0, (b, c1)  # nothing to copy on a fresh record
            r2, c2, l2 = _decode(eng, "b", prompt, reply)
            assert c2 >= r2 - 2, (b, r2, c2)
            assert l2 >= 0.9 * 5 * c2 - 5, (b, l2, c2)
            assert r2 < r1 * 0.75, (b, r1, r2)
            eng.close()


@test
def engine_async_second_pass():
    with TmpDir() as d:
        r = rc.RecallRecord(d, IDENT, chunk_ids=1 << 14)
        eng = rc.RecallEngine(r, gamma=5, async_add=True)
        prompt, reply = rnd(5000, 22), rnd(800, 23)
        _decode(eng, "a", prompt, reply)
        eng.flush()
        r2, c2, _ = _decode(eng, "b", prompt, reply)
        assert c2 >= r2 - 2, (r2, c2)
        eng.close()


@test
def engine_cooldown_after_miss():
    with TmpDir() as d:
        r = rc.RecallRecord(d, IDENT, chunk_ids=1 << 14)
        eng = rc.RecallEngine(r, gamma=5, async_add=False)
        s = rnd(400, 24)
        r.add(s)
        eng.begin_stream("c", s[:100])
        prop = eng.propose("c")
        assert prop is not None and prop.source == "record"
        eng.feedback("c", 0)
        got = [eng.propose("c") for _ in range(5)]
        assert got[:4] == [None] * 4, got
        assert got[4] is not None
        eng.close()


@test
def engine_probe_backoff():
    with TmpDir() as d:
        r = rc.RecallRecord(d, IDENT, chunk_ids=1 << 14)
        eng = rc.RecallEngine(r, gamma=5, async_add=False, cooldown_rounds=0, probe_start=4, probe_max=64)
        s = rnd(400, 25)
        r.add(s)
        eng.begin_stream("p", s[:100])
        for _ in range(4):
            eng.observe_own("p", 4.0)
        for t in range(3):  # push every record tier's expectation down
            st = eng._stats[("record", t)]
            st.n, st.s = 50.0, 0.0
        offers = []
        for i in range(200):
            prop = eng.propose("p", own_ema=4.0)
            if prop is not None:
                assert prop.probe
                offers.append(i)
                eng.feedback("p", 0)
        gaps = np.diff(offers).tolist()
        assert offers and offers[0] == 3, offers
        assert gaps[:4] == [4 * 2, 4 * 4, 4 * 8, 4 * 16], gaps
        eng.close()


@test
def engine_own_wins_tie():
    with TmpDir() as d:
        r = rc.RecallRecord(d, IDENT, chunk_ids=1 << 14)
        eng = rc.RecallEngine(r, gamma=5, async_add=False)
        s = rnd(60, 26)
        r.add(s)
        eng.begin_stream("o", rnd(20, 27))
        eng.extend_stream("o", s)       # first occurrence in the stream's own output
        eng.extend_stream("o", s[:30])  # tail s[14:30] occurs earlier in own context and in the record
        prop = eng.propose("o")
        assert prop is not None and prop.source == "own", prop
        assert np.array_equal(prop.tokens, s[30:35])
        eng.close()


@test
def engine_periodic_own_copy():
    with TmpDir() as d:
        r = rc.RecallRecord(d, IDENT, chunk_ids=1 << 14)
        eng = rc.RecallEngine(r, gamma=5, async_add=False)
        pat = [7, 8, 9]
        eng.begin_stream("q", rnd(20, 28))
        eng.extend_stream("q", pat * 8)
        prop = eng.propose("q")
        assert prop is not None and prop.source == "own"
        assert prop.tokens.tolist() == [7, 8, 9, 7, 8], prop.tokens
        eng.close()


@test
def env_singleton_and_stats():
    with TmpDir() as d:
        env = {"SGLANG_D_RECALL": "1", "SGLANG_D_RECALL_DIR": d, "SGLANG_D_RECALL_GIB": "0.01"}
        rc._ENGINE = None
        assert rc.engine_from_env(5, IDENT, environ={"SGLANG_D_RECALL": "0"}) is None
        eng = rc.engine_from_env(5, IDENT, environ=env)
        assert eng is not None and rc.engine_from_env(5, IDENT, environ=env) is eng
        eng.record.add(rnd(300, 29))
        eng.tick(now=1e9)
        eng.flush()
        import json
        with open(os.path.join(d, "stats.json"), encoding="utf-8") as f:
            st = json.load(f)
        assert st["ids"] >= 300 and st["state"] == "writing"
        open(os.path.join(d, "clear.request"), "w").close()
        eng.tick(now=2e9)
        eng.flush()
        assert eng.record.end_pos - eng.record.base_pos == 0
        assert eng.last_clear.startswith("ok"), eng.last_clear
        eng.close()
        rc._ENGINE = None


@test
def env_new_name_wins_old_name_is_fallback():
    """The new SGLANG_D_RECALL* names win; the old DSV41_RECALL* names still work as a fallback."""
    with TmpDir() as d:
        rc._ENGINE = None
        env = {"SGLANG_D_RECALL": "1", "SGLANG_D_RECALL_DIR": d, "SGLANG_D_RECALL_GIB": "0.01",
               "SGLANG_D_RECALL_BACKEND": "memory",
               "DSV41_RECALL": "0", "DSV41_RECALL_DIR": "/nonexistent", "RECALL_BACKEND": "disk"}
        eng = rc.engine_from_env(5, IDENT, environ=env)
        assert eng is not None, "the new names did not switch the engine on"
        assert eng.record.backend == "memory", eng.record.backend
        eng.close()
        rc._ENGINE = None
        # only the old names set: they must still work
        env = {"DSV41_RECALL": "1", "DSV41_RECALL_DIR": d, "DSV41_RECALL_GIB": "0.01",
               "RECALL_BACKEND": "memory"}
        eng = rc.engine_from_env(5, IDENT, environ=env)
        assert eng is not None, "the old names did not switch the engine on"
        assert eng.record.backend == "memory", eng.record.backend
        eng.close()
        rc._ENGINE = None
        # a blank value counts as "not set": the fallback and the default still apply
        assert rc._env("SGLANG_D_RECALL_GIB", "DSV41_RECALL_GIB", "0.0625", {}) == "0.0625"
        assert rc._env("SGLANG_D_RECALL_GIB", "DSV41_RECALL_GIB", "0.0625",
                       {"SGLANG_D_RECALL_GIB": " ", "DSV41_RECALL_GIB": "0.5"}) == "0.5"


@test
def env_backend_switch():
    """SGLANG_D_RECALL_BACKEND picks the store; SGLANG_D_RECALL_DEVICE is only used by the cuda backend."""
    with TmpDir() as d:
        rc._ENGINE = None
        env = {"SGLANG_D_RECALL": "1", "SGLANG_D_RECALL_DIR": d, "SGLANG_D_RECALL_GIB": "0.01",
               "SGLANG_D_RECALL_BACKEND": "memory"}
        eng = rc.engine_from_env(5, IDENT, environ=env)
        assert eng is not None and eng.record.backend == "memory", eng.record.backend
        eng.close()
        rc._ENGINE = None
        env["SGLANG_D_RECALL_BACKEND"] = "disk"
        eng = rc.engine_from_env(5, IDENT, environ=env)
        assert eng is not None and eng.record.backend == "disk", eng.record.backend
        eng.close()
        rc._ENGINE = None


@test
def engine_overlap_fifo():
    with TmpDir() as d:
        r = rc.RecallRecord(d, IDENT, chunk_ids=1 << 14)
        eng = rc.RecallEngine(r, gamma=5, async_add=False)
        s = rnd(400, 32)
        r.add(s)
        eng.begin_stream("f", s[:100])
        a = eng.propose("f", n=11)   # step k launched
        eng.extend_stream("f", s[100:103])
        b = eng.propose("f", n=11)   # step k+1 launched before k's result is processed
        assert a is not None and b is not None
        assert np.array_equal(a.tokens, s[100:111]) and np.array_equal(b.tokens, s[103:114])
        assert eng.stream_out_len("f") == 3
        eng.feedback("f", 5)          # result of step k
        eng.feedback("f", 0)          # result of step k+1: miss -> cooldown
        assert eng._stats[("record", 2)].offers + eng._stats[("record", 1)].offers >= 2
        assert eng.propose("f") is None
        eng.drop_pending("f")
        eng.end_stream("f", s[100:103])
        assert eng.stream_out_len("f") == -1
        eng.close()


@test
def perf_and_memory():
    with TmpDir() as d:
        r = rc.RecallRecord(d, IDENT)  # default 64 MiB budget, 1M-id chunks
        big = rnd(1_000_000, 30)
        t0 = time.perf_counter()
        r.add(big)
        t_big = time.perf_counter() - t0
        new = rnd(100_000, 31)
        t0 = time.perf_counter()
        r.add(new)
        t_add = time.perf_counter() - t0
        eng = rc.RecallEngine(r, gamma=5, async_add=False)
        eng.begin_stream("x", big[5000:6000])
        lat = []
        nxt = 6000
        for i in range(10_000):
            if i % 2:
                eng.extend_stream("x", big[nxt:nxt + 1])  # stays on recorded text: lookups hit
                nxt += 1
            t0 = time.perf_counter()
            prop = eng.propose("x", own_ema=0.0)
            lat.append(time.perf_counter() - t0)
            if prop is not None:
                eng.feedback("x", prop.real)
        lat = np.array(lat) * 1e6
        p50, p99 = float(np.percentile(lat, 50)), float(np.percentile(lat, 99))
        mem = r.stats()["bytes_mem"] / (1 << 20)
        print("    perf: add 1M=%.0f ms, add 100K=%.0f ms, propose p50=%.1f us p99=%.1f us, mem=%.1f MiB, offers=%d"
              % (t_big * 1e3, t_add * 1e3, p50, p99, mem, eng.offers))
        assert t_add < 0.3, t_add
        assert p50 < 100 and p99 < 1000, (p50, p99)
        short_mib = sum(sk.nbytes + sp.nbytes for _w, sk, sp in r._short) / (1 << 20)
        assert mem < 100 + short_mib, (mem, short_mib)
        assert eng.offers > 4000
        eng.close()


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
    print("numpy %s, python %s, backends=%s" % (np.__version__, sys.version.split()[0], ",".join(BACKENDS)))
    if not CUDA_OK:
        print("SKIP cuda backend: no torch/CUDA on this machine")
        skip += 1
    if fail:
        print("FAILED %d / %d (skip %d)" % (fail, ok + fail, skip))
        sys.exit(1)
    print("ALL PASS n=%d skip=%d" % (ok, skip))


if __name__ == "__main__":
    main()
