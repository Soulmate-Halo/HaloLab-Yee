"""Yee 3.6.0: several processes share one disk record directory (yee.core and vllm_d_recall.core).

Up to 3.5.2 the first process to open a directory held the lock for good and was the only writer; every
other process could not write and read the directory only once, when it opened it, so it never saw what
the writer appended later. With the shared record (RECALL_ACCEL_SHARED / VLLM_D_RECALL_SHARED, default 1)
every process appends under a short exclusive lock and reads what the others appended (refresh(), called
by RecallEngine.tick() every 2 s). Shared off gives the 3.5.2 behaviour back.

Each test drives real processes (one interpreter per process, commands over a pipe). Needs fcntl:
on Windows every test is a SKIP.

Run: python tests/test_recall_shared.py (pytest also collects the test_* functions)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from yee import core as rc  # noqa: E402

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(rc.__file__)))
MODS = ("yee.core", "vllm_d_recall.core")
SHARED_ENV = {"yee.core": "RECALL_ACCEL_SHARED", "vllm_d_recall.core": "VLLM_D_RECALL_SHARED"}
CLEAR = ("YEE", "RECALL_ACCEL", "SGLANG_D_RECALL", "VLLM_D_RECALL", "DSV41_RECALL", "RECALL_")
RESULTS = []

WORKER = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import importlib
rc = importlib.import_module(sys.argv[2])
import numpy as np
IDENT = {"tokenizer_digest": "a" * 64, "vocab_size": 129280, "model": "test"}
rec = None
eng = None

def seq(seed, n):
    return np.random.default_rng(seed).integers(10, 120000, size=n, dtype=np.int64)

def has(seed, n):
    a = seq(seed, n)
    r = rec.lookup(a[100:140])
    if r is None:
        return False
    c = rec.continuation(r[0], 8)
    return c is not None and len(c) >= 8 and [int(x) for x in c[:8]] == [int(x) for x in a[140:148]]

for line in sys.stdin:
    cmd = json.loads(line)
    op = cmd["op"]
    try:
        out = {}
        if op == "open":
            if "lock_wait" in cmd:
                rc.SHARED_LOCK_WAIT = float(cmd["lock_wait"])
            kw = {"chunk_ids": int(cmd.get("chunk_ids", 4096))}
            for k in ("shared", "gib", "evict"):
                if k in cmd:
                    kw[k] = cmd[k]
            rec = rc.RecallRecord(cmd["dir"], IDENT, **kw)
            out = {"writer": bool(rec.writer), "shared": bool(getattr(rec, "shared", False)),
                   "ids": int(rec.end_pos - rec.base_pos), "err": rec.last_error}
        elif op == "add":
            out = {"added": int(rec.add(seq(cmd["seed"], cmd.get("n", 600))))}
        elif op == "addmany":
            out = {"added": sum(int(rec.add(seq(s, cmd.get("n", 600)))) for s in cmd["seeds"])}
        elif op == "refresh":
            out = {"new": int(rec.refresh())}
        elif op == "tick":
            if eng is None:
                eng = rc.RecallEngine(rec, 4, async_add=False)
            eng.tick(now=float(cmd.get("now", 1000.0)))
        elif op == "has":
            out = {"hits": sum(1 for s in cmd["seeds"] if has(s, cmd.get("n", 600)))}
        elif op == "digest":
            import hashlib
            ids = rec._read(rec.base_pos, rec.end_pos)
            out = {"n": int(ids.size), "sha": hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()}
        elif op == "stats":
            st = rec.stats()
            out = {k: st.get(k) for k in ("ids", "writer", "shared", "state", "added_ids", "synced_ids",
                                          "shared_resets", "lock_timeouts", "chunks", "last_error")}
        elif op == "clear":
            rec.clear()
        elif op == "close":
            rec.close()
        out["ok"] = True
    except Exception as e:
        out = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    print(json.dumps(out), flush=True)
    if op == "close":
        break
'''


class Worker(object):
    """One process holding one RecallRecord; every call sends one command and waits for its answer."""

    def __init__(self, mod, env=None):
        e = {k: v for k, v in os.environ.items() if not k.startswith(CLEAR)}
        e.update(env or {})
        self.p = subprocess.Popen([sys.executable, "-c", WORKER, PKG_ROOT, mod], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=e)

    def send(self, **cmd):
        self.p.stdin.write(json.dumps(cmd) + "\n")
        self.p.stdin.flush()

    def recv(self):
        line = self.p.stdout.readline()
        if not line:
            raise AssertionError("worker died: %s" % self.p.stderr.read()[-3000:])
        r = json.loads(line)
        if not r.get("ok"):
            raise AssertionError("worker: %s" % r.get("error"))
        return r

    def __call__(self, **cmd):
        self.send(**cmd)
        return self.recv()

    def close(self):
        if self.p.poll() is None:
            try:
                self(op="close")
            except Exception:
                pass
            try:
                self.p.stdin.close()
            except OSError:
                pass
            try:
                self.p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.p.kill()


class TmpDir(object):
    def __enter__(self):
        self.d = tempfile.mkdtemp(prefix="recall_sh_")
        return os.path.join(self.d, "rec")

    def __exit__(self, *a):
        shutil.rmtree(self.d, ignore_errors=True)


def test(fn):
    RESULTS.append(fn)
    return fn


test.__test__ = False  # decorator, not a test (pytest)


def _no_fcntl():
    return "SKIP (no fcntl on this OS)" if rc.fcntl is None else None


def _chunks_whole(d):
    for name in os.listdir(d):
        if name.startswith("chunk-"):
            assert os.path.getsize(os.path.join(d, name)) % 4 == 0, name


# ------------------------------------------------------------------ tests (each for both cores)
@test
def shared_two_processes():
    """B opens before A writes; B sees A's later ids after a refresh, and A sees B's."""
    if _no_fcntl():
        return _no_fcntl()
    for mod in MODS:
        with TmpDir() as d:
            a, b = Worker(mod), Worker(mod)
            try:
                ra, rb = a(op="open", dir=d), b(op="open", dir=d)
                assert ra["shared"] and rb["shared"] and ra["writer"] and rb["writer"], (mod, ra, rb)
                assert a(op="add", seed=1)["added"] == 600, mod
                assert b(op="refresh")["new"] >= 600, mod
                assert b(op="has", seeds=[1])["hits"] == 1, mod
                assert b(op="add", seed=2)["added"] == 600, mod
                assert a(op="refresh")["new"] >= 600, mod
                assert a(op="has", seeds=[1, 2])["hits"] == 2, mod
                sa, sb = a(op="stats"), b(op="stats")
                assert sa["ids"] == sb["ids"] and sa["shared"] and sb["shared"], (mod, sa, sb)
                _chunks_whole(d)
            finally:
                a.close()
                b.close()


@test
def unshared_keeps_old_behaviour():
    """shared off (argument or environment): the 3.5.2 single writer, and its bug, come back."""
    if _no_fcntl():
        return _no_fcntl()
    for mod in MODS:
        for how in ("arg", "env"):
            with TmpDir() as d:
                env = {SHARED_ENV[mod]: "0"} if how == "env" else None
                kw = {"shared": False} if how == "arg" else {}
                a, b = Worker(mod, env), Worker(mod, env)
                try:
                    ra, rb = a(op="open", dir=d, **kw), b(op="open", dir=d, **kw)
                    assert ra["writer"] and not rb["writer"], (mod, how, ra, rb)
                    assert not ra["shared"] and not rb["shared"], (mod, how, ra, rb)
                    assert a(op="add", seed=1)["added"] == 600
                    assert b(op="refresh")["new"] == 0
                    assert b(op="has", seeds=[1])["hits"] == 0, (mod, how)  # never re-reads the directory
                    assert b(op="add", seed=2)["added"] == 0, (mod, how)  # and cannot write
                    assert a(op="has", seeds=[2])["hits"] == 0
                finally:
                    a.close()
                    b.close()


@test
def shared_concurrent_writers():
    """4 processes append 40 sequences each at the same time; afterwards every process, and a fresh one, holds
    the same ids (same digest) and finds the sequences. vllm_d_recall.core's index is lossier than yee.core's
    (one process alone finds 159 of these 160), so it may miss up to 3 lookups; the ids must still be identical."""
    if _no_fcntl():
        return _no_fcntl()
    for mod in MODS:
        slack = 0 if mod == "yee.core" else 3
        with TmpDir() as d:
            ws = [Worker(mod) for _ in range(4)]
            f = None
            try:
                for w in ws:
                    assert w(op="open", dir=d)["shared"]
                seeds = [list(range(1000 + 100 * i, 1000 + 100 * i + 40)) for i in range(4)]
                for w, s in zip(ws, seeds):
                    w.send(op="addmany", seeds=s)
                for w in ws:
                    assert w.recv()["added"] == 40 * 600, mod
                every = sum(seeds, [])
                digests = set()
                for w in ws:
                    w(op="refresh")
                    hits = w(op="has", seeds=every)["hits"]
                    assert hits >= len(every) - slack, (mod, hits)
                    dg = w(op="digest")
                    assert dg["n"] >= 160 * 600, (mod, dg)
                    digests.add(dg["sha"])
                assert len(digests) == 1, (mod, digests)
                f = Worker(mod)
                f(op="open", dir=d)
                assert f(op="digest")["sha"] in digests, mod
                assert f(op="has", seeds=every)["hits"] >= len(every) - slack, mod
                _chunks_whole(d)
            finally:
                for w in ws + ([f] if f else []):
                    w.close()


@test
def shared_clear():
    """A clears: B drops its copy at the next refresh; what B writes afterwards reaches A."""
    if _no_fcntl():
        return _no_fcntl()
    for mod in MODS:
        with TmpDir() as d:
            a, b = Worker(mod), Worker(mod)
            try:
                a(op="open", dir=d)
                b(op="open", dir=d)
                a(op="add", seed=1)
                b(op="refresh")
                assert b(op="has", seeds=[1])["hits"] == 1
                a(op="clear")
                b(op="refresh")
                assert b(op="has", seeds=[1])["hits"] == 0, mod
                assert b(op="stats")["shared_resets"] >= 1, mod
                assert b(op="add", seed=3)["added"] == 600, mod
                a(op="refresh")
                assert a(op="has", seeds=[3])["hits"] == 1, mod
                assert a(op="has", seeds=[1])["hits"] == 0, mod
            finally:
                a.close()
                b.close()


@test
def shared_eviction():
    """A small record: the oldest chunks go on both sides, the newest ids stay findable everywhere."""
    if _no_fcntl():
        return _no_fcntl()
    gib = 3 * 4096 * 4 / float(1 << 30)  # 3 chunks of 4096 ids
    for mod in MODS:
        with TmpDir() as d:
            a, b = Worker(mod), Worker(mod)
            try:
                a(op="open", dir=d, gib=gib, evict="fifo")
                b(op="open", dir=d, gib=gib, evict="fifo")
                assert a(op="addmany", seeds=list(range(1, 41)))["added"] == 40 * 600
                b(op="refresh")
                assert b(op="has", seeds=list(range(36, 41)))["hits"] == 5, mod
                assert b(op="has", seeds=list(range(1, 6)))["hits"] == 0, mod
                assert b(op="addmany", seeds=list(range(41, 51)))["added"] == 10 * 600, mod
                a(op="refresh")
                assert a(op="has", seeds=list(range(46, 51)))["hits"] == 5, mod
                n = len([x for x in os.listdir(d) if x.startswith("chunk-")])
                assert n <= 5, (mod, n)
                _chunks_whole(d)
            finally:
                a.close()
                b.close()


@test
def shared_engine_tick():
    """RecallEngine.tick() refreshes a shared record (every 2 s; synchronous here: async_add=False)."""
    if _no_fcntl():
        return _no_fcntl()
    for mod in MODS:
        with TmpDir() as d:
            a, b = Worker(mod), Worker(mod)
            try:
                a(op="open", dir=d)
                b(op="open", dir=d)
                a(op="add", seed=7)
                assert b(op="has", seeds=[7])["hits"] == 0
                b(op="tick", now=1000.0)
                assert b(op="has", seeds=[7])["hits"] == 1, mod
            finally:
                a.close()
                b.close()


@test
def shared_next_to_an_old_writer():
    """A process that keeps the lock for good (3.5.x, or shared off) makes a shared one fall back to read-only
    after SHARED_LOCK_WAIT instead of waiting at every append."""
    if _no_fcntl():
        return _no_fcntl()
    for mod in MODS:
        with TmpDir() as d:
            a, b = Worker(mod), Worker(mod)
            try:
                assert a(op="open", dir=d, shared=False)["writer"]
                rb = b(op="open", dir=d, lock_wait=0.3)
                assert not rb["shared"] and not rb["writer"] and rb["err"], (mod, rb)
                assert b(op="add", seed=1)["added"] == 0
            finally:
                a.close()
                b.close()


# ------------------------------------------------------------------ pytest entry points
def _run(fn):
    res = fn()
    if isinstance(res, str) and res.startswith("SKIP"):
        import pytest
        pytest.skip(res)


def test_shared_two_processes():
    _run(shared_two_processes)


def test_unshared_keeps_old_behaviour():
    _run(unshared_keeps_old_behaviour)


def test_shared_concurrent_writers():
    _run(shared_concurrent_writers)


def test_shared_clear():
    _run(shared_clear)


def test_shared_eviction():
    _run(shared_eviction)


def test_shared_engine_tick():
    _run(shared_engine_tick)


def test_shared_next_to_an_old_writer():
    _run(shared_next_to_an_old_writer)


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
    print("python %s, fcntl %s, modules %s" % (sys.version.split()[0], rc.fcntl is not None, ",".join(MODS)))
    if fail:
        print("FAILED %d / %d (skip %d)" % (fail, ok + fail, skip))
        sys.exit(1)
    if ok == 0:
        print("SKIP all n=%d" % skip)
        return
    print("ALL PASS n=%d skip=%d" % (ok, skip))


if __name__ == "__main__":
    main()
