"""Cross-backend equivalence: disk / memory / cuda must answer add, lookup, continuation, clear and
stats the same way. Run: python tests/test_backends_equiv.py

The check is a byte-for-byte comparison of what each backend returns for the same random sequences, so a
behavioural drift between the stores shows up here.
"""
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from sglang_d_recall import core as rc  # noqa: E402

IDENT = {"tokenizer_digest": "e" * 64, "vocab_size": 129280, "model": "equiv"}
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn
test.__test__ = False  # decorator, not a test (pytest)


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


def available():
    return ["disk", "memory"] + (["cuda"] if CUDA_OK else [])


def rnd(n, seed, lo=10, hi=120000):
    return np.random.default_rng(seed).integers(lo, hi, size=n, dtype=np.int64)


class TmpDir:
    def __enter__(self):
        self.d = tempfile.mkdtemp(prefix="recall_eq_")
        return os.path.join(self.d, "rec")

    def __exit__(self, *a):
        shutil.rmtree(self.d, ignore_errors=True)


def make(backend, directory=None, **kw):
    if backend == "disk":
        return rc.RecallRecord(directory, IDENT, **kw)
    return rc.RecallRecord(None, IDENT, backend=backend, **kw)


def probe(rec, tails, cont_n=5):
    """Everything the three backends must agree on, for a list of tails."""
    out = []
    for tail in tails:
        hit = rec.lookup(tail)
        if hit is None:
            out.append(None)
            continue
        p, L = hit
        cont = rec.continuation(p, cont_n, min_n=1)
        out.append((int(p), int(L), None if cont is None else cont.tolist()))
    return out


@test
def lookup_and_continuation_agree():
    """Same sequence -> same lookups and same continuations on every backend."""
    backends = available()
    seqs = [rnd(2000, 1), rnd(700, 2), np.concatenate((rnd(500, 3), rnd(500, 4)))]
    tails = []
    for k, s in enumerate(seqs):
        for i in (20, 33, 64, 100, 199, 400):
            if i < s.size:
                tails.append(s[max(0, i - 40):i])
    tails.append(rnd(40, 99))
    results = {}
    for b in backends:
        with TmpDir() as d:
            r = make(b, d, chunk_ids=512)
            for s in seqs:
                r.add(s)
            results[b] = (probe(r, tails), r.end_pos - r.base_pos, r.added_ids, r.deduped_ids)
            r.close()
    base = results[backends[0]]
    for b in backends[1:]:
        assert results[b] == base, "backend %s differs from %s" % (b, backends[0])
    print("    %d backends agree on %d tails" % (len(backends), len(tails)))


@test
def add_counts_agree():
    """add() must report the same number of appended ids on every backend."""
    backends = available()
    counts = {}
    for b in backends:
        with TmpDir() as d:
            r = make(b, d, chunk_ids=512)
            seq = rnd(3000, 7)
            counts[b] = (r.add(seq), r.add(seq), r.add(rnd(200, 8)), r.adds, r.deduped_ids)
            r.close()
    base = counts[backends[0]]
    for b in backends[1:]:
        assert counts[b] == base, "backend %s differs: %s vs %s" % (b, counts[b], base)


@test
def clear_agrees():
    """clear() must empty the record on every backend."""
    backends = available()
    for b in backends:
        with TmpDir() as d:
            r = make(b, d, chunk_ids=256)
            s = rnd(1000, 9)
            r.add(s)
            assert r.lookup(s[100:140]) is not None, b
            r.clear()
            assert r.lookup(s[100:140]) is None, b
            assert r.end_pos - r.base_pos == 0, b
            assert r.added_ids == 0 and r.adds == 0, b
            r.close()


@test
def stats_agree():
    """stats() must agree on the fields that do not depend on where the ids live."""
    backends = available()
    common = {}
    for b in backends:
        with TmpDir() as d:
            r = make(b, d, chunk_ids=512)
            r.add(rnd(1500, 11))
            st = r.stats()
            common[b] = {k: st[k] for k in ("state", "writer", "ids", "cap_ids", "added_ids",
                                            "deduped_ids", "adds", "add_blocked")}
            r.close()
    base = common[backends[0]]
    for b in backends[1:]:
        assert common[b] == base, "backend %s stats differ: %s vs %s" % (b, common[b], base)


@test
def eviction_agrees():
    """Over the capacity the backends must evict the same way and forget the same ids."""
    backends = available()
    gib = 1024 * 4 / float(1 << 30)  # 1024 ids
    out = {}
    for b in backends:
        with TmpDir() as d:
            r = make(b, d, gib=gib, chunk_ids=256)
            segs = [rnd(300, 100 + k) for k in range(10)]
            for sg in segs:
                r.add(sg)
            out[b] = (r.end_pos - r.base_pos, r.evicted_chunks,
                      r.lookup(segs[0][:40]) is None,
                      r.lookup(segs[-1][200:240]) is not None)
            r.close()
    base = out[backends[0]]
    for b in backends[1:]:
        assert out[b] == base, "backend %s differs: %s vs %s" % (b, out[b], base)


@test
def engine_proposals_agree():
    """The engine must offer the same copies on every backend for the same conversation."""
    backends = available()
    outs = {}
    for b in backends:
        with TmpDir() as d:
            r = make(b, d, chunk_ids=1 << 13)
            eng = rc.RecallEngine(r, gamma=5, async_add=False)
            prompt, reply = rnd(200, 20), rnd(300, 21)
            for rid in ("a", "b"):
                eng.begin_stream(rid, prompt)
                pos = 0
                while pos < len(reply):
                    prop = eng.propose(rid, own_ema=0.0)
                    if prop is None:
                        eng.observe_own(rid, 2.0)
                        adv = 3
                    else:
                        truth = reply[pos:pos + 5]
                        k = 0
                        while k < min(len(truth), prop.real) and prop.tokens[k] == truth[k]:
                            k += 1
                        eng.feedback(rid, k)
                        adv = k + 1
                    eng.extend_stream(rid, reply[pos:pos + adv])
                    pos += adv
                eng.end_stream(rid, reply)
            outs[b] = eng.stats()["tiers"]
            eng.close()
    base = outs[backends[0]]
    for b in backends[1:]:
        assert outs[b] == base, "backend %s tiers differ: %s vs %s" % (b, outs[b], base)


def main():
    ok = fail = 0
    for fn in TESTS:
        try:
            fn()
            ok += 1
            print("PASS %s" % fn.__name__)
        except Exception:
            fail += 1
            print("FAIL %s" % fn.__name__)
            traceback.print_exc()
    print("numpy %s, python %s, backends=%s" % (np.__version__, sys.version.split()[0], ",".join(available())))
    if not CUDA_OK:
        print("SKIP cuda backend: no torch/CUDA on this machine")
    if fail:
        print("FAILED %d / %d" % (fail, ok + fail))
        sys.exit(1)
    print("ALL PASS n=%d" % ok)


if __name__ == "__main__":
    main()
