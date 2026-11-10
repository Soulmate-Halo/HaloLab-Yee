"""Plain-assert tests for the heat eviction (hot zone) of sglang_d_recall.core. Run: python tests/test_heat.py

Every test pins its own evict mode, so the result does not depend on RECALL_EVICT in the environment.
Also collected by pytest (the test_* functions).
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

IDENT = {"tokenizer_digest": "b" * 64, "vocab_size": 129280, "model": "test"}
RESULTS = []
CAP = 1 << 16  # total ids
GIB = CAP * 4 / float(1 << 30)
CHUNK = 4096


def test(fn):
    RESULTS.append(fn)
    return fn
test.__test__ = False  # decorator, not a test (pytest)


def rnd(n, seed, lo=10, hi=120000):
    return np.random.default_rng(seed).integers(lo, hi, size=n, dtype=np.int64)


def rec(**kw):
    args = dict(gib=GIB, chunk_ids=CHUNK, backend="memory", evict="heat", hot_frac=0.5,
                promote_tokens=32, heat_seen_weight=0.25)
    args.update(kw)
    return rc.RecallRecord(None, IDENT, **args)


def fill(r, n, seed):
    """Add n unique random ids in 1000-id requests (pushes the cold ring forward)."""
    for k in range(n // 1000):
        r.add(rnd(1000, seed * 100003 + k))


@test
def test_promote_on_credit():
    r = rec()
    a = rnd(2000, 1)
    r.add(a)
    p, L = r.lookup(a[:100])
    assert p == 100 and p < rc.HOT_FLAG, (p, L)
    r.credit(p, 20)
    assert r.promotions == 0 and abs(r.heat_of(p) - 20) < 1e-6, (r.promotions, r.heat_of(p))
    r.credit(p, 20)
    st = r.stats()
    assert st["promotions"] == 1 and st["hot_segments"] == 1, st
    assert st["hot_ids"] == rc.HEAT_SEG + rc.HOT_POST, st  # segment 0 has no ids before it
    assert abs(r.heat_of(p) - 40) < 1e-6
    r.credit(p, 5)  # keeps accruing on the hot copy
    assert abs(r.heat_of(p) - 45) < 1e-6 and r.promotions == 1


@test
def test_hot_preferred_on_tie():
    r = rec()
    a = rnd(2000, 2)
    r.add(a)
    r.credit(100, 64)
    hit = r.lookup(a[150:200])
    assert hit is not None and hit[0] >= rc.HOT_FLAG, hit
    assert np.array_equal(r.continuation(hit[0], 8), a[200:208])
    # beyond the hot copy only the cold ring has it
    hit2 = r.lookup(a[1000:1100])
    assert hit2 is not None and hit2[0] < rc.HOT_FLAG
    assert r.hot_hits == 1 and r.cold_hits == 1, (r.hot_hits, r.cold_hits)


@test
def test_hot_survives_cold_eviction():
    r = rec()
    a = rnd(2000, 3)
    r.add(a)
    r.credit(300, 64)  # segment 1 = [256, 512), copied as [240, 576)
    fill(r, 3 * CAP, 30)
    assert r.base_pos > 2000, r.base_pos
    assert r.stats()["evicted_chunks"] > 0
    for i in (260, 300, 400, 511, 560):
        hit = r.lookup(a[max(0, i - 100):i])
        assert hit is not None and hit[0] >= rc.HOT_FLAG, (i, hit)
        cont = r.continuation(hit[0], 8, min_n=1)
        want = a[i:min(i + 8, 576)]
        assert np.array_equal(cont, want), (i, cont, want)
    assert r.lookup(a[1000:1100]) is None  # never promoted, evicted with the cold chunk


@test
def test_hot_full_evicts_lowest_heat():
    r = rec(hot_frac=0.03, heat_halflife_ids=1e12)  # 5 hot slots, no decay
    n = r._hot_n
    assert n == int(CAP * 0.03) // rc.HOT_SLOT == 5, n
    seqs = [rnd(600, 40 + k) for k in range(n + 1)]
    starts = []
    for s in seqs:
        r.add(s)
        starts.append(r.lookup(s[:100])[0])
    heats = [100, 40, 90, 80, 70]
    for k in range(n):
        r.credit(starts[k], heats[k])
    assert r.stats()["hot_segments"] == n and r.hot_evictions == 0
    r.credit(starts[n], 50)
    st = r.stats()
    assert st["hot_evictions"] == 1 and st["hot_segments"] == n and st["promotions"] == n + 1, st
    assert r.heat_of(starts[1]) == 0.0  # the 40 one was dropped and its segment demoted

    def hot_hits(s):
        return sum(1 for i in range(120, 300)
                   if (lambda h: h is not None and h[0] >= rc.HOT_FLAG)(r.lookup(s[i - 50:i])))
    for k in (0, 2, 3, 4, 5):
        assert hot_hits(seqs[k]) >= 30, (k, hot_hits(seqs[k]))  # lossy index: not every window survives
    assert hot_hits(seqs[1]) == 0


@test
def test_decay_halflife():
    r = rec(heat_halflife_ids=1000, promote_tokens=1e9)
    a = rnd(1000, 5)
    r.add(a)  # clock = 1000
    r.credit(100, 10)
    assert abs(r.heat_of(100) - 10) < 1e-9
    r.add(rnd(1000, 6))  # clock + 1000 = one half-life
    assert abs(r.heat_of(100) - 5) < 1e-9, r.heat_of(100)
    r.add(rnd(2000, 7))
    assert abs(r.heat_of(100) - 1.25) < 1e-9, r.heat_of(100)


@test
def test_dedup_counts_seen_heat():
    r = rec(promote_tokens=1e9)
    assert r.dedup_run == rc.DEDUP_RUN_HEAT == 80
    a = rnd(500, 8)
    r.add(a)
    assert r.heat_of(100) == 0.0
    added = r.add(np.concatenate((rnd(300, 9), a, rnd(300, 10))))
    assert r.deduped_ids > 0 and added < 1100, (r.deduped_ids, added)
    h = r.heat_of(100)
    assert abs(h - 256 * 0.25) < 1e-6, h  # segment 0 fully inside the skipped run
    assert r.heat_of(300) > 0


@test
def test_dedup_run_80_keeps_lookup_across_boundary():
    r = rec(promote_tokens=1e9)
    a = rnd(400, 11)
    r.add(a)
    u1, u2 = rnd(200, 12), rnd(200, 13)
    b = np.concatenate((u1, a[100:200], u2))  # a 100-id duplicate run between novel parts
    added = r.add(b)
    assert added < b.size, added
    miss = 0
    for i in range(16, b.size - 8):
        hit = r.lookup(b[max(0, i - 64):i])
        if hit is None:  # the index is lossy (newest wins on a slot collision)
            miss += 1
            continue
        cont = r.continuation(hit[0], 1, min_n=1)
        assert cont is not None and int(cont[0]) == int(b[i]), (i, hit, cont)
    assert miss <= 0.02 * b.size, miss


@test
def test_fifo_has_no_hot_zone():
    r = rec(evict="fifo")
    st = r.stats()
    assert st["evict_mode"] == "fifo" and st["dedup_run"] == rc.DEDUP_RUN == 128 and st["hot_cap_ids"] == 0, st
    a = rnd(2000, 14)
    r.add(a)
    r.credit(100, 1000)
    st = r.stats()
    assert st["promotions"] == 0 and st["hot_ids"] == 0 and r.heat_of(100) == 0.0, st
    assert r.cold_cap_ids == r.cap_ids
    r0 = rec(hot_frac=0.0)  # no hot zone = fifo storage
    assert r0.evict_mode == "fifo" and r0.dedup_run == 128


@test
def test_memory_save_load_keeps_hot_zone():
    d = tempfile.mkdtemp(prefix="recall_h_")
    try:
        r = rec()
        a = rnd(2000, 15)
        r.add(a)
        r.credit(300, 64)
        r.credit(1100, 48)
        fill(r, 3 * CAP, 16)
        assert r.lookup(a[1000:1100]) is None or r.lookup(a[1000:1100])[0] >= rc.HOT_FLAG
        path = r.save(os.path.join(d, "snap.npz"))
        r2 = rc.RecallRecord.load(path, IDENT, backend="memory", gib=GIB, chunk_ids=CHUNK, evict="heat",
                                  hot_frac=0.5)
        s1, s2 = r.stats(), r2.stats()
        assert s2["hot_segments"] == s1["hot_segments"] == 2 and s2["hot_ids"] == s1["hot_ids"], (s1, s2)
        assert r2._clock == r._clock
        for i in (300, 1100):
            hit = r2.lookup(a[i - 100:i])
            assert hit is not None and hit[0] >= rc.HOT_FLAG, (i, hit)
            assert np.array_equal(r2.continuation(hit[0], 8), a[i:i + 8])
            assert abs(r2.heat_of(hit[0]) - r.heat_of(r.lookup(a[i - 100:i])[0])) < 1e-6
        assert r2.end_pos == r.end_pos and r2.base_pos == r.base_pos
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test
def test_clear_empties_hot_zone():
    r = rec()
    a = rnd(2000, 17)
    r.add(a)
    r.credit(100, 64)
    assert r.stats()["hot_segments"] == 1
    r.clear()
    st = r.stats()
    assert st["hot_ids"] == 0 and st["hot_segments"] == 0 and st["promotions"] == 0, st
    assert r.lookup(a[150:200]) is None and r.heat_of(100) == 0.0


@test
def test_engine_feedback_credits_source():
    r = rec()
    a = rnd(3000, 18)
    r.add(a)
    eng = rc.RecallEngine(r, gamma=8, async_add=False)
    eng.begin_stream("x", a[:200])
    nxt = 200
    for _ in range(40):
        prop = eng.propose("x", own_ema=0.0)
        if prop is not None:
            assert prop.source == "record" and np.array_equal(prop.tokens[:prop.real], a[nxt:nxt + prop.real])
            eng.feedback("x", prop.real)
            eng.extend_stream("x", a[nxt:nxt + prop.real])
            nxt += prop.real
        else:
            eng.extend_stream("x", a[nxt:nxt + 1])
            nxt += 1
    assert r.promotions >= 1, r.stats()
    assert r.hot_hits >= 1, r.stats()
    eng.close()


@test
def test_continuation_never_returns_wrong_ids():
    """Random adds, credits, cold and hot evictions: every hit's continuation is what really followed."""
    r = rec(hot_frac=0.25, promote_tokens=8, heat_halflife_ids=CAP // 2)
    rng = np.random.default_rng(19)
    pool = [rnd(int(rng.integers(300, 900)), 1000 + k) for k in range(40)]
    truth = {}  # 16-gram -> ids that followed it at least once
    checked = hot_checked = 0
    for it in range(400):
        k = int(min(39, rng.zipf(1.3) - 1))
        tmpl = pool[k]
        cut = int(rng.integers(0, tmpl.size // 2))
        seq = np.concatenate((rnd(int(rng.integers(20, 200)), 50000 + it), tmpl[cut:], rnd(64, 90000 + it)))
        for i in range(rc.WINDOW, seq.size):
            truth.setdefault(tuple(seq[i - rc.WINDOW:i].tolist()), set()).add(tuple(seq[i:i + 8].tolist()))
        for i in rng.integers(rc.WINDOW, seq.size - 1, size=12).tolist():
            hit = r.lookup(seq[max(0, i - 128):i])
            if hit is None:
                continue
            cont = r.continuation(hit[0], 8, min_n=1)
            if cont is None:
                continue
            nexts = truth[tuple(seq[i - rc.WINDOW:i].tolist())]
            assert any(tuple(cont.tolist()) == n[:cont.size] for n in nexts), (it, i, hit)
            checked += 1
            hot_checked += hit[0] >= rc.HOT_FLAG
            good = 0
            while good < cont.size and i + good < seq.size and int(cont[good]) == int(seq[i + good]):
                good += 1
            r.credit(hit[0], good)
        r.add(seq)
    st = r.stats()
    assert st["evicted_chunks"] > 0 and st["promotions"] > 0 and st["hot_evictions"] > 0, st
    assert checked > 1000 and hot_checked > 100, (checked, hot_checked)


def main():
    ok = fail = 0
    for fn in RESULTS:
        try:
            fn()
            ok += 1
            print("PASS %s" % fn.__name__)
        except Exception:
            fail += 1
            print("FAIL %s" % fn.__name__)
            traceback.print_exc()
    print("numpy %s, python %s" % (np.__version__, sys.version.split()[0]))
    if fail:
        print("FAILED %d / %d" % (fail, ok + fail))
        sys.exit(1)
    print("ALL PASS n=%d" % ok)


if __name__ == "__main__":
    main()
