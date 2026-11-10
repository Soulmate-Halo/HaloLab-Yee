"""Plain-assert tests for the 8/4-id short-window fallback of yee.core (r593).
Run: python tests/test_recall_multilen.py
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
from yee import core as rc  # noqa: E402

IDENT = {"tokenizer_digest": "a" * 64, "vocab_size": 129280, "model": "test"}
RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn
test.__test__ = False  # decorator, not a test (pytest)


def rnd(n, seed, lo=10, hi=120000):
    return np.random.default_rng(seed).integers(lo, hi, size=n, dtype=np.int64)


def mem_record(**kw):
    return rc.RecallRecord(None, IDENT, backend="memory", **kw)


@test
def windows_parse():
    assert rc._parse_windows("16,8,4") == (8, 4)
    assert rc._parse_windows("4, 12 ,8,8,3,20,x") == (12, 8, 4)
    assert rc._parse_windows("16") == ()
    assert rc._parse_windows("") == ()
    assert rc.SHORT_WINDOWS == rc._parse_windows(os.environ.get("RECALL_ACCEL_WINDOWS", "16,8,4"))


@test
def short_tail_hits_8_then_4():
    r = mem_record()
    s = rnd(400, 1)
    r.add(s)
    p, L = r.lookup(s[100:110])  # 10 ids: below 16, the 8-id window finds it and reaches back to 10
    assert p == r.base_pos + 110 and L == 10, (p, L)
    assert np.array_equal(r.continuation(p, 5), s[110:115])
    p, L = r.lookup(s[200:205])  # 5 ids: the 4-id window
    assert L == 5 and np.array_equal(r.continuation(p, 5), s[205:210]), L
    assert r.lookup(s[200:203]) is None  # below the shortest window
    assert r.lookup(rnd(12, 77)) is None
    assert r.short_hits == 2
    r.close()


@test
def fork_recovers_after_four_ids():
    r = mem_record()
    s = rnd(500, 2)
    r.add(s)
    live = np.concatenate((s[:200], [7], s[201:204]))  # one divergent id, then 3 ids of the record
    assert r.lookup(live) is None
    live = np.concatenate((s[:200], [7], s[201:205]))  # 4 ids after the fork
    p, L = r.lookup(live)
    assert L == 4 and np.array_equal(r.continuation(p, 8), s[205:213]), L
    r.close()


@test
def long_window_still_wins():
    r = mem_record()
    s = rnd(400, 3)
    r.add(s)
    p, L = r.lookup(s[50:150])
    assert L == 100 and np.array_equal(r.continuation(p, 5), s[150:155])
    r.close()


@test
def engine_offers_short_copy_capped_at_twice_the_match():
    r = mem_record()
    s = rnd(600, 4)
    r.add(np.concatenate((rnd(30, 40), s)))
    eng = rc.RecallEngine(r, gamma=15, async_add=False)
    # the stream's own context: 20 unrelated ids, then 5 ids of the record (a 5-id agreement)
    eng.begin_stream("q", np.concatenate((rnd(20, 41), s[300:305])))
    prop = eng.propose("q", own_ema=0.0, n=15)
    assert prop is not None and prop.source == "record", prop
    assert prop.tier == rc.SHORT_TIER and prop.match_len == 5, (prop.tier, prop.match_len)
    assert prop.real == 10, prop.real
    assert np.array_equal(np.asarray(prop.tokens)[:10], s[305:315])
    assert len(prop.tokens) == 15
    eng.feedback("q", 10)
    st = eng.stats()
    assert any(k.endswith("<16") for k in st.get("tiers", st)), st
    eng.close()


@test
def short_tier_has_half_prior():
    r = mem_record()
    eng = rc.RecallEngine(r, gamma=10, async_add=False)
    assert abs(eng._expect("record", rc.SHORT_TIER) - 4.0) < 1e-9
    assert abs(eng._expect("record", 0) - 8.0) < 1e-9
    assert rc.tier_of(4) == rc.SHORT_TIER and rc.tier_of(15) == rc.SHORT_TIER and rc.tier_of(16) == 0
    eng.close()


@test
def clear_and_reload_rebuild_short_index():
    d = tempfile.mkdtemp(prefix="recall_ml_")
    try:
        r = mem_record()
        s = rnd(300, 5)
        r.add(s)
        assert r.lookup(s[100:106]) is not None
        path = r.save(os.path.join(d, "rec.npz"))
        r.clear()
        assert r.lookup(s[100:106]) is None
        r.close()
        r2 = rc.RecallRecord.load(path, IDENT, backend="memory")
        p, L = r2.lookup(s[100:106])
        assert L == 6 and np.array_equal(r2.continuation(p, 4), s[106:110])
        r2.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test
def eviction_purges_short_index():
    r = rc.RecallRecord(None, IDENT, backend="memory", gib=4096 * 4 / (1 << 30) * 3, chunk_ids=4096)
    first = rnd(4000, 6)
    r.add(first)
    for k in range(8):
        r.add(rnd(4000, 100 + k))
    if r.base_pos > 0:
        assert r.lookup(first[1000:1006]) is None
    for _w, sk, sp in r._short:
        live = sp[sp >= 0]
        assert (live >= r.base_pos).all()
    r.close()


@test
def memory_counts_short_indexes():
    r = mem_record()
    extra = sum(sk.nbytes + sp.nbytes for _w, sk, sp in r._short)
    assert extra == len(rc.SHORT_WINDOWS) * r._sslots * 16
    assert r._sslots == max(1 << 14, r._slots // 4)
    assert r.stats()["bytes_mem"] >= r._keys.nbytes + r._pos.nbytes + extra
    r.close()


def main():
    fails = 0
    for fn in RESULTS:
        try:
            fn()
            print("  PASS", fn.__name__)
        except Exception:
            fails += 1
            print("> FAIL", fn.__name__)
            traceback.print_exc()
    print("windows=%s numpy %s" % (rc.SHORT_WINDOWS, np.__version__))
    print("ALL PASS n=%d" % len(RESULTS) if not fails else "FAILED %d / %d" % (fails, len(RESULTS)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
