"""Plain-assert tests for the index density parameter of yee.core. Run: python tests/test_density.py

Every test pins its own density and evict mode, so the result does not depend on the environment.
Also collected by pytest (the test_* functions).
"""
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from yee import core as rc  # noqa: E402

IDENT = {"tokenizer_digest": "d" * 64, "vocab_size": 129280, "model": "density"}
RESULTS = []

CAP = 1 << 20  # total ids of the record
CHUNK = 1 << 16
GIB = CAP * 4 / float(1 << 30)
BATCH = 4096  # ids per add()
MULT = 4  # write this many times the capacity
QUERIES = 20000  # lookups after the pool is full
TPL = 16  # ids of one repeated template
TPL_REPEAT = 4  # how often the template is repeated inside one batch


def test(fn):
    RESULTS.append(fn)
    return fn
test.__test__ = False  # decorator, not a test (pytest)


def rec(evict="heat", density=None, **kw):
    args = dict(gib=GIB, chunk_ids=CHUNK, backend="memory", evict=evict, hot_frac=0.5, environ={})
    if density is not None:
        args["index_density"] = density
    args.update(kw)
    return rc.RecallRecord(None, IDENT, **args)


def old_slots(ids):
    """The 1.1.0 slot count: the next power of two of ids // 4."""
    want = max(2, ids // 4)
    return 1 << max(16, (want - 1).bit_length())


def old_hot_slots(ids):
    want = max(2, ids // 4)
    return 1 << max(12, (want - 1).bit_length())


# ------------------------------------------------------------------ the parameter itself
@test
def test_default_density_keeps_the_old_table():
    r = rec()
    assert r.index_density == 0.25, r.index_density
    ids = r.cold_cap_ids if r.evict_mode == "heat" else r.cap_ids
    assert r._slots == old_slots(ids), (r._slots, old_slots(ids))
    assert r._hot_slots == old_hot_slots(r.hot_cap_ids), (r._hot_slots, old_hot_slots(r.hot_cap_ids))
    r.close()


@test
def test_density_one_is_one_slot_per_id():
    r = rec(density=1.0)
    ids = r.cold_cap_ids if r.evict_mode == "heat" else r.cap_ids
    want = 1 << max(16, (ids - 1).bit_length())  # the next power of two of the ids
    assert r._slots == want, (r._slots, want)
    assert r.index_density == 1.0
    r.close()


@test
def test_density_is_clamped():
    r = rec(density=0.0)
    assert r.index_density == rc.MIN_INDEX_DENSITY, r.index_density
    r.close()
    r = rec(density=1000.0)
    assert r.index_density == rc.MAX_INDEX_DENSITY, r.index_density
    r.close()


@test
def test_density_from_environment():
    env = {rc.ENV_NEW["index_density"]: "0.5"}
    r = rc.RecallRecord(None, IDENT, gib=GIB, chunk_ids=CHUNK, backend="memory", evict="heat", environ=env)
    assert r.index_density == 0.5, r.index_density
    r.close()
    env = {rc.ENV_OLD["index_density"]: "2"}  # the old name is the fallback
    r = rc.RecallRecord(None, IDENT, gib=GIB, chunk_ids=CHUNK, backend="memory", evict="heat", environ=env)
    assert r.index_density == 2.0, r.index_density
    r.close()
    env = {rc.ENV_NEW["index_density"]: "1", rc.ENV_OLD["index_density"]: "0.1"}  # the new name wins
    r = rc.RecallRecord(None, IDENT, gib=GIB, chunk_ids=CHUNK, backend="memory", evict="heat", environ=env)
    assert r.index_density == 1.0, r.index_density
    r.close()


@test
def test_stats_report_density_and_slots():
    r = rec(density=1.0)
    st = r.stats()
    assert st["index_density"] == 1.0, st
    assert st["index_slots"] == r._slots, st
    assert st["hot_index_slots"] == r._hot_slots, st
    r.close()


@test
def test_snapshot_keeps_the_density():
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="recall_density_")
    try:
        p = os.path.join(d, "snap.npz")
        r = rec(density=1.0)
        r.add(np.arange(100, 300, dtype=np.int64))
        r.save(p)
        r.close()
        r2 = rc.RecallRecord.load(p, IDENT, backend="memory")
        assert r2.index_density == 1.0, r2.index_density
        r2.close()
        r3 = rc.RecallRecord.load(p, IDENT, backend="memory", index_density=0.5)
        assert r3.index_density == 0.5, r3.index_density
        r3.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ recall rate under load
def _write_pool(r, seed=1234):
    """Fill the record with MULT x capacity ids: repeated templates mixed with random noise.

    The repeated part stays below the dedup run length, so the record keeps every id and the layout
    (batch, sentinel, batch, sentinel, ...) is exact."""
    rng = np.random.default_rng(seed)
    templates = [rng.integers(10, 120000, size=TPL, dtype=np.int64) for _ in range(40)]
    batches = []
    for k in range(MULT * CAP // BATCH):
        t = templates[int(rng.integers(len(templates)))]
        head = np.concatenate([t] * TPL_REPEAT)
        noise = rng.integers(10, 120000, size=BATCH - head.size, dtype=np.int64)
        seq = np.concatenate((head, noise))
        r.add(seq)
        batches.append(seq)
        if k % 32 == 0:  # a bit of heat so the hot zone (heat mode) really gets used
            hit = r.lookup(seq[BATCH // 2:BATCH // 2 + rc.WINDOW], max_back=rc.WINDOW)
            if hit is not None:
                r.credit(hit[0], 40)
    return batches


def _expected_layout(batches):
    """What the record holds: every batch, a sentinel in front of every batch but the first."""
    parts = []
    for k, b in enumerate(batches):
        if k:
            parts.append(np.array([rc.SENTINEL], dtype=np.int64))
        parts.append(b)
    return np.concatenate(parts)


def _sent_before(p):
    """How many sentinels sit before the record position ``p`` (one per batch but the first)."""
    return p // (BATCH + 1)


def _run(evict, density):
    r = rec(evict=evict, density=density)
    batches = _write_pool(r)
    original = np.concatenate(batches)
    layout = _expected_layout(batches)
    assert r.end_pos == layout.size, (r.end_pos, layout.size)
    live = r._read(r.base_pos, r.end_pos)
    assert live is not None and np.array_equal(live, layout[r.base_pos:r.end_pos]), "the record changed the ids"
    # hash index over the original stream (for the hot zone hits: their continuation is a copy)
    hs = rc._window_hashes(original)
    hs_sorted = np.sort(hs)
    order = np.argsort(hs, kind="stable")

    def hot_ok(gram, cont):
        """True when ``cont`` really followed ``gram`` somewhere in the original write."""
        t = rc._hash_window_int(gram.tolist())
        lo = int(np.searchsorted(hs_sorted, t, side="left"))
        hi = int(np.searchsorted(hs_sorted, t, side="right"))
        L = int(cont.size)
        for p in order[lo:hi].tolist():
            if p + rc.WINDOW + L > original.size:
                continue
            if not np.array_equal(original[p:p + rc.WINDOW], gram):
                continue  # a hash collision, not the same 16-gram
            if np.array_equal(original[p + rc.WINDOW:p + rc.WINDOW + L], cont):
                return True
        return False

    tail = original[original.size - CAP:]
    rng = np.random.default_rng(99)
    hi = tail.size - rc.WINDOW
    positions = np.sort(rng.integers(0, hi, size=QUERIES))
    hits = 0
    cold = hot = 0
    for p in positions.tolist():
        g = tail[p:p + rc.WINDOW]
        if g.size < rc.WINDOW:
            continue
        hit = r.lookup(g, max_back=rc.WINDOW)
        if hit is None:
            continue
        hits += 1
        cont = r.continuation(hit[0], 8, min_n=1)
        assert cont is not None and cont.size > 0, (evict, density, p, hit)
        if hit[0] >= rc.HOT_FLAG:
            hot += 1
            assert hot_ok(g, cont), (evict, density, p, cont)
        else:
            cold += 1
            q = hit[0]
            o = q - _sent_before(q)  # the position of the same id in the original write
            assert np.array_equal(cont, original[o:o + cont.size]), (evict, density, p, q, cont)
    rate = hits / float(QUERIES)
    r.close()
    return rate, hits, cold, hot


@test
def test_density_one_recalls_at_least_as_much():
    for evict in ("heat", "fifo"):
        low, low_hits, low_cold, low_hot = _run(evict, 0.25)
        high, high_hits, high_cold, high_hot = _run(evict, 1.0)
        assert low_hits > 0 and high_hits > 0, (evict, low_hits, high_hits)
        assert high >= low, (evict, low, high)
        print("    density %s: recall 0.25 -> %.4f (%d hits, %d cold, %d hot), 1.0 -> %.4f (%d hits, %d cold, %d hot)"
              % (evict, low, low_hits, low_cold, low_hot, high, high_hits, high_cold, high_hot))


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
