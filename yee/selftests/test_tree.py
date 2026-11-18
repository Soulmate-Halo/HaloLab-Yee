"""Plain-assert tests of Yee 2.3.0's tree drafts (yee.tree, RecallRecord.lookup_all, RecallEngine.candidates /
propose_select, the chain heads' plan switch and the ``tree`` field of ``yee bases``).
Run: python tests/test_tree.py
"""
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from yee import core as rc  # noqa: E402
from yee import tree as yt  # noqa: E402

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn
test.__test__ = False  # decorator, not a test (pytest)


def tv(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum())


# ------------------------------------------------------------------------------------------------ candidates

@test
def test_dedupe_keeps_the_longer_of_a_prefix_pair():
    c = [yt.Candidate(np.array([1, 2, 3]), 0.5), yt.Candidate(np.array([1, 2, 3, 4, 5]), 0.4),
         yt.Candidate(np.array([7, 8]), 0.9), yt.Candidate(np.array([1, 2]), 0.95)]
    out = yt.dedupe(c)
    assert len(out) == 2, out
    assert out[0].tokens.tolist() == [1, 2, 3, 4, 5] and abs(out[0].p - 0.95) < 1e-12
    assert out[1].tokens.tolist() == [7, 8]


# ------------------------------------------------------------------------------------------------------ trees

@test
def test_build_tree_shares_prefixes_and_scales_siblings():
    t = yt.build_tree(0, [([1, 2, 3, 4], [0.5] * 4), ([1, 2, 9, 9], [0.5] * 4)], 100)
    assert t.n == 1 + 2 + 2 + 2, t.tokens
    assert abs(t.expected_tokens() - (1 + 0.5 + 0.25 + 0.125 * 2 + 0.0625 * 2)) < 1e-12
    s = yt.build_tree(0, [([5], [0.9]), ([6], [0.9])], 100)
    assert abs(s.expected_tokens() - 2.0) < 1e-12
    # parents first, depths consistent
    for i in range(1, t.n):
        assert t.parents[i] < i and t.depth[i] == t.depth[t.parents[i]] + 1


@test
def test_prune_keeps_the_best_nodes_with_their_ancestors():
    rng = np.random.default_rng(3)
    for _ in range(200):
        chains = []
        for _c in range(rng.integers(1, 5)):
            n = int(rng.integers(1, 8))
            chains.append((rng.integers(0, 4, size=n).tolist(), rng.uniform(0.05, 1.0, size=n).tolist()))
        full = yt.build_tree(9, chains, 10 ** 6)
        b = int(rng.integers(1, full.n + 1))
        t = yt.build_tree(9, chains, b)
        assert t.n == min(b, full.n), (t.n, b, full.n)
        for i in range(1, t.n):
            assert 0 <= t.parents[i] < i
        # the kept chances are the best b-1 of the full tree's (ancestor-closed, so best-first never needs more)
        want = sorted(full.score[1:], reverse=True)[: t.n - 1]
        assert np.allclose(sorted(t.score[1:], reverse=True), want), (t.score, want)


@test
def test_sglang_layout_matches_a_hand_built_tree():
    # root(t1=50) -> a(10) -> c(12); root -> b(11) -> d(13), b -> e(14)
    t = yt.tree_from_parents([50, 10, 11, 12, 13, 14], [-1, 0, 0, 1, 2, 2])
    s = yt.to_sglang(t, prefix_len=7)
    assert s["draft_tokens"].tolist() == [50, 10, 11, 12, 13, 14]
    assert s["positions"].tolist() == [7, 8, 8, 9, 9, 9]
    assert s["retrive_next_token"].tolist() == [1, 3, 4, -1, -1, -1]
    assert s["retrive_next_sibling"].tolist() == [-1, 2, -1, -1, 5, -1]
    m = s["tree_mask"].astype(int)
    assert m[5].tolist() == [1, 0, 1, 0, 0, 1] and m[3].tolist() == [1, 1, 0, 1, 0, 0]
    full = yt.sglang_full_mask(t, 7).reshape(6, 13)
    assert full[:, :7].all() and (full[:, 7:] == s["tree_mask"]).all()
    back = yt.from_sglang(s["draft_tokens"], s["retrive_next_token"], s["retrive_next_sibling"])
    assert back.tokens.tolist() == t.tokens.tolist() and back.parents.tolist() == t.parents.tolist()


@test
def test_merge_chains_keeps_the_node_count_and_adds_a_strong_copy():
    own = yt.tree_from_parents([50, 10, 11, 12, 13], [-1, 0, 0, 1, 2], score=[1, 0.4, 0.3, 0.1, 0.05])
    m = yt.merge_chains(own, [([20, 21, 22], [0.95, 0.95, 0.95])])
    assert m.n == own.n
    toks = m.tokens.tolist()
    assert 20 in toks and 21 in toks, toks
    m2 = yt.merge_chains(own, [([20, 21], [0.01, 0.01])])
    assert 20 not in m2.tokens.tolist()


@test
def test_ngram_insert_adds_only_new_nodes():
    trie = {0: {5: 1}, 1: {6: 2}, 2: {}}
    assert yt.ngram_insert(trie, 0, [5, 6, 7, 8]) == 2
    assert yt.ngram_insert(trie, 0, [5, 6, 7]) == 0


# --------------------------------------------------------------------------------------------- verification

V = 5
DENS = np.array([
    [0.10, 0.50, 0.20, 0.10, 0.10],
    [0.30, 0.10, 0.40, 0.10, 0.10],
    [0.20, 0.20, 0.20, 0.20, 0.20],
    [0.60, 0.10, 0.10, 0.10, 0.10],
    [0.05, 0.05, 0.10, 0.30, 0.50],
])


def _serial(rng, t1, L):
    out, last = [], t1
    for _ in range(L):
        last = int(rng.choice(V, p=DENS[last]))
        out.append(last)
    return out


def _emit(tree, verdict, rng, L):
    seq = [int(tree.tokens[i]) for i in verdict.path[1:]][:L]
    if len(seq) < L:
        seq.append(verdict.next_token)
    while len(seq) < L:
        seq.append(int(rng.choice(V, p=DENS[seq[-1]])))
    return seq[:L]


def _hist(samples, L):
    h = np.zeros(V ** L)
    for s in samples:
        k = 0
        for x in s:
            k = k * V + x
        h[k] += 1
    return h / len(samples)


@test
def test_greedy_walk_is_serial_argmax():
    t = yt.build_tree(0, [([1, 2, 4], [0.9] * 3), ([1, 0, 1], [0.5] * 3), ([3], [0.2])], 100)
    v = yt.verify_walk(t, lambda node: int(np.argmax(DENS[int(t.tokens[node])])))
    # serial argmax from 0: 1, 2(argmax of row 1), 0 (row 2 ties -> 0) ...
    assert [int(t.tokens[i]) for i in v.path] == [0, 1, 2], v
    assert v.next_token == int(np.argmax(DENS[2]))
    c = yt.verify_walk(t, lambda node: int(np.argmax(DENS[int(t.tokens[node])])), cap=1)
    assert c.accepted == 1 and c.next_token == 2


@test
def test_point_mass_walk_is_lossless():
    rng = np.random.default_rng(11)
    t = yt.build_tree(0, [([1, 2, 0], [0.9] * 3), ([1, 0, 1], [0.5] * 3), ([2, 3], [0.4] * 2)], 100)
    L, n = 2, 40000
    tree_s, ser_s = [], []
    for _ in range(n):
        v = yt.verify_walk(t, lambda node: int(rng.choice(V, p=DENS[int(t.tokens[node])])))
        tree_s.append(_emit(t, v, rng, L))
        ser_s.append(_serial(rng, 0, L))
    exact = np.zeros(V ** L)
    for a in range(V):
        for b in range(V):
            exact[a * V + b] = DENS[0, a] * DENS[a, b]
    ht = _hist(tree_s, L)
    assert tv(ht, exact) < 0.02, tv(ht, exact)
    assert tv(ht, _hist(ser_s, L)) < 0.025


@test
def test_specinfer_with_sampled_children_is_lossless():
    rng = np.random.default_rng(5)
    Q = np.array([0.4, 0.1, 0.1, 0.2, 0.2])  # the head's (wrong) density at every node
    L, n = 2, 30000
    out = []
    for _ in range(n):
        # a fresh tree a round: 3 children of the root and 2 under each, drawn i.i.d. from Q
        toks, par = [0], [-1]
        for _c in range(3):
            toks.append(int(rng.choice(V, p=Q)))
            par.append(0)
        for parent in (1, 2, 3):
            for _c in range(2):
                toks.append(int(rng.choice(V, p=Q)))
                par.append(parent)
        order = sorted(range(len(toks)), key=lambda i: (0 if par[i] < 0 else (1 if par[i] == 0 else 2), i))
        new = {o: k for k, o in enumerate(order)}
        t = yt.tree_from_parents([toks[i] for i in order], [new[par[i]] if par[i] >= 0 else -1 for i in order])
        v = yt.verify_specinfer(t, lambda node: DENS[int(t.tokens[node])], lambda node: Q, rng)
        out.append(_emit(t, v, rng, L))
    exact = np.zeros(V ** L)
    for a in range(V):
        for b in range(V):
            exact[a * V + b] = DENS[0, a] * DENS[a, b]
    d = tv(_hist(out, L), exact)
    assert d < 0.02, d


# ------------------------------------------------------------------------------------------------- selection

@test
def test_choose_chain_keeps_the_head_alone_and_on_ties():
    assert yt.choose_chain([[0.7, 0.7, 0.7]]) == 0
    assert yt.choose_chain([[0.7, 0.7], [0.7, 0.7]]) == 0
    assert yt.choose_chain([[0.5, 0.5, 0.5], [0.9, 0.9, 0.9]]) == 1
    assert yt.choose_chain([[0.9, 0.9], [0.95, 0.1]]) == 0


@test
def test_choose_plan_matches_the_mlx_serve_cases():
    stage = [list(range(1, 7)), [0.4] * 6]
    strong = [list(range(7, 17)), [0.9] * 10]
    own, mixed = yt.choose_plan([stage, strong], 2)
    assert mixed.lens[1] > 0 and mixed.rate > own.rate * 1.05 and mixed.rows <= 16
    weak = [list(range(7, 17)), [0.05] * 10]
    own, mixed = yt.choose_plan([stage, weak], 2)
    assert mixed.rate < own.rate
    # one branch alone: the brute-force best cut
    p = [0.85] * 8
    own, mixed = yt.choose_plan([[list(range(8)), p]], 2)
    best = max(range(9), key=lambda l: yt.chain_tokens(p, l) / (yt.rel_cost(l + 1) + 0.3))
    assert own.lens[0] == best and mixed.rows == 0


@test
def test_rel_cost_is_the_ladder_and_extends_past_16():
    assert abs(yt.rel_cost(1) - 1.0) < 1e-12
    assert abs(yt.rel_cost(16) - 69.1 / 17.7) < 1e-12
    assert yt.rel_cost(20) > yt.rel_cost(16)
    assert abs(yt.rel_cost(10) - (51.1 + (60.5 - 51.1) / 3) / 17.7) < 1e-12


# -------------------------------------------------------------------------------------------- core and wiring

IDENT = {"tokenizer_digest": "a" * 64, "vocab_size": 129280, "model": "test"}


def _engine():
    rec = rc.RecallRecord(None, IDENT, backend="memory")
    return rec, rc.RecallEngine(rec, gamma=6)


def _seq(n, seed):
    return np.random.default_rng(seed).integers(10, 120000, size=n, dtype=np.int64)


@test
def test_lookup_all_lists_every_hit_without_counting():
    rec, _ = _engine()
    a = _seq(200, 1)
    b = a.copy()
    b[100:] = _seq(100, 2)
    rec.add(a)
    rec.add(b)
    before = (rec.short_hits, getattr(rec, "hot_hits", 0), getattr(rec, "cold_hits", 0))
    hits = rec.lookup_all(a[:100])
    assert len(hits) >= 1
    assert (rec.short_hits, getattr(rec, "hot_hits", 0), getattr(rec, "cold_hits", 0)) == before
    one = rec.lookup(a[:100])
    assert one is not None and one[0] in [h[0] for h in hits]


@test
def test_propose_select_equals_propose_with_one_candidate():
    rec1, e1 = _engine()
    rec2, e2 = _engine()
    a = _seq(300, 4)
    for r in (rec1, rec2):
        r.add(a)
    for e in (e1, e2):
        e.begin_stream("r", a[:120])
    p1 = e1.propose("r", n=6)
    p2 = e2.propose_select("r", n=6)
    assert (p1 is None) == (p2 is None)
    if p1 is not None:
        assert p1.tokens.tolist() == p2.tokens.tolist() and p1.match_len == p2.match_len and p1.probe == p2.probe
        assert p1.tokens.tolist() == a[120:126].tolist()
    assert e1.offers == e2.offers and e1.probes == e2.probes


@test
def test_candidates_find_two_continuations_of_one_context():
    rec, e = _engine()
    a = _seq(300, 7)
    b = a.copy()
    b[150:] = _seq(150, 8)
    rec.add(a)
    rec.add(b)
    e.begin_stream("r", a[:150])
    cands = e.candidates("r", n=6, k=4)
    firsts = {int(c[3][0]) for c in cands}
    assert int(a[150]) in firsts and int(b[150]) in firsts, (firsts, a[150], b[150])
    assert e.offers == 0
    p = e.propose_select("r", n=6)
    assert p is not None and int(p.tokens[0]) in (int(a[150]), int(b[150]))


@test
def test_plan_switch_falls_back_without_propose_select_and_under_yee_tree_0():
    class Old:
        def propose(self, rid, n=None):
            return ("old", rid, n)

    assert yt.propose(Old(), "x", n=3) == ("old", "x", 3)

    class New(Old):
        def propose_select(self, rid, n=None, k=4):
            return ("new", rid, n, k)

    os.environ.pop("YEE_TREE", None)
    os.environ.pop("RECALL_ACCEL_TREE", None)
    assert yt.propose(New(), "x", n=3)[0] == "new"
    os.environ["YEE_TREE"] = "0"
    try:
        assert yt.propose(New(), "x", n=3)[0] == "old"
    finally:
        os.environ.pop("YEE_TREE", None)


@test
def test_env_aliases():
    env = {"RECALL_ACCEL_TREE_K": "3"}
    assert yt.tree_k(env) == 3
    env["YEE_TREE_K"] = "1"
    assert yt.tree_k(env) == 1
    assert yt.tree_k({"YEE_TREE_K": "9"}) == 3
    assert yt.tree_nodes(8, {}) == 8 and yt.tree_nodes(8, {"YEE_TREE_NODES": "12"}) == 12
    assert yt.tree_enabled({}) and not yt.tree_enabled({"SGLANG_D_RECALL_TREE": "0"})


@test
def test_bases_carry_the_tree_mode():
    from yee import __main__ as cli

    rows = cli.bases_rows()
    by = {(r["base"], r["head"]): r for r in rows}
    assert by[("mlx-serve", "DSpark (M5 xypd)")]["tree"] == "native"
    assert by[("sglang", "DSPARK")]["tree"] == "select"
    assert by[("vllm", "NGRAM")]["tree"] == "none"
    for r in rows:
        assert r.get("tree") in ("native", "select", "none"), r
    from yee.integrations import registry

    assert registry.tree_mode("sglang", "EAGLE", topk=4)[0] == "none"
    assert registry.tree_mode("sglang", "EAGLE", topk=1)[0] == "select"


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
    print("ALL PASS n=%d" % len(RESULTS) if not fails else "FAILED %d / %d" % (fails, len(RESULTS)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
