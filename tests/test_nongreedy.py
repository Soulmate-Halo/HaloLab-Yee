"""Sampled requests taking recall copies (r593). Run: python tests/test_nongreedy.py

* reference rejection samplers (numpy): a copied id as a point-mass proposal keeps the target distribution, for the
  target-only rule (SGLang EAGLE/NGRAM/DFLASH) and for the q-ratio rule with q made one-hot (SGLang DSPARK, vLLM);
  the q-ratio rule with a STALE q (the drafter's own) does not - the reason DSPARK rows are rewritten.
* ``sglang_dspark.onehot_rows`` and the DSPARK / vLLM / common gates.
"""
import os
import sys
import traceback
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from yee.integrations import sglang_common as sc  # noqa: E402
from yee.integrations import sglang_dspark as sd  # noqa: E402
from yee.integrations import vllm_spec as vd  # noqa: E402

RESULTS = []
V = 12


def test(fn):
    RESULTS.append(fn)
    return fn
test.__test__ = False  # decorator, not a test (pytest)


def target_dist(logits, temp, top_k=0, top_p=1.0):
    z = np.asarray(logits, dtype=np.float64) / temp
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-p, kind="stable")
    keep = np.ones(V, dtype=bool)
    if top_k:
        keep[:] = False
        keep[order[:top_k]] = True
    if top_p < 1.0:
        c = np.cumsum(p[order])
        cut = np.zeros(V, dtype=bool)
        cut[order[: int(np.searchsorted(c, top_p) + 1)]] = True
        keep &= cut
    p = np.where(keep, p, 0.0)
    return p / p.sum()


def accept_first(p, x, q, rng):
    """One position of chain speculative sampling: proposal x drawn with prob q[x]; returns the emitted id."""
    if rng.random() < min(1.0, p[x] / q[x]):
        return x
    r = np.maximum(p - q, 0.0)
    return int(rng.choice(V, p=r / r.sum()))


def chi2_p(counts, probs):
    n = counts.sum()
    m = probs > 0
    exp = probs[m] * n
    stat = float((((counts[m] - exp) ** 2) / exp).sum())
    k = int(m.sum()) - 1
    if k == 0:  # a single id carries all of p: only "nothing outside it" can be checked
        return 1.0, int(counts[~m].sum())
    # Wilson-Hilferty normal approximation of the chi-square upper tail
    z = ((stat / k) ** (1 / 3) - (1 - 2 / (9 * k))) / np.sqrt(2 / (9 * k))
    from math import erf, sqrt
    return 0.5 * (1 - erf(z / sqrt(2))), int(counts[~m].sum())


def emitted_counts(p, x, q, n, seed):
    rng = np.random.default_rng(seed)
    c = np.zeros(V)
    for _ in range(n):
        c[accept_first(p, x, q, rng)] += 1
    return c


CASES = [(0.6, 0, 1.0), (1.0, 0, 1.0), (1.0, 5, 1.0), (0.8, 0, 0.9), (1.3, 4, 0.95)]
LOGITS = np.random.default_rng(3).normal(0, 0.8, V)


@test
def onehot_proposal_keeps_target_distribution():
    for temp, k, tp in CASES:
        p = target_dist(LOGITS, temp, k, tp)
        for x in (int(np.argmax(p)), int(np.argsort(-p)[2]), int(np.argmin(p))):  # incl. an id p gives 0
            q = np.zeros(V)
            q[x] = 1.0
            c = emitted_counts(p, x, q, 20000, seed=x * 7 + int(temp * 10))
            pv, outside = chi2_p(c, p)
            assert outside == 0 and pv > 0.01, (temp, k, tp, x, pv, outside)


@test
def target_only_rule_equals_onehot_rule():
    """SGLang's target-only kernel: accept with p(x), residual relu(p - 0) without x == one-hot q rule."""
    for temp, k, tp in CASES:
        p = target_dist(LOGITS, temp, k, tp)
        x = int(np.argsort(-p)[1])
        rng = np.random.default_rng(11)
        c = np.zeros(V)
        for _ in range(20000):
            if rng.random() < p[x]:
                c[x] += 1
            else:
                r = p.copy()
                r[x] = 0.0
                c[int(rng.choice(V, p=r / r.sum()))] += 1
        pv, outside = chi2_p(c, p)
        assert outside == 0 and pv > 0.01, (temp, k, tp, pv)


@test
def stale_q_breaks_the_distribution():
    """A copied id judged against the drafter's own q (not one-hot) is NOT lossless: why DSPARK rows change."""
    p = target_dist(LOGITS, 1.0)
    q = target_dist(np.random.default_rng(9).normal(0, 2, V), 1.0)
    x = int(np.argmax(p))  # the copied id; q gives it little mass, so it is over-accepted
    c = emitted_counts(p, x, q, 20000, seed=5)
    pv, _ = chi2_p(c, p)
    assert pv < 1e-6, pv


@test
def onehot_rows_rewrites_only_copied_rows():
    import torch

    bs, g = 3, 4
    lg = torch.randn(bs, g, V)
    before = lg.clone()
    tok = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 0]], dtype=torch.int32)
    ok = torch.tensor([True, False, True])
    sd.onehot_rows(lg, ok, tok)
    assert torch.equal(lg[1], before[1])
    for b in (0, 2):
        q = torch.softmax(lg[b] / 0.7, dim=-1)
        for j in range(g):
            assert abs(float(q[j, int(tok[b, j])]) - 1.0) < 1e-6
            assert float(q[j].sum()) > 0.999999
    big = torch.randn(5, 6, V)  # a buffer view wider than the batch: rows past bs untouched
    keep = big[3:].clone()
    sd.onehot_rows(big, ok, tok)
    assert torch.equal(big[3:], keep)
    sd.onehot_rows(None, ok, tok)  # no logits: nothing to do


class _SI:
    def __init__(self, greedy):
        self.is_all_greedy = greedy


def _dspark_kw(greedy, corrected):
    import torch

    batch = types.SimpleNamespace(has_grammar=False)
    dt = torch.zeros((2, 3), dtype=torch.int64)
    blk = types.SimpleNamespace(draft_tokens=dt, corrected_logits=corrected)
    return {"batch": batch, "bs": 2, "sampling_info": _SI(greedy)}, types.SimpleNamespace(draft_block=blk)


class NonGreedy:
    def __init__(self, v):
        self.v = v

    def __enter__(self):
        self.old = os.environ.get("RECALL_ACCEL_NONGREEDY")
        os.environ["RECALL_ACCEL_NONGREEDY"] = self.v

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("RECALL_ACCEL_NONGREEDY", None)
        else:
            os.environ["RECALL_ACCEL_NONGREEDY"] = self.old


@test
def dspark_gate():
    import torch

    cl = torch.zeros((2, 3, V))
    assert sd._participates(*_dspark_kw(True, None))
    assert sd._participates(*_dspark_kw(False, cl))
    assert not sd._participates(*_dspark_kw(False, None))  # no q to make one-hot
    with NonGreedy("0"):
        assert not sd._participates(*_dspark_kw(False, cl))
        assert sd._participates(*_dspark_kw(True, None))


@test
def common_gate():
    b = types.SimpleNamespace(has_grammar=False, sampling_info=_SI(False))
    assert sc.copy_batch(b)
    with NonGreedy("off"):
        assert not sc.copy_batch(b)
    assert sc.copy_batch(types.SimpleNamespace(has_grammar=False, sampling_info=_SI(True)))
    assert not sc.copy_batch(types.SimpleNamespace(has_grammar=True, sampling_info=_SI(True)))
    assert not sc.copy_batch(types.SimpleNamespace(has_grammar=False, sampling_info=None))
    assert sc.nongreedy_enabled({}) and not sc.nongreedy_enabled({"RECALL_ACCEL_NONGREEDY": "0"})


@test
def vllm_gate():
    vd._state.pop("spec", None)
    spec = types.SimpleNamespace(draft_logits=None)
    assert vd._nongreedy_ok(spec, None)
    assert not vd._nongreedy_ok(types.SimpleNamespace(draft_logits=object()), None)
    assert not vd._nongreedy_ok(None, types.SimpleNamespace())  # speculator unknown: stay greedy-only
    assert vd._nongreedy_ok(None, types.SimpleNamespace(speculator=spec))
    with NonGreedy("0"):
        assert not vd._nongreedy_ok(spec, None)


def main():
    fails = skips = 0
    for fn in RESULTS:
        try:
            fn()
            print("  PASS", fn.__name__)
        except ImportError as e:
            skips += 1
            print("  SKIP %s (%s)" % (fn.__name__, e))
        except Exception:
            fails += 1
            print("> FAIL", fn.__name__)
            traceback.print_exc()
    print("ALL PASS n=%d skip=%d" % (len(RESULTS), skips) if not fails else "FAILED %d / %d" % (fails, len(RESULTS)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
