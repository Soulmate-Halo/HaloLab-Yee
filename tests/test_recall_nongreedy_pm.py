"""r648 S4-4.1a: recall copies on temperature > 0 rows with the point-mass draft-distribution fix.

VLLM_D_RECALL_NONGREEDY unset/0 keeps the legacy rule (copies only when the whole batch is greedy). With 1,
every row may take copies; for probabilistic drafting (spec.draft_logits cached, rows addressed by
req_state_idx) each copied position of a temperature > 0 row becomes a point mass on the copied id, which keeps
standard and block verification lossless (.diagnostics/r648/s5/RESULT_S5A.md).
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from vllm_d_recall.integrations import vllm_dspark as vd  # noqa: E402

MAXR, G, V = 6, 4, 10


class _Backed:
    def __init__(self, arr):
        self.np = np.asarray(arr, dtype=np.float32)


def runner_with(temps_by_state):
    st = types.SimpleNamespace(temperature=_Backed(temps_by_state))
    return types.SimpleNamespace(sampler=types.SimpleNamespace(sampling_states=st))


def batch(idx, structured=False):
    idx = np.asarray(idx, dtype=np.int32)
    return types.SimpleNamespace(num_reqs=len(idx), req_ids=[f"r{i}" for i in idx], idx_mapping_np=idx,
                                 idx_mapping=torch.from_numpy(idx.copy()), has_structured_output_reqs=structured)


def make_temps(temps_rows, idx):
    t = np.zeros(MAXR, np.float32)
    t[np.asarray(idx)] = temps_rows
    return t


def plan_for(rows_tokens, true_len, anchors, gamma=G):
    plan = torch.zeros((len(rows_tokens), 2 + vd.width(gamma)), dtype=torch.int64)
    for i, toks in enumerate(rows_tokens):
        if toks is None:
            continue
        plan[i, 0] = 1
        plan[i, 1] = int(true_len[i])  # offset 0
        plan[i, 2] = int(anchors[i])
        plan[i, 3:3 + gamma] = torch.tensor(toks)
    return plan


@pytest.fixture
def flag(monkeypatch):
    def set_(v):
        if v is None:
            monkeypatch.delenv(vd.ENV_NONGREEDY, raising=False)
        else:
            monkeypatch.setenv(vd.ENV_NONGREEDY, v)
    return set_


CASES = [([0.0, 0.0, 0.0], [2, 0, 5]), ([0.0, 0.7, 0.7], [2, 0, 5]), ([0.7, 0.7, 0.7], [0, 1, 2])]


def test_flag_off_equals_legacy_rule(flag):
    for v in (None, "0", "off"):
        flag(v)
        for temps, idx in CASES:
            r = runner_with(make_temps(temps, idx))
            legacy = bool(np.all(np.asarray(temps) == 0.0))
            assert vd.participates(r, batch(idx), torch.zeros(len(idx), G, dtype=torch.int64), False, False) is legacy


def test_flag_on_mixed_batches_participate(flag):
    flag("1")
    for temps, idx in CASES:
        r = runner_with(make_temps(temps, idx))
        assert vd.participates(r, batch(idx), torch.zeros(len(idx), G, dtype=torch.int64), False, False) is True
    temps, idx = CASES[1]
    r = runner_with(make_temps(temps, idx))
    dt = torch.zeros(3, G, dtype=torch.int64)
    assert vd.participates(r, batch(idx, structured=True), dt, False, False) is False  # whole-batch exclusion kept
    assert vd.participates(r, batch(idx), dt, True, False) is False  # dummy run
    assert vd.participates(r, batch(idx), dt, False, True) is False  # profile run


def test_pointmass_rows_and_columns(flag):
    flag("1")
    torch.manual_seed(0)
    idx = [2, 0, 5]
    r = runner_with(make_temps([0.0, 0.7, 0.7], idx))
    spec = types.SimpleNamespace(draft_logits=torch.randn(MAXR, G, V))
    before = spec.draft_logits.clone()
    draft = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3]], dtype=torch.int64)
    ok = torch.tensor([True, True, False])
    vd._fix_pointmass(spec, r, torch.tensor(idx), np.asarray(idx), draft, ok)
    dl = spec.draft_logits
    for j in range(G):  # batch row 1 -> state row 0
        row = dl[0, j]
        assert row[draft[1, j]] == 0.0
        rest = torch.ones(V, dtype=torch.bool)
        rest[draft[1, j]] = False
        assert torch.isneginf(row[rest]).all()
    for s in (1, 2, 3, 4, 5):  # greedy row (state 2), not-copied row (state 5), rows outside the batch
        assert torch.equal(dl[s], before[s])
    q = torch.softmax(dl[0] / 0.7, dim=-1)  # the verifier divides by the temperature first
    assert torch.allclose(q.max(dim=-1).values, torch.ones(G))


def test_pointmass_only_first_k_columns(flag):
    flag("1")
    idx = [4]
    r = runner_with(make_temps([1.0], idx))
    spec = types.SimpleNamespace(draft_logits=torch.randn(MAXR, G, V))
    before = spec.draft_logits.clone()
    draft = torch.tensor([[3, 4, 5]], dtype=torch.int64)  # narrower buffer than the cache
    vd._fix_pointmass(spec, r, torch.tensor(idx), np.asarray(idx), draft, torch.tensor([True]))
    assert (spec.draft_logits[4, :3].argmax(-1) == torch.tensor([3, 4, 5])).all()
    assert torch.equal(spec.draft_logits[4, 3], before[4, 3])


def test_no_draft_logits_is_noop(flag):
    flag("1")
    idx = [0, 1]
    r = runner_with(make_temps([0.7, 0.7], idx))
    for spec in (types.SimpleNamespace(draft_logits=None), types.SimpleNamespace()):
        vd._fix_pointmass(spec, r, torch.tensor(idx), np.asarray(idx), torch.zeros(2, G, dtype=torch.int64),
                          torch.tensor([True, True]))


def _merge_setup(temps=(0.0, 0.8), idx=(3, 1)):
    r = runner_with(make_temps(list(temps), list(idx)))
    spec = types.SimpleNamespace(draft_logits=torch.randn(MAXR, G, V))
    draft = torch.full((2, G), 7, dtype=torch.int64)
    tl = torch.tensor([10, 20])
    an = torch.tensor([4, 5])
    plan = plan_for([[1, 2, 3, 4], [2, 3, 4, 5]], tl, an)
    return r, spec, draft, tl, an, plan, torch.tensor(list(idx)), np.asarray(idx)


def test_merge_copies_and_pointmass(flag):
    flag("1")
    r, spec, draft, tl, an, plan, idx, idx_np = _merge_setup()
    before = spec.draft_logits.clone()
    ok = vd._merge(spec, r, idx, idx_np, plan, tl, an, draft, G, None)
    assert ok.tolist() == [True, True]
    assert draft.tolist() == [[1, 2, 3, 4], [2, 3, 4, 5]]
    assert torch.equal(spec.draft_logits[3], before[3])  # greedy row: cache untouched
    assert (spec.draft_logits[1].argmax(-1) == torch.tensor([2, 3, 4, 5])).all()


def test_merge_fix_error_drops_copies(flag, monkeypatch):
    flag("1")
    r, spec, draft, tl, an, plan, idx, idx_np = _merge_setup()
    before = spec.draft_logits.clone()

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(vd, "_fix_pointmass", boom)
    vd._merge(spec, r, idx, idx_np, plan, tl, an, draft, G, None)
    assert draft.tolist() == [[7] * G, [7] * G]
    assert torch.equal(spec.draft_logits, before)


def test_merge_flag_off_is_device_merge(flag):
    flag(None)
    r, spec, draft, tl, an, plan, idx, idx_np = _merge_setup(temps=(0.0, 0.0))
    before = spec.draft_logits.clone()
    ref = draft.clone()
    ok_ref = vd.device_merge(plan.clone(), tl, an, ref, G)
    ok = vd._merge(spec, r, idx, idx_np, plan, tl, an, draft, G, None)
    assert torch.equal(ok, ok_ref) and torch.equal(draft, ref)
    assert torch.equal(spec.draft_logits, before)
