"""r648 S1-b2: the worker-side shadow instrument of the vLLM DSpark adapter.

The shadow is off by default (``VLLM_D_RECALL_SHADOW``): then the worker computes nothing, copies
nothing and records nothing.  When it is on it writes one ``wreq`` record per request of a step and
one ``wstep`` record per step, with the draft probabilities of the step (the largest probability of
the draft distribution and the probability of the drafted token itself).

The instrument never synchronises the device: the probabilities are computed on the GPU, copied into
pinned host memory without blocking and read on the CPU only after the event of the copy has
completed.
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
from vllm_d_recall.integrations import shadow  # noqa: E402
from vllm_d_recall.integrations import vllm_dspark as vd  # noqa: E402

MAXR, G, V = 8, 4, 12


# --------------------------------------------------------------------------- fakes
class _Backed:
    def __init__(self, arr):
        self.np = np.asarray(arr, dtype=np.float32)


def runner_with(temps_by_state):
    st = types.SimpleNamespace(temperature=_Backed(temps_by_state))
    return types.SimpleNamespace(sampler=types.SimpleNamespace(sampling_states=st))


def batch(idx):
    idx = np.asarray(idx, dtype=np.int32)
    return types.SimpleNamespace(num_reqs=len(idx), req_ids=[f"r{i}" for i in idx], idx_mapping_np=idx,
                                 idx_mapping=torch.from_numpy(idx.copy()), has_structured_output_reqs=False)


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


class FakeReqStates:
    def __init__(self, total_len):
        self.total_len = types.SimpleNamespace(gpu=torch.as_tensor(total_len, dtype=torch.int64))


class FakeRunner:
    """The runner of the fake worker: the temperatures and the true lengths (one per state row)."""

    def __init__(self, temps, idx, lengths):
        self.sampler = runner_with(temps).sampler
        tl = torch.zeros(MAXR, dtype=torch.int64)
        tl[np.asarray(idx, dtype=np.int64)] = torch.tensor(lengths, dtype=torch.int64)
        self.req_states = FakeReqStates(tl)


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def shadow_env(monkeypatch, tmp_path):
    """The shadow off, a fresh directory and a clean state of both modules."""
    monkeypatch.delenv("VLLM_D_RECALL_SHADOW", raising=False)
    monkeypatch.setenv("VLLM_D_RECALL_SHADOW_DIR", str(tmp_path / "shadow"))
    shadow._reset_for_tests()
    vd.reset_state()
    yield tmp_path
    shadow._reset_for_tests()
    vd.reset_state()


def _shadow_on(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_D_RECALL_SHADOW", "1")
    monkeypatch.setenv("VLLM_D_RECALL_SHADOW_DIR", str(tmp_path / "shadow"))
    shadow._reset_for_tests()
    assert shadow.enabled() is True


def _read(tmp_path):
    """Read the shadow records back; returns (wreqs, wsteps)."""
    recs = list(shadow.load(str(tmp_path / "shadow")))
    wreqs = [r for r in recs if r["k"] == shadow.KIND_WREQ]
    wsteps = [r for r in recs if r["k"] == shadow.KIND_WSTEP]
    return wreqs, wsteps


def _numpy_ref(dl_row, temp, tokens):
    """The numpy reference of one row: (qmax, qx) of the draft distribution (dl_row is [G, V])."""
    s = np.asarray(dl_row, dtype=np.float64) / (temp if temp > 0.0 else 1.0)
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    qmax = p.max(axis=-1)
    qx = p[np.arange(len(tokens)), np.asarray(tokens)]
    return qmax, qx


# --------------------------------------------------------------------------- tests
def test_shadow_off_computes_nothing(shadow_env, monkeypatch, tmp_path):
    """The shadow off: no tensor, no copy, no record (torch.softmax would raise)."""
    def boom(*a, **k):
        raise AssertionError("the shadow computed something while it was off")

    monkeypatch.setattr(torch, "softmax", boom)
    monkeypatch.setattr(torch, "logsumexp", boom)
    idx = [1, 3]
    runner = FakeRunner(make_temps([0.0, 0.7], idx), idx, [10, 20])
    spec = types.SimpleNamespace(draft_logits=torch.randn(MAXR, G, V))
    draft = torch.zeros(len(idx), G, dtype=torch.int64)
    ok = torch.tensor([True, False])
    true_len = torch.tensor([10, 20])

    vd._shadow_collect(spec, runner, torch.tensor(idx), np.asarray(idx), draft, G, ok, true_len, ["a", "b"])
    assert vd._shadow_drain() == 0
    assert len(vd._shadow_state["pending"]) == 0
    assert vd._shadow_state["qm"] is None and vd._shadow_state["okv"] is None
    out = tmp_path / "shadow"
    assert not out.exists() or list(out.iterdir()) == []


def test_shadow_off_in_the_propose_hook_computes_nothing(shadow_env, monkeypatch):
    """The shadow off: the propose hook of the worker does not touch the device either."""
    def boom(*a, **k):
        raise AssertionError("the shadow computed something while it was off")

    monkeypatch.setattr(torch, "softmax", boom)
    monkeypatch.setattr(torch, "logsumexp", boom)
    idx = [0, 2]
    runner = FakeRunner(make_temps([0.0, 0.7], idx), idx, [5, 6])
    spec = types.SimpleNamespace(draft_logits=torch.randn(MAXR, G, V))
    draft = torch.zeros(len(idx), G, dtype=torch.int64)
    last = torch.zeros(MAXR, dtype=torch.int64)
    last[np.asarray(idx, dtype=np.int64)] = 4
    vd._on_propose(spec, runner, batch(idx), draft, last, G)
    assert len(vd._shadow_state["pending"]) == 0
    assert vd._shadow_state["qm"] is None


def test_shadow_on_records_the_probabilities(shadow_env, monkeypatch, tmp_path):
    """The shadow on: the qmax, the qx, the mask and the position are the expected ones."""
    _shadow_on(monkeypatch, tmp_path)
    idx = [1, 3, 5]
    temps = [0.0, 0.7, 0.0]
    runner = FakeRunner(make_temps(temps, idx), idx, [10, 20, 30])
    torch.manual_seed(0)
    dl = torch.randn(MAXR, G, V)
    spec = types.SimpleNamespace(draft_logits=dl)
    draft = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3]], dtype=torch.int64)
    ok = torch.tensor([True, False, True])
    true_len = torch.tensor([10, 20, 30])

    vd._shadow_collect(spec, runner, torch.tensor(idx), np.asarray(idx), draft, G, ok, true_len,
                        ["ra", "rb", "rc"])
    assert len(vd._shadow_state["pending"]) == 1
    assert vd._shadow_flush() == 1
    wreqs, wsteps = _read(tmp_path)
    assert len(wreqs) == 3 and len(wsteps) == 1
    for rec in wreqs + wsteps:
        assert shadow.validate(rec["k"], rec) == [], rec

    by_rid = {r["rid"]: r for r in wreqs}
    assert sorted(by_rid) == ["ra", "rb", "rc"]
    for i, rid in enumerate(["ra", "rb", "rc"]):
        rec = by_rid[rid]
        assert rec["wstep"] == 0
        assert rec["w"] == G
        assert rec["pos"] == [10, 20, 30][i]
        assert rec["conf"] == []
        assert rec["rec"] == (((1 << G) - 1) if [True, False, True][i] else 0)
        qmax_ref, qx_ref = _numpy_ref(dl[[1, 3, 5][i]].numpy(), temps[i], draft[i].numpy())
        assert len(rec["qmax"]) == G and len(rec["qx"]) == G
        assert np.allclose(rec["qmax"], qmax_ref, atol=1e-3)
        assert np.allclose(rec["qx"], qx_ref, atol=1e-3)

    step = wsteps[0]
    assert step["wstep"] == 0 and step["n"] == 3
    assert step["bt"] == -1 and step["fwd_ms"] == -1 and step["drf_ms"] == -1


def test_shadow_on_two_steps(shadow_env, monkeypatch, tmp_path):
    """The shadow on: two steps give two wstep records and the step numbers grow."""
    _shadow_on(monkeypatch, tmp_path)
    idx = [2]
    runner = FakeRunner(make_temps([0.7], idx), idx, [11])
    spec = types.SimpleNamespace(draft_logits=torch.randn(MAXR, G, V))
    draft = torch.zeros(1, G, dtype=torch.int64)
    ok = torch.tensor([True])
    true_len = torch.tensor([11])
    for _ in range(2):
        vd._shadow_collect(spec, runner, torch.tensor(idx), np.asarray(idx), draft, G, ok, true_len, ["rx"])
    assert vd._shadow_flush() == 2
    wreqs, wsteps = _read(tmp_path)
    assert len(wreqs) == 2 and len(wsteps) == 2
    assert [r["wstep"] for r in wsteps] == [0, 1]
    assert [r["wstep"] for r in wreqs] == [0, 1]


def test_shadow_on_without_draft_logits(shadow_env, monkeypatch, tmp_path):
    """No cached draft distribution: the probabilities stay empty and nothing breaks."""
    _shadow_on(monkeypatch, tmp_path)
    idx = [0, 4]
    runner = FakeRunner(make_temps([0.0, 0.7], idx), idx, [3, 4])
    spec = types.SimpleNamespace(draft_logits=None)
    draft = torch.zeros(2, G, dtype=torch.int64)
    ok = torch.tensor([True, False])
    true_len = torch.tensor([3, 4])
    vd._shadow_collect(spec, runner, torch.tensor(idx), np.asarray(idx), draft, G, ok, true_len, ["a", "b"])
    assert vd._shadow_flush() == 1
    wreqs, wsteps = _read(tmp_path)
    assert len(wreqs) == 2 and len(wsteps) == 1
    for rec in wreqs:
        assert rec["qmax"] == [] and rec["qx"] == []
        assert rec["conf"] == []
        assert rec["w"] == G
    assert [r["rec"] for r in wreqs] == [(1 << G) - 1, 0]
    assert [r["pos"] for r in wreqs] == [3, 4]


def test_shadow_on_a_broken_recorder_does_not_break_the_merge(shadow_env, monkeypatch, tmp_path, capsys):
    """A recorder that raises: the merge keeps its result and the shadow is off for good."""
    _shadow_on(monkeypatch, tmp_path)

    class Boom:
        def record(self, kind, rec):
            raise RuntimeError("boom")

    monkeypatch.setattr(shadow, "get_recorder", lambda role, rank=0: Boom())
    idx = [1, 2]
    runner = FakeRunner(make_temps([0.0, 0.7], idx), idx, [10, 20])
    spec = types.SimpleNamespace(draft_logits=torch.randn(MAXR, G, V))
    draft = torch.full((2, G), 7, dtype=torch.int64)
    tl = torch.tensor([10, 20])
    an = torch.tensor([4, 5])
    plan = plan_for([[1, 2, 3, 4], [2, 3, 4, 5]], tl, an)
    ok = vd._merge(spec, runner, torch.tensor(idx), np.asarray(idx), plan, tl, an, draft, G, None)
    assert ok.tolist() == [True, True]
    assert draft.tolist() == [[1, 2, 3, 4], [2, 3, 4, 5]]
    # the broken recorder only breaks the reading of the queue, not the merge
    vd._shadow_collect(spec, runner, torch.tensor(idx), np.asarray(idx), draft, G, ok, tl, ["a", "b"])
    assert vd._shadow_drain() == 0
    assert vd._shadow_state["ok"] is False
    out = capsys.readouterr().out
    assert out.count("worker shadow instrumentation failed") == 1


def test_shadow_on_the_propose_hook_records(shadow_env, monkeypatch, tmp_path):
    """The shadow on: the propose hook of the worker queues the record of the step."""
    _shadow_on(monkeypatch, tmp_path)
    idx = [1, 3]
    runner = FakeRunner(make_temps([0.0, 0.7], idx), idx, [10, 20])
    torch.manual_seed(0)
    spec = types.SimpleNamespace(draft_logits=torch.randn(MAXR, G, V))
    draft = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int64)
    last = torch.zeros(MAXR, dtype=torch.int64)
    last[np.asarray(idx, dtype=np.int64)] = 4
    vd._on_propose(spec, runner, batch(idx), draft, last, G)
    assert len(vd._shadow_state["pending"]) == 1
    assert vd._shadow_flush() == 1
    wreqs, wsteps = _read(tmp_path)
    assert len(wreqs) == 2 and len(wsteps) == 1
    assert [r["rid"] for r in wreqs] == ["r1", "r3"]
    assert [r["pos"] for r in wreqs] == [10, 20]


def test_shadow_without_an_event_reads_the_records(shadow_env, monkeypatch, tmp_path):
    """No event (the CPU-only path): the entry is read right away and the records are there."""
    _shadow_on(monkeypatch, tmp_path)
    monkeypatch.setattr(vd, "_event_factory", lambda dev: None)
    idx = [0]
    runner = FakeRunner(make_temps([0.5], idx), idx, [7])
    spec = types.SimpleNamespace(draft_logits=torch.randn(MAXR, G, V))
    draft = torch.zeros(1, G, dtype=torch.int64)
    ok = torch.tensor([True])
    true_len = torch.tensor([7])
    vd._shadow_collect(spec, runner, torch.tensor(idx), np.asarray(idx), draft, G, ok, true_len, ["rz"])
    assert vd._shadow_flush() == 1
    wreqs, _wsteps = _read(tmp_path)
    assert len(wreqs) == 1
    assert wreqs[0]["rid"] == "rz" and wreqs[0]["pos"] == 7
    assert wreqs[0]["rec"] == (1 << G) - 1
    assert len(wreqs[0]["qmax"]) == G and len(wreqs[0]["qx"]) == G


def test_shadow_the_queue_waits_for_the_oldest_entry(shadow_env, monkeypatch, tmp_path):
    """A long queue: the next step waits for the oldest entry instead of growing the queue."""
    _shadow_on(monkeypatch, tmp_path)
    idx = [0]
    runner = FakeRunner(make_temps([0.5], idx), idx, [7])
    spec = types.SimpleNamespace(draft_logits=torch.randn(MAXR, G, V))
    draft = torch.zeros(1, G, dtype=torch.int64)
    ok = torch.tensor([True])
    true_len = torch.tensor([7])

    class Slow:
        """An event that is never ready until it has been waited for."""

        def __init__(self):
            self.ready = False

        def record(self):
            return None

        def query(self):
            return self.ready

        def synchronize(self):
            self.ready = True

    events = []

    def factory(dev):
        ev = Slow()
        events.append(ev)
        return ev

    monkeypatch.setattr(vd, "_event_factory", factory)
    for _ in range(vd.SHADOW_QUEUE_MAX + 2):
        vd._shadow_collect(spec, runner, torch.tensor(idx), np.asarray(idx), draft, G, ok, true_len, ["rz"])
    assert len(vd._shadow_state["pending"]) == vd.SHADOW_QUEUE_MAX + 2
    # the drain of a long queue waits for the oldest entry and then reads everything
    assert vd._shadow_drain(block=True) == vd.SHADOW_QUEUE_MAX + 2
    assert len(vd._shadow_state["pending"]) == 0
    shadow.get_recorder("worker", 0).flush()
    recs = list(shadow.load(str(tmp_path / "shadow")))
    wreqs = [r for r in recs if r["k"] == shadow.KIND_WREQ]
    wsteps = [r for r in recs if r["k"] == shadow.KIND_WSTEP]
    assert len(wreqs) == vd.SHADOW_QUEUE_MAX + 2
    assert len(wsteps) == vd.SHADOW_QUEUE_MAX + 2


def test_shadow_the_merge_result_is_untouched(shadow_env, monkeypatch, tmp_path):
    """The shadow on and off: the drafts and the recall mask of the step are the same."""
    _shadow_on(monkeypatch, tmp_path)
    idx = [1, 2]
    runner = FakeRunner(make_temps([0.0, 0.7], idx), idx, [10, 20])
    spec = types.SimpleNamespace(draft_logits=torch.randn(MAXR, G, V))
    draft = torch.full((2, G), 7, dtype=torch.int64)
    tl = torch.tensor([10, 20])
    an = torch.tensor([4, 5])
    plan = plan_for([[1, 2, 3, 4], [2, 3, 4, 5]], tl, an)
    ok = vd._merge(spec, runner, torch.tensor(idx), np.asarray(idx), plan, tl, an, draft, G, None)
    before = draft.clone()
    vd._shadow_collect(spec, runner, torch.tensor(idx), np.asarray(idx), draft, G, ok, tl, ["a", "b"])
    assert torch.equal(draft, before)
    assert ok.tolist() == [True, True]
