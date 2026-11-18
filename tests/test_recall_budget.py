"""Yee 4.0: recall-aware verify budget (CPU only; needs torch, skipped without it).

The budget planner of compact verify sees the confidence of two steps ago; rank0 records which pool rows carried
a recall plan in the latest step (``_state["recall_hot"]``) and the wrapped ``compute_budget`` treats those rows
as certain (confidence 1.0, like ``device_merge`` already does for the row allocation of the current step).
"""
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU tensors only

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from yee.integrations import sglang_dspark as sd  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    saved = dict(sd._state)
    saved_counts = sd._counts.copy()
    monkeypatch.delenv("YEE_RECALL_BUDGET", raising=False)
    yield
    sd._state.clear()
    sd._state.update(saved)
    sd._counts.clear()
    sd._counts.update(saved_counts)


def _fake_module():
    calls = []

    class HostConfidenceBudgetPlanner:
        def compute_budget(self, *, confidence, generation, current_generation, req_pool_indices_cpu):
            calls.append(confidence)
            return int(round(float(confidence.sum())))

    mod = types.ModuleType("fake_dspark_planner")
    mod.HostConfidenceBudgetPlanner = HostConfidenceBudgetPlanner
    return mod, calls


def _kw(conf, pools):
    return dict(confidence=conf, generation=torch.ones(len(pools), dtype=torch.int64),
                current_generation=torch.ones(len(pools), dtype=torch.int64),
                req_pool_indices_cpu=torch.tensor(pools, dtype=torch.int64))


def test_recall_aware_confidence_sets_hot_rows_only():
    conf = torch.full((3, 6), 0.25)
    out, n = sd.recall_aware_confidence(conf, torch.tensor([7, 3, 9]), frozenset({3, 9}))
    assert n == 2
    assert torch.equal(out[0], torch.full((6,), 0.25))
    assert torch.equal(out[1], torch.ones(6)) and torch.equal(out[2], torch.ones(6))
    assert torch.equal(conf, torch.full((3, 6), 0.25)), "caller's tensor must not change"
    assert out is not conf


def test_recall_aware_confidence_no_change_returns_same_object():
    conf = torch.full((2, 6), 0.5)
    for hot in (frozenset(), frozenset({42})):
        out, n = sd.recall_aware_confidence(conf, torch.tensor([1, 2]), hot)
        assert out is conf and n == 0
    out, n = sd.recall_aware_confidence(torch.ones(6), torch.tensor([1]), frozenset({1}))
    assert n == 0
    out, n = sd.recall_aware_confidence(None, torch.tensor([1]), frozenset({1}))
    assert out is None and n == 0


def test_recall_aware_confidence_ignores_pool_rows_past_the_batch():
    conf = torch.zeros((2, 6))
    out, n = sd.recall_aware_confidence(conf, torch.tensor([5, 6, 7]), frozenset({7}))
    assert out is conf and n == 0


def test_hot_rows_skips_unflagged_and_missing_pool_index():
    reqs = [types.SimpleNamespace(req_pool_idx=4), types.SimpleNamespace(req_pool_idx=None),
            types.SimpleNamespace(), types.SimpleNamespace(req_pool_idx="x"), types.SimpleNamespace(req_pool_idx=8)]
    assert sd._hot_rows(reqs, [1, 1, 1, 1, 0]) == frozenset({4})
    assert sd._hot_rows(reqs, [0, 0, 0, 0, 1]) == frozenset({8})


def test_hot_rows_reads_the_nested_kv_pool_index():
    """The deployed SGLang tree keeps the pool row under Req.kv (ReqKvInfo), not flat on the Req."""
    reqs = [types.SimpleNamespace(kv=types.SimpleNamespace(req_pool_idx=4)),
            types.SimpleNamespace(kv=types.SimpleNamespace(req_pool_idx=None)),
            types.SimpleNamespace(kv=None),
            types.SimpleNamespace(req_pool_idx=8, kv=types.SimpleNamespace(req_pool_idx=99))]
    assert sd._hot_rows(reqs, [1, 1, 1, 1]) == frozenset({4, 8})


def test_planner_hook_feeds_modified_confidence():
    mod, calls = _fake_module()
    sd._patch_planner_module(mod)
    sd._state["recall_hot"] = frozenset({11})
    conf = torch.full((2, 6), 0.5)
    res = mod.HostConfidenceBudgetPlanner().compute_budget(**_kw(conf, [10, 11]))
    assert res == 9  # 6 * 0.5 + 6 * 1.0
    assert calls[-1] is not conf and torch.equal(calls[-1][1], torch.ones(6))
    assert torch.equal(conf, torch.full((2, 6), 0.5))
    assert sd._counts["budget_calls"] == 1 and sd._counts["budget_hot_rows"] == 1


def test_planner_hook_passes_through_when_off_or_cold(monkeypatch):
    mod, calls = _fake_module()
    sd._patch_planner_module(mod)
    planner = mod.HostConfidenceBudgetPlanner()
    conf = torch.full((2, 6), 0.5)
    sd._state["recall_hot"] = frozenset()
    planner.compute_budget(**_kw(conf, [10, 11]))
    assert calls[-1] is conf
    sd._state["recall_hot"] = frozenset({10, 11})
    monkeypatch.setenv("YEE_RECALL_BUDGET", "0")
    planner.compute_budget(**_kw(conf, [10, 11]))
    assert calls[-1] is conf
    assert sd._counts["budget_hot_rows"] == 0 and sd._counts["budget_calls"] == 2


def test_planner_hook_is_not_stacked_twice():
    mod, calls = _fake_module()
    sd._patch_planner_module(mod)
    first = mod.HostConfidenceBudgetPlanner.compute_budget
    sd._patch_planner_module(mod)
    assert mod.HostConfidenceBudgetPlanner.compute_budget is first
    mod.HostConfidenceBudgetPlanner().compute_budget(**_kw(torch.zeros((1, 6)), [1]))
    assert sd._counts["budget_calls"] == 1


def test_planner_hook_tolerates_a_module_without_the_class(capsys):
    sd._patch_planner_module(types.ModuleType("empty_planner"))
    assert "budget hook off" in capsys.readouterr().out


def test_planner_target_registered():
    assert sd.TARGETS["sglang.srt.speculative.dspark_components.dspark_planner"] is sd._patch_planner_module


class _Group:
    def __init__(self, rank):
        self.rank_in_group = rank
        self.broadcasts = 0

    def broadcast(self, t, src=0):
        self.broadcasts += 1


def _propose_inputs(bs, gamma):
    reqs = [types.SimpleNamespace(rid="r%d" % i, req_pool_idx=20 + i) for i in range(bs)]
    batch = types.SimpleNamespace(reqs=reqs, seq_lens=torch.full((bs,), 10, dtype=torch.int64))
    block = types.SimpleNamespace(draft_tokens=torch.zeros((bs, gamma), dtype=torch.int64))
    out = types.SimpleNamespace(draft_block=block, draft_block_ids=torch.zeros((bs, gamma + 1), dtype=torch.int64),
                                confidence=torch.full((bs, gamma), 0.5))
    return {"batch": batch, "bs": bs}, out


def _stub_engine(monkeypatch, flags, gamma):
    width = 2 + gamma + sd._pad(gamma)

    def build_plan(eng, reqs, g, paused=False):
        arr = np.zeros((len(reqs), width), dtype=np.int64)
        arr[:, 0] = flags
        return arr, [(r.rid, int(f), 0, None, 0) for r, f in zip(reqs, flags)]

    monkeypatch.setattr(sd, "build_plan", build_plan)
    monkeypatch.setattr(sd, "_get_engine", lambda g: object())
    monkeypatch.setattr(sd, "_check_pause", lambda eng: False)
    monkeypatch.setattr(sd, "_pinned", lambda w: torch.zeros((sd.MAX_BS, w), dtype=torch.int64))
    monkeypatch.setattr(sd, "device_merge", lambda *a, **k: torch.zeros(len(flags), dtype=torch.bool))
    sd._fifo.clear()


def test_on_propose_rank0_records_hot_rows(monkeypatch):
    gamma = 6
    _stub_engine(monkeypatch, [1, 0, 1], gamma)
    sd._state["group"] = _Group(0)
    kw, out = _propose_inputs(3, gamma)
    sd._on_propose(None, kw, out)
    assert sd._state["recall_hot"] == frozenset({20, 22})
    assert sd._state["group"].broadcasts == 1
    _stub_engine(monkeypatch, [0, 0, 0], gamma)
    sd._on_propose(None, kw, out)
    assert sd._state["recall_hot"] == frozenset()


def test_on_propose_other_ranks_leave_hot_rows_alone(monkeypatch):
    gamma = 6
    _stub_engine(monkeypatch, [1, 1, 1], gamma)
    sd._state["group"] = _Group(2)
    sentinel = frozenset({999})
    sd._state["recall_hot"] = sentinel
    kw, out = _propose_inputs(3, gamma)
    sd._on_propose(None, kw, out)
    assert sd._state["recall_hot"] is sentinel


def test_on_propose_plan_error_clears_hot_rows(monkeypatch):
    gamma = 6
    _stub_engine(monkeypatch, [1, 1, 1], gamma)

    def boom(*a, **k):
        raise RuntimeError("plan failed")

    monkeypatch.setattr(sd, "build_plan", boom)
    sd._state["group"] = _Group(0)
    sd._state["recall_hot"] = frozenset({20})
    kw, out = _propose_inputs(3, gamma)
    sd._on_propose(None, kw, out)
    assert sd._state["recall_hot"] == frozenset()


# ---------------------------------------------------------------- STS collector under folded proposals

class _FakeRecorder:
    instances = []

    def __init__(self, *, path_stem, gamma, flush_every):
        self.path_stem, self.gamma, self.flush_every = path_stem, gamma, flush_every
        self.calls = []
        _FakeRecorder.instances.append(self)

    def record(self, *, confidence_raw, num_correct_drafts):
        self.calls.append((confidence_raw.clone(), num_correct_drafts.clone()))


def _sts_env(monkeypatch, rank=0):
    fake = types.ModuleType("sglang.srt.speculative.dspark_components.dspark_sts")
    fake.StsDataRecorder = _FakeRecorder
    monkeypatch.setitem(sys.modules, "sglang.srt.speculative.dspark_components.dspark_sts", fake)
    monkeypatch.setenv("YEE_STS_COLLECT_PATH", "/state/x/sts")
    _FakeRecorder.instances.clear()
    sd._state["sts_recorder"] = None
    sd._state["group"] = _Group(rank)
    mod = types.ModuleType("fake_obs")

    class DsparkStepObservers:
        def observe_verify_step(self, **kw):
            return "orig"

    mod.DsparkStepObservers = DsparkStepObservers
    sd._patch_observability_module(mod)
    return mod.DsparkStepObservers()


def _obs_kw(conf, correct, pools):
    return dict(bs=len(pools), confidence=conf, correct_len=torch.tensor(correct, dtype=torch.int32),
                req_pool_indices=torch.tensor(pools, dtype=torch.int64), proposal_folded=True)


def test_sts_collector_records_logits_and_skips_recall_rows(monkeypatch):
    obs = _sts_env(monkeypatch)
    sd._state["recall_hot"] = frozenset({21})
    conf = torch.tensor([[0.9] * 6, [1.0] * 6, [0.2] * 6])
    assert obs.observe_verify_step(**_obs_kw(conf, [5, 6, 1], [20, 21, 22])) == "orig"
    rec = _FakeRecorder.instances[0]
    assert rec.path_stem == "/state/x/sts" and rec.gamma == 6
    raw, cnt = rec.calls[0]
    assert raw.shape == (2, 6) and cnt.tolist() == [5, 1]
    assert torch.allclose(torch.sigmoid(raw[0]), torch.full((6,), 0.9), atol=1e-5)
    assert sd._counts["sts_rows"] == 2 and sd._counts["sts_skipped_recall"] == 1


def test_sts_collector_only_on_rank0_and_only_when_enabled(monkeypatch):
    obs = _sts_env(monkeypatch, rank=1)
    obs.observe_verify_step(**_obs_kw(torch.full((1, 6), 0.5), [3], [5]))
    assert not _FakeRecorder.instances
    sd._state["group"] = _Group(0)
    monkeypatch.delenv("YEE_STS_COLLECT_PATH")
    obs.observe_verify_step(**_obs_kw(torch.full((1, 6), 0.5), [3], [5]))
    assert not _FakeRecorder.instances


def test_sts_collector_errors_never_reach_serving(monkeypatch):
    obs = _sts_env(monkeypatch)
    bad = dict(_obs_kw(torch.full((1, 6), 0.5), [3], [5]), correct_len="not a tensor")
    assert obs.observe_verify_step(**bad) == "orig"
    assert sd._counts["sts_errors"] == 1


def test_observability_target_registered():
    assert (sd.TARGETS["sglang.srt.speculative.dspark_components.dspark_observability"]
            is sd._patch_observability_module)
