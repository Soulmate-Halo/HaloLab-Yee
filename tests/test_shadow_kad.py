"""The shadow instrument of the vLLM KAD scheduler adapter: it only records, it never changes.

The shadow is off by default (``VLLM_D_RECALL_SHADOW``): then nothing is written and the width
decisions of KAD are exactly the ones of a build without the shadow.  When it is on, the adapter
writes one ``step`` record per scheduling step, one ``req`` record per request that carries
drafts and one ``acc`` record per verified request.
"""
from __future__ import annotations

import pytest

from vllm_d_recall.integrations import shadow
from vllm_d_recall.integrations import vllm_kad as vk

STEPS = 2
KMAX = 12


# --------------------------------------------------------------------------- fakes
class FakeSP:
    """The sampling parameters of a request."""

    def __init__(self, temperature, top_p, top_k):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k


class FakeReq:
    """A scheduled request of the fake scheduler."""

    def __init__(self, rid, spec, computed, prompt, sp=None, structured=False):
        self.request_id = rid
        self.spec_token_ids = list(spec)
        self.num_computed_tokens = computed
        self.num_prompt_tokens = prompt
        self.sampling_params = sp
        self.use_structured_output = structured
        self.next_decode_eligible_step = 0
        self.is_prefill_chunk = False


class FakeSched:
    """The fake scheduler: the requests, the draft budget and the step counter."""

    def __init__(self, requests, kmax=KMAX):
        self.requests = {r.request_id: r for r in requests}
        self.running = list(requests)
        self.num_spec_tokens = kmax
        self.current_step = 0


class FakeSO:
    """The fake SchedulerOutput."""

    def __init__(self, num_scheduled_tokens, spec_tokens):
        self.num_scheduled_tokens = dict(num_scheduled_tokens)
        self.scheduled_spec_decode_tokens = dict(spec_tokens)


class FakeMRO:
    """The fake ModelRunnerOutput: the request index and the sampled tokens."""

    def __init__(self, idx_map, sampled):
        self.req_id_to_index = dict(idx_map)
        self.sampled_token_ids = [list(s) for s in sampled]


# --------------------------------------------------------------------------- scenario
def _requests():
    """Four decode requests with drafts, one without, one without sampling params, one structured."""
    reqs = [
        FakeReq("d1", range(12), 10, 8, FakeSP(0.7, 0.9, 0)),
        FakeReq("d2", range(12), 20, 8, FakeSP(1.0, 1.0, None)),
        FakeReq("d3", range(12), 30, 8, FakeSP(0.0, 0.5, 20)),
        FakeReq("d4", [], 40, 8, FakeSP(0.6, 0.8, 40)),
        FakeReq("d5", range(12), 50, 8, None),
        FakeReq("p1", [], 3, 100, None),
        FakeReq("s1", range(12), 60, 8, FakeSP(0.4, 0.7, 5), structured=True),
    ]
    reqs[5].is_prefill_chunk = True
    return reqs


NUM_SCHEDULED = {"d1": 13, "d2": 13, "d3": 13, "d4": 1, "d5": 13, "p1": 97, "s1": 13}
IDX_MAP = {"d1": 0, "d2": 1, "d3": 2, "d5": 3}
SAMPLED = [[1] * 13, [2] * 3, [3] * 13, [5] * 5]


def _spec_tokens(reqs):
    """The tokens the scheduler really verifies: the (shortened) draft lists."""
    out = {}
    for r in reqs:
        if r.spec_token_ids:
            out[r.request_id] = list(r.spec_token_ids)
    return out


def _run(steps=STEPS):
    """Drive the fake scheduler through ``steps`` steps; returns (widths, scheduler, requests)."""
    reqs = _requests()
    sched = FakeSched(reqs)
    widths = []
    for step in range(steps):
        sched.current_step = step
        vk.pre_schedule(sched)
        so = FakeSO(NUM_SCHEDULED, _spec_tokens(reqs))
        vk.post_schedule(sched, so)
        widths.append(dict(getattr(so, vk.ATTR_SO)["w"]))
        vk.ingest(sched, so, FakeMRO(IDX_MAP, SAMPLED))
    return widths, sched, reqs


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def kad_env(monkeypatch, tmp_path):
    """KAD on, the shadow off, and a clean state for both modules."""
    monkeypatch.setenv("VLLM_D_RECALL", "1")
    monkeypatch.setenv("VLLM_D_RECALL_KAD", "1")
    monkeypatch.setenv("VLLM_D_RECALL_DIR", str(tmp_path / "kad"))
    monkeypatch.setenv("VLLM_D_RECALL_KAD_LOG_S", "0")
    monkeypatch.delenv("VLLM_D_RECALL_SHADOW", raising=False)
    monkeypatch.setenv("VLLM_D_RECALL_SHADOW_DIR", str(tmp_path / "shadow"))
    shadow._reset_for_tests()
    vk.reset()
    yield tmp_path
    shadow._reset_for_tests()
    vk.reset()


def _shadow_on(monkeypatch, tmp_path):
    """Switch the shadow on and point it at a fresh directory."""
    monkeypatch.setenv("VLLM_D_RECALL_SHADOW", "1")
    monkeypatch.setenv("VLLM_D_RECALL_SHADOW_DIR", str(tmp_path / "shadow"))
    shadow._reset_for_tests()
    assert shadow.enabled() is True


def _read(tmp_path):
    """Read the shadow records back; returns (steps, reqs, accs)."""
    recs = list(shadow.load(str(tmp_path / "shadow")))
    steps = [r for r in recs if r["k"] == shadow.KIND_STEP]
    reqs = [r for r in recs if r["k"] == shadow.KIND_REQ]
    accs = [r for r in recs if r["k"] == shadow.KIND_ACC]
    return steps, reqs, accs


# --------------------------------------------------------------------------- tests
def test_shadow_off_writes_nothing_and_keeps_the_widths(kad_env, tmp_path):
    """The shadow off: no file at all and the KAD widths are the ones of a build without it."""
    widths, _sched, _reqs = _run()
    out = tmp_path / "shadow"
    assert not out.exists() or list(out.iterdir()) == []
    # the ladder: the first step is the base width, a fully accepted request moves up
    assert widths[0] == {"d1": 5, "d2": 5, "d3": 5, "d4": 5, "d5": 5, "p1": 5, "s1": 12}
    assert widths[1] == {"d1": 7, "d2": 5, "d3": 7, "d4": 5, "d5": 5, "p1": 5, "s1": 12}
    snap = vk.snapshot()
    assert snap["steps"] == STEPS and snap["v_steps"] == 2 * 4
    # the first step verifies 12 drafts per request, the second one only the shortened 5
    assert snap["sum_v"] == 4 * 12 + 4 * 5
    assert snap["sum_a"] == (12 + 2 + 12 + 4) + (5 + 2 + 5 + 4)
    assert snap["errors"] == 0
    # the width histogram counts every scheduled request: 7 per step
    assert snap["w_hist"] == {"5": 10, "7": 2, "12": 2}


def test_shadow_on_records_the_step_the_requests_and_the_results(kad_env, monkeypatch, tmp_path):
    """The shadow on: the three kinds of records carry the values of the fake scheduler."""
    _shadow_on(monkeypatch, tmp_path)
    _run()
    shadow.get_recorder("sched", 0).flush()
    steps, reqs, accs = _read(tmp_path)
    assert len(steps) == STEPS and len(reqs) == 2 * 5 and len(accs) == 2 * 4
    for rec in steps + reqs + accs:
        assert shadow.validate(rec["k"], rec) == [], rec

    # the step records: 5 decode requests with drafts, 1 prefill request with 97 tokens
    assert steps[0]["step"] == 0 and steps[1]["step"] == 1
    assert steps[0]["n_dec"] == 5 and steps[0]["n_pre"] == 1
    assert steps[0]["tok_pre"] == 97
    assert steps[0]["bt"] == 5 * 13
    assert steps[0]["dt_ms"] == 0.0
    assert steps[1]["dt_ms"] >= 0.0
    # the drafts were shortened to the width of the previous step: 5 + 1 tokens per request
    assert steps[1]["bt"] == 4 * 6 + 13
    assert steps[1]["n_dec"] == 5 and steps[1]["n_pre"] == 1

    # the request records: the position, the sampling parameters and the sent width
    by_rid = {r["rid"]: r for r in reqs if r["step"] == 0}
    assert sorted(by_rid) == ["d1", "d2", "d3", "d5", "s1"]
    assert by_rid["d1"]["pos"] == 10 and by_rid["d1"]["w"] == 12
    assert by_rid["d1"]["temp"] == 0.7 and by_rid["d1"]["top_p"] == 0.9
    assert by_rid["d1"]["top_k"] == -1 and by_rid["d1"]["st"] is False
    assert by_rid["d2"]["pos"] == 20 and by_rid["d2"]["top_k"] == -1
    assert by_rid["d3"]["pos"] == 30 and by_rid["d3"]["top_k"] == 20
    assert by_rid["d3"]["temp"] == 0.0 and by_rid["d3"]["top_p"] == 0.5
    assert by_rid["d5"]["pos"] == 50 and by_rid["d5"]["w"] == 12
    assert by_rid["d5"]["temp"] == -1 and by_rid["d5"]["top_p"] == -1
    assert by_rid["d5"]["top_k"] == -1
    # the structured request is never shortened: it always sends the full width
    assert by_rid["s1"]["pos"] == 60 and by_rid["s1"]["w"] == 12
    assert by_rid["s1"]["st"] is True and by_rid["s1"]["top_k"] == 5
    second = {r["rid"]: r for r in reqs if r["step"] == 1}
    assert second["d1"]["w"] == 5 and second["d5"]["w"] == 5
    assert second["s1"]["w"] == 12

    # the result records: the step of the scheduler output, the position and the counts
    acc0 = {r["rid"]: r for r in accs if r["step"] == 0}
    assert sorted(acc0) == ["d1", "d2", "d3", "d5"]
    assert acc0["d1"]["pos"] == 10 and acc0["d1"]["v"] == 12 and acc0["d1"]["a"] == 12
    assert acc0["d2"]["pos"] == 20 and acc0["d2"]["v"] == 12 and acc0["d2"]["a"] == 2
    assert acc0["d3"]["pos"] == 30 and acc0["d3"]["v"] == 12 and acc0["d3"]["a"] == 12
    assert acc0["d5"]["pos"] == 50 and acc0["d5"]["v"] == 12 and acc0["d5"]["a"] == 4
    acc1 = {r["rid"]: r for r in accs if r["step"] == 1}
    assert acc1["d1"]["v"] == 5 and acc1["d1"]["a"] == 5
    assert acc1["d2"]["v"] == 5 and acc1["d2"]["a"] == 2


def test_shadow_does_not_change_the_width_decisions(kad_env, monkeypatch, tmp_path):
    """The shadow on and off: the widths of every step and the snapshot are the same."""
    off_widths, _s, _r = _run()
    off_snapshot = vk.snapshot()
    vk.reset()  # the same scenario from the same initial state
    _shadow_on(monkeypatch, tmp_path)
    on_widths, _s, _r = _run()
    on_snapshot = vk.snapshot()
    assert off_widths == on_widths
    assert off_snapshot == on_snapshot


def test_a_broken_recorder_does_not_break_kad(kad_env, monkeypatch, tmp_path, capsys):
    """A recorder that raises: KAD keeps working and the shadow is switched off for good."""
    _shadow_on(monkeypatch, tmp_path)

    class Boom:
        def record(self, kind, rec):
            raise RuntimeError("boom")

    monkeypatch.setattr(shadow, "get_recorder", lambda role, rank=0: Boom())
    widths, _sched, _reqs = _run()
    assert widths[0] == {"d1": 5, "d2": 5, "d3": 5, "d4": 5, "d5": 5, "p1": 5, "s1": 12}
    assert widths[1] == {"d1": 7, "d2": 5, "d3": 7, "d4": 5, "d5": 5, "p1": 5, "s1": 12}
    assert vk.snapshot()["errors"] == 0
    assert vk._state["shadow_ok"] is False
    out = capsys.readouterr().out
    assert out.count("shadow instrumentation failed") == 1
    # the shadow stays off: a second run does not log again
    _run()
    assert capsys.readouterr().out.count("shadow instrumentation failed") == 0


def test_the_step_number_reaches_the_scheduler_output(kad_env, monkeypatch, tmp_path):
    """post_schedule publishes the step number; an ingest without it records the step -1."""
    _shadow_on(monkeypatch, tmp_path)
    reqs = _requests()
    sched = FakeSched(reqs)
    so = FakeSO(NUM_SCHEDULED, _spec_tokens(reqs))
    vk.post_schedule(sched, so)
    assert getattr(so, vk.ATTR_SHADOW_STEP) == 0
    so2 = FakeSO(NUM_SCHEDULED, _spec_tokens(reqs))
    vk.ingest(sched, so2, FakeMRO(IDX_MAP, SAMPLED))
    shadow.get_recorder("sched", 0).flush()
    _steps, _reqs_rec, accs = _read(tmp_path)
    assert {r["step"] for r in accs} == {-1}
