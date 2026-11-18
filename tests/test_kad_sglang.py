"""KAD on the Mia SGLang build (r641-SG-T): the adaptive-runtime adapter, on fake SGLang modules.

Neither torch nor sglang is needed: the Mia modules the adapter touches (the adaptive controller,
the EAGLE worker and the batch result processor) are replaced by stand-ins with the same module
names, class names, method names and call order, so the adapter's own logic (width choice, batch
width, verification feedback, overlap tagging, multi-rank policy sync) is what is under test.
"""
import importlib
import importlib.machinery
import json
import os
import sys
import types

import pytest

from yee import kad
from yee.integrations import sglang_kad

STEPS = (5, 7, 9, 11, 12)
SGLANG_NAMES = ("sglang", "sglang.srt", "sglang.speculative", "sglang.managers")


# --------------------------------------------------------------------------------- stand-ins
class FakeReq:
    def __init__(self, rid, grammar=None):
        self.rid = rid
        self.grammar = grammar


class FakeForwardMode:
    def __init__(self, decode=True):
        self.decode = decode

    def is_decode(self):
        return self.decode

    def is_extend(self):
        return not self.decode


class FakeBatch:
    def __init__(self, reqs, accepts=None, decode=True):
        self.reqs = list(reqs)
        self.accepts = list(accepts or [])
        self.forward_mode = FakeForwardMode(decode)


class FakeResult:
    def __init__(self, accepts=None):
        self.num_correct_drafts_per_req_cpu = list(accepts or [])


class FakeParams:
    """The Mia AdaptiveSpeculativeParams stand-in: the candidate steps and the native routing."""

    def __init__(self, candidate_steps):
        self.candidate_steps = list(candidate_steps)
        self.native_steps = min(self.candidate_steps)

    def get_steps_for_batch(self, batch_size):
        return self.native_steps


class AdaptiveController:
    """The Mia AdaptiveController stand-in: one runtime state per candidate step."""

    def __init__(self, worker, candidate_steps):
        self.worker = worker
        self.params = FakeParams(candidate_steps)
        self._states = {int(s): object() for s in candidate_steps}
        self.activations = []
        self.native_calls = []
        self.native_verifies = []

    def _activate(self, speculative_num_steps):
        steps = int(speculative_num_steps)
        self.activations.append(steps)
        self.worker.speculative_num_steps = steps

    def activate_step_by_batch(self, batch_size):
        self.native_calls.append(batch_size)

    def on_verify_complete(self, num_correct_drafts_per_req, batch_size=0):
        self.native_verifies.append((list(num_correct_drafts_per_req), batch_size))


class EAGLEWorkerV2:
    """The Mia EAGLEWorkerV2 stand-in: decode calls activate_step_by_batch, verify feeds the controller."""

    def __init__(self, candidate_steps=STEPS):
        self.speculative_num_steps = max(candidate_steps)
        self.adaptive_controller = AdaptiveController(self, candidate_steps)
        self.forward_calls = []
        self.native_activates = []
        self.native_verifies = []

    def forward_batch_generation(self, batch, *args, **kwargs):
        self.forward_calls.append(batch)
        if batch.forward_mode.is_decode():
            self.activate_step_by_batch(len(batch.reqs))
        return FakeResult(list(getattr(batch, "accepts", []) or []))

    def activate_step_by_batch(self, batch_size):
        self.native_activates.append(batch_size)
        self.adaptive_controller.activate_step_by_batch(batch_size)

    def on_verify_complete_cpu(self, num_correct_drafts_per_req, batch_size=0):
        self.native_verifies.append((list(num_correct_drafts_per_req), batch_size))
        self.adaptive_controller.on_verify_complete(num_correct_drafts_per_req, batch_size=batch_size)


class BatchResultProcessor:
    """The Mia BatchResultProcessor stand-in: resolves the tokens, then feeds the worker."""

    def __init__(self, model_worker):
        self.model_worker = model_worker
        self.resolve_calls = []

    def _resolve_spec_v2_tokens(self, result, batch, *args, **kwargs):
        self.resolve_calls.append((result, batch))
        accepts = list(getattr(result, "num_correct_drafts_per_req_cpu", None) or [])
        self.model_worker.on_verify_complete_cpu(accepts, batch_size=len(batch.reqs))
        return []


# --------------------------------------------------------------------------------- fixtures
def _new_module(name, package=False):
    mod = types.ModuleType(name)
    mod.__file__ = "<yee-fake %s>" % name
    if package:
        mod.__path__ = []
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=package)
    return mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    for key in [k for k in os.environ if k.startswith("YEE_KAD")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("YEE_KAD_DIR", str(tmp_path))
    monkeypatch.setenv("YEE_KAD_LOG_S", "3600")
    monkeypatch.setattr(sglang_kad, "_KAD", None)
    monkeypatch.setattr(sglang_kad, "_ORIG", [])
    monkeypatch.setattr(sglang_kad, "_SEEN", {})
    # one process, no TP group: never touch torch
    monkeypatch.setattr(sglang_kad, "_tp_sync", lambda: (None, True))
    sglang_kad.uninstall()
    yield tmp_path
    sglang_kad.uninstall()
    sglang_kad._KAD = None
    sglang_kad._ORIG = []
    sglang_kad._SEEN = {}


@pytest.fixture
def fake_sglang(env, monkeypatch):
    saved = {k: v for k, v in sys.modules.items() if k == "sglang" or k.startswith("sglang.")}
    for k in saved:
        del sys.modules[k]

    sglang = _new_module("sglang", True)
    srt = _new_module("sglang.srt", True)
    spec_pkg = _new_module("sglang.srt.speculative", True)
    mgr = _new_module("sglang.srt.managers", True)
    sched = _new_module("sglang.srt.managers.scheduler_components", True)
    ars = _new_module("sglang.srt.speculative.adaptive_runtime_state")
    ewg = _new_module("sglang.srt.speculative.eagle_worker_v2")
    brp = _new_module("sglang.srt.managers.scheduler_components.batch_result_processor")

    sglang.srt = srt
    srt.speculative = spec_pkg
    srt.managers = mgr
    mgr.scheduler_components = sched
    ars.AdaptiveController = AdaptiveController
    ewg.EAGLEWorkerV2 = EAGLEWorkerV2
    brp.BatchResultProcessor = BatchResultProcessor

    for name, mod in (("sglang", sglang), ("sglang.srt", srt), ("sglang.srt.speculative", spec_pkg),
                      ("sglang.srt.managers", mgr),
                      ("sglang.srt.managers.scheduler_components", sched),
                      ("sglang.srt.speculative.adaptive_runtime_state", ars),
                      ("sglang.srt.speculative.eagle_worker_v2", ewg),
                      ("sglang.srt.managers.scheduler_components.batch_result_processor", brp)):
        sys.modules[name] = mod

    yield {"ars": ars, "ewg": ewg, "brp": brp}

    for name in [k for k in list(sys.modules) if k == "sglang" or k.startswith("sglang.")]:
        del sys.modules[name]
    sys.modules.update(saved)


def _install():
    ok, why = sglang_kad.install()
    assert ok, why
    return why


def _worker(steps=STEPS):
    worker = EAGLEWorkerV2(steps)
    return worker, BatchResultProcessor(worker)


def _run(worker, reqs, accepts=None, decode=True):
    """One batch: forward, then the result the batch produced."""
    batch = FakeBatch(reqs, accepts, decode)
    result = worker.forward_batch_generation(batch)
    return batch, result


def _accepts(result, accepts):
    result.num_correct_drafts_per_req_cpu = list(accepts)
    return result


# --------------------------------------------------------------------------------- the tests
def test_install_off_without_env(fake_sglang):
    before = (EAGLEWorkerV2.forward_batch_generation, EAGLEWorkerV2.activate_step_by_batch,
              EAGLEWorkerV2.on_verify_complete_cpu, BatchResultProcessor._resolve_spec_v2_tokens)
    assert sglang_kad.install() == (False, "YEE_KAD is off")
    after = (EAGLEWorkerV2.forward_batch_generation, EAGLEWorkerV2.activate_step_by_batch,
             EAGLEWorkerV2.on_verify_complete_cpu, BatchResultProcessor._resolve_spec_v2_tokens)
    assert after == before
    assert sglang_kad.status()["installed"] is False


def test_install_needs_the_mia_adaptive_runtime(fake_sglang, monkeypatch):
    monkeypatch.setenv("YEE_KAD", "1")
    before = EAGLEWorkerV2.forward_batch_generation
    del sys.modules["sglang.srt.speculative.adaptive_runtime_state"]
    ok, why = sglang_kad.install()
    assert ok is False and "Mia" in why
    assert EAGLEWorkerV2.forward_batch_generation is before


def test_single_request_climbs_to_12(fake_sglang, monkeypatch):
    monkeypatch.setenv("YEE_KAD", "1")
    _install()
    worker, processor = _worker()
    req = FakeReq("r1")
    seen = []
    for _ in range(6):
        batch, result = _run(worker, [req])
        seen.append(result._yee_kad_steps)
        _accepts(result, [result._yee_kad_steps])  # everything accepted
        processor._resolve_spec_v2_tokens(result, batch)
    assert seen == [5, 7, 9, 11, 12, 12]
    assert worker.adaptive_controller.activations == [5, 7, 9, 11, 12]
    assert worker.native_activates == []  # the native routing never ran
    snap = sglang_kad.kad().snapshot()
    assert snap["errors"] == 0 and snap["v_steps"] == 6
    assert snap["b_hist"] == {"5": 1, "7": 1, "9": 1, "11": 1, "12": 2}
    assert snap["mixed_steps"] == 0


def test_two_requests_use_the_widest_and_round_up(fake_sglang, monkeypatch):
    monkeypatch.setenv("YEE_KAD", "1")
    _install()
    worker, processor = _worker(steps=(5, 8, 12))
    k = sglang_kad.kad()
    k.observe("r2", 12, 12)  # r2 already climbed to 7, r1 is new (5)
    a, b = FakeReq("r1"), FakeReq("r2")
    batch, result = _run(worker, [a, b], accepts=[8, 3])
    assert result._yee_kad_steps == 8  # max(5, 7) rounded up to the captured step 8
    assert k.widths(["r1", "r2"]) == {"r1": 5, "r2": 7}
    processor._resolve_spec_v2_tokens(result, batch)
    assert k.state("r1")["cur"] == 7  # accepted everything at v=8
    assert k.state("r2")["cur"] == 5  # only 3 of 8 accepted
    assert k.snapshot()["mixed_steps"] == 1


def test_rejection_falls_back_to_5(fake_sglang, monkeypatch):
    monkeypatch.setenv("YEE_KAD", "1")
    _install()
    worker, processor = _worker()
    k = sglang_kad.kad()
    req = FakeReq("r1")
    for _ in range(4):
        batch, result = _run(worker, [req])
        _accepts(result, [result._yee_kad_steps])
        processor._resolve_spec_v2_tokens(result, batch)
    assert k.state("r1")["cur"] == 12
    batch, result = _run(worker, [req])
    assert result._yee_kad_steps == 12
    _accepts(result, [3])  # only 3 of 12 accepted
    processor._resolve_spec_v2_tokens(result, batch)
    assert k.state("r1")["cur"] == 5
    batch, result = _run(worker, [req])
    assert result._yee_kad_steps == 5


def test_grammar_batch_runs_the_widest_step(fake_sglang, monkeypatch):
    monkeypatch.setenv("YEE_KAD", "1")
    _install()
    worker, processor = _worker()
    plain, structured = FakeReq("r1"), FakeReq("r2", grammar=object())
    batch, result = _run(worker, [plain, structured], accepts=[12, 12])
    assert result._yee_kad_steps == 12
    assert worker.speculative_num_steps == 12
    processor._resolve_spec_v2_tokens(result, batch)
    assert sglang_kad.kad().snapshot()["b_hist"] == {"12": 1}


def test_overlap_uses_the_steps_of_its_own_batch(fake_sglang, monkeypatch):
    monkeypatch.setenv("YEE_KAD", "1")
    _install()
    worker, processor = _worker()
    k = sglang_kad.kad()
    # r2 was accepted once, so it is at 7 while r1 is still at 5
    batch0, result0 = _run(worker, [FakeReq("r2")])
    _accepts(result0, [result0._yee_kad_steps])
    processor._resolve_spec_v2_tokens(result0, batch0)
    r1 = FakeReq("r1")
    batch_n, result_n = _run(worker, [r1])  # batch N: only r1, width 5
    batch_n1, result_n1 = _run(worker, [r1, FakeReq("r2")])  # batch N+1: the width changed to 7
    assert result_n._yee_kad_steps == 5 and result_n1._yee_kad_steps == 7
    _accepts(result_n, [5])
    processor._resolve_spec_v2_tokens(result_n, batch_n)  # batch N is resolved only now
    assert k.state("r1")["cur"] == 7  # updated with v=5 (its own batch), not with 7
    _accepts(result_n1, [7, 7])
    processor._resolve_spec_v2_tokens(result_n1, batch_n1)
    assert k.state("r1")["cur"] == 9  # now v=7 with a full acceptance


def test_non_decode_batch_keeps_the_native_choice(fake_sglang, monkeypatch):
    monkeypatch.setenv("YEE_KAD", "1")
    _install()
    worker, processor = _worker()
    batch, result = _run(worker, [FakeReq("r1")], decode=False)
    assert worker.speculative_num_steps == 12  # nothing changed
    assert worker.adaptive_controller.activations == []
    assert getattr(result, "_yee_kad_steps", None) is None
    worker.activate_step_by_batch(1)  # outside a KAD decode batch the native path runs
    assert worker.native_activates == [1]
    assert worker.adaptive_controller.native_calls == [1]


def test_choose_steps_error_falls_back_to_the_native_width(fake_sglang, monkeypatch):
    monkeypatch.setenv("YEE_KAD", "1")
    _install()
    worker, processor = _worker()
    sglang_kad.kad()  # the controller exists, so the error is counted

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(sglang_kad, "choose_steps", boom)
    batch, result = _run(worker, [FakeReq("r1")])
    assert worker.native_activates == [1]  # the native activation ran
    assert worker.adaptive_controller.activations == []
    assert worker.speculative_num_steps == 12
    assert sglang_kad.kad().snapshot()["errors"] == 1


def test_verify_outside_the_resolver_runs_the_original(fake_sglang, monkeypatch):
    monkeypatch.setenv("YEE_KAD", "1")
    _install()
    worker, processor = _worker()
    worker.on_verify_complete_cpu([3], batch_size=1)
    assert worker.native_verifies == [([3], 1)]
    assert worker.adaptive_controller.native_verifies == [([3], 1)]
    assert sglang_kad.kad().snapshot()["v_steps"] == 0


def test_two_ranks_agree_and_switch_at_the_same_step(tmp_path, monkeypatch):
    policy = tmp_path / "pol0.json"
    policy.write_text(json.dumps(kad.DEFAULT_POLICY), encoding="utf-8")
    other = tmp_path / "pol1.json"
    other.write_text(json.dumps({"mode": "fixed", "fixed": 3}), encoding="utf-8")
    box = {}

    def sync_leader(data):
        box["data"] = data
        return data

    def sync_follower(data):
        return box.get("data")

    rank0 = kad.Kad("sglang", 12, enabled=True, policy_file=str(policy), leader=True,
                     sync=sync_leader, reload_every=4, log_s=3600)
    rank1 = kad.Kad("sglang", 12, enabled=True, policy_file=str(other), leader=False,
                     sync=sync_follower, reload_every=4, log_s=3600)
    widths0, widths1 = [], []
    for step in range(1, 9):
        if step == 5:
            policy.write_text(json.dumps({"mode": "fixed", "fixed": 7}), encoding="utf-8")
        rank0.begin_step()
        rank1.begin_step()
        w0 = rank0.widths(["r1"])
        w1 = rank1.widths(["r1"])
        widths0.append(max(w0.values()))
        widths1.append(max(w1.values()))
        assert w0 == w1
        v = max(w0.values())
        rank0.observe("r1", v, v)
        rank1.observe("r1", v, v)
    assert widths0 == widths1
    assert widths0[:7] == [5, 7, 9, 11, 12, 12, 12]
    assert widths0[7] == 7 and widths1[7] == 7  # both switched to fixed 7 at the same step
    assert rank0.snapshot()["policy_sha256"] == rank1.snapshot()["policy_sha256"]


def test_uninstall_restores_the_original_methods(fake_sglang, monkeypatch):
    monkeypatch.setenv("YEE_KAD", "1")
    before = (EAGLEWorkerV2.forward_batch_generation, EAGLEWorkerV2.activate_step_by_batch,
              EAGLEWorkerV2.on_verify_complete_cpu, BatchResultProcessor._resolve_spec_v2_tokens)
    _install()
    after = (EAGLEWorkerV2.forward_batch_generation, EAGLEWorkerV2.activate_step_by_batch,
             EAGLEWorkerV2.on_verify_complete_cpu, BatchResultProcessor._resolve_spec_v2_tokens)
    assert after != before
    assert sglang_kad.uninstall() is True
    restored = (EAGLEWorkerV2.forward_batch_generation, EAGLEWorkerV2.activate_step_by_batch,
                EAGLEWorkerV2.on_verify_complete_cpu, BatchResultProcessor._resolve_spec_v2_tokens)
    assert restored == before
    assert sglang_kad.uninstall() is False
