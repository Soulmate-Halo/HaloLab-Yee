"""TensorRT-LLM KAD adapter on fake PyExecutor / Drafter modules (no torch, no tensorrt_llm needed)."""
import importlib
import importlib.util
import json
import os
import sys
import types

import pytest

from yee import kad as core

EXEC = "tensorrt_llm._torch.pyexecutor.py_executor"
DRAFT = "tensorrt_llm._torch.speculative.drafter"
NAMES = ("tensorrt_llm", "tensorrt_llm._torch", "tensorrt_llm._torch.pyexecutor", EXEC,
         "tensorrt_llm._torch.speculative", DRAFT)


class Req:
    def __init__(self, rid, guided=False, gen=True):
        self.py_request_id = rid
        self.py_guided_decoding_params = {"json_schema": "{}"} if guided else None
        self.py_num_accepted_draft_tokens = None
        self._gen = gen

    def is_generation_in_progress_state(self):
        return self._gen


class Batch:
    def __init__(self, reqs):
        self.generation_requests = list(reqs)
        self.context_requests = []


class SampleState:
    def __init__(self, scheduled_requests):
        self.scheduled_requests = scheduled_requests


class Bus:
    def __init__(self):
        self.value = None


class Dist:
    """tp_broadcast between two in-process 'ranks': rank 0 publishes, rank 1 reads."""

    def __init__(self, rank, bus):
        self.tp_size = 2
        self.tp_rank = rank
        self.bus = bus

    def tp_broadcast(self, obj, root=0):
        if self.tp_rank == root:
            self.bus.value = obj
            return obj
        return self.bus.value


def _classes():
    """Fresh classes for every test, so a patch can never leak into the next one."""

    class Drafter:
        def __init__(self, schedule=None, static=12):
            self.draft_len_schedule = schedule
            self._static_max_total_draft_tokens = static
            self.max_total_draft_tokens = static
            self.calls = 0

        def get_draft_len_for_batch_size(self, batch_size):
            self.calls += 1
            value = self._static_max_total_draft_tokens
            for threshold in sorted(self.draft_len_schedule):
                if batch_size >= threshold:
                    value = self.draft_len_schedule[threshold]
            return value

        def update_max_total_draft_tokens(self, n):
            self.max_total_draft_tokens = n

    class PyExecutor:
        def __init__(self, drafter, requests, dist=None):
            self.drafter = drafter
            self.active_requests = list(requests)
            self.max_total_draft_tokens = drafter._static_max_total_draft_tokens
            self.accept = {}
            if dist is not None:
                self.dist = dist

        def _prepare_and_schedule_batch(self):
            # same order as TensorRT-LLM 1.2.1 py_executor.py: schedule length first, then scheduling
            if self.drafter is not None and self.drafter.draft_len_schedule is not None:
                self.max_total_draft_tokens = self.drafter.get_draft_len_for_batch_size(len(self.active_requests))
                self.drafter.update_max_total_draft_tokens(self.max_total_draft_tokens)
            scheduled_batch = Batch([r for r in self.active_requests if r.is_generation_in_progress_state()])
            return scheduled_batch, None

        def _update_requests(self, sample_state, resource_manager=None):
            for r in sample_state.scheduled_requests.generation_requests:
                r.py_num_accepted_draft_tokens = self.accept.get(r.py_request_id, 0)

    return Drafter, PyExecutor


@pytest.fixture
def trt(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("YEE_KAD"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YEE_KAD_DIR", str(tmp_path))
    monkeypatch.setenv("YEE_KAD_LOG_S", "3600")
    drafter_cls, executor_cls = _classes()
    mods = {name: types.ModuleType(name) for name in NAMES}
    mods["tensorrt_llm"].__version__ = "1.2.1"
    mods[DRAFT].Drafter = drafter_cls
    mods[EXEC].PyExecutor = executor_cls
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    tk = importlib.import_module("yee.integrations.trtllm_kad")
    tk.uninstall()
    tk._KAD = None
    tk._SEEN.clear()
    yield types.SimpleNamespace(tk=tk, Drafter=drafter_cls, PyExecutor=executor_cls, dir=tmp_path)
    tk.uninstall()
    tk._KAD = None
    tk._SEEN.clear()


def _methods(trt):
    return ((trt.Drafter, "get_draft_len_for_batch_size"), (trt.PyExecutor, "_prepare_and_schedule_batch"),
            (trt.PyExecutor, "_update_requests"))


def _on(monkeypatch, trt):
    monkeypatch.setenv("YEE_KAD", "1")
    ok, why = trt.tk.install()
    assert ok, why


def _round(ex, accept=None):
    """One iteration: schedule, then apply the samples (every draft accepted unless ``accept`` says otherwise)."""
    batch, _ = ex._prepare_and_schedule_batch()
    n = ex.max_total_draft_tokens
    ex.accept = {r.py_request_id: n for r in batch.generation_requests}
    if accept:
        ex.accept.update(accept)
    ex._update_requests(SampleState(batch))
    return n, batch


# 1 ------------------------------------------------------------------------------------------------
def test_off_by_default_and_skipped_without_engine(trt, monkeypatch, capsys):
    before = [cls.__dict__[n] for cls, n in _methods(trt)]
    ok, why = trt.tk.install()
    assert ok is False and why == "YEE_KAD is off"
    assert all(cls.__dict__[n] is b for (cls, n), b in zip(_methods(trt), before))

    monkeypatch.setenv("YEE_KAD", "1")
    sys.modules[DRAFT].Drafter = type("Drafter", (), {"get_draft_len_for_batch_size": lambda self, b: 12})
    ok, why = trt.tk.install()
    assert ok is False and "Drafter lacks" in why
    assert all(cls.__dict__[n] is b for (cls, n), b in zip(_methods(trt), before))

    for name in NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    if importlib.util.find_spec("tensorrt_llm") is None:
        ok, why = trt.tk.install()
        assert ok is False and "not importable" in why
        assert "[yee-kad] tensorrt-llm not installed" in capsys.readouterr().out


# 2 ------------------------------------------------------------------------------------------------
def test_single_request_ladder(trt, monkeypatch, capsys):
    _on(monkeypatch, trt)
    assert "[yee-kad] tensorrt-llm installed on TensorRT-LLM 1.2.1" in capsys.readouterr().out
    d = trt.Drafter({1: 12})
    ex = trt.PyExecutor(d, [Req("a")])
    lens = []
    for _ in range(6):
        n, _ = _round(ex)
        assert d.max_total_draft_tokens == n
        lens.append(n)
    assert lens == [5, 7, 9, 11, 12, 12]
    snap = trt.tk.kad().snapshot()
    assert snap["errors"] == 0 and snap["v_steps"] == 6


# 3 ------------------------------------------------------------------------------------------------
def test_batch_runs_max_width_capped_by_schedule(trt, monkeypatch):
    _on(monkeypatch, trt)
    d = trt.Drafter({1: 12})
    a = Req("a")
    ex = trt.PyExecutor(d, [a])
    assert [_round(ex)[0] for _ in range(2)] == [5, 7]  # a is now at 9
    ex.active_requests = [a, Req("b")]
    n, _ = _round(ex)
    assert n == 9  # K_b = max(9, 5)
    k = trt.tk.kad()
    assert k.state("b")["cur"] == 7  # b was verified with v = 9 and accepted everything
    snap = k.snapshot()
    assert snap["mixed_steps"] >= 1 and snap["b_hist"].get("9") == 1

    d.draft_len_schedule = {1: 12, 4: 6}
    ex.active_requests = [a, Req("b"), Req("c"), Req("e")]
    n, _ = _round(ex)
    assert n == 6  # a wants 11, the schedule allows 6 for this batch size

    d.draft_len_schedule = {1: 0}
    n, batch = _round(ex)
    assert n == 0 and d.max_total_draft_tokens == 0


# 4 ------------------------------------------------------------------------------------------------
def test_rejection_falls_back_to_accepted_plus_one(trt, monkeypatch):
    _on(monkeypatch, trt)
    ex = trt.PyExecutor(trt.Drafter({1: 12}), [Req("a")])
    assert [_round(ex)[0] for _ in range(5)] == [5, 7, 9, 11, 12]
    n, _ = _round(ex, accept={"a": 3})
    assert n == 12
    n, _ = _round(ex)
    assert n == 5  # max(kmin 5, min(12, 3 + 1))


# 5 ------------------------------------------------------------------------------------------------
def test_guided_request_keeps_schedule_value(trt, monkeypatch):
    _on(monkeypatch, trt)
    ex = trt.PyExecutor(trt.Drafter({1: 12}), [Req("g", guided=True)])
    n, _ = _round(ex)
    assert n == 12
    ex.active_requests = [Req("a")]
    n, _ = _round(ex)
    assert n == 5


# 6 ------------------------------------------------------------------------------------------------
def test_overlap_observes_with_the_length_of_its_own_batch(trt, monkeypatch):
    _on(monkeypatch, trt)
    a = Req("a")
    ex = trt.PyExecutor(trt.Drafter({1: 12}), [a])
    assert [_round(ex)[0] for _ in range(2)] == [5, 7]  # a is now at 9
    batch_n, _ = ex._prepare_and_schedule_batch()
    assert ex.max_total_draft_tokens == 9
    ex.active_requests = [a, Req("g", guided=True)]
    batch_n1, _ = ex._prepare_and_schedule_batch()
    assert ex.max_total_draft_tokens == 12  # batch N+1 has another length
    ex.accept = {"a": 9}
    ex._update_requests(SampleState(batch_n))  # samples of batch N arrive after N+1 was scheduled
    assert trt.tk.kad().state("a")["cur"] == 11  # observed with v = 9 (all accepted), not 12


# 7 ------------------------------------------------------------------------------------------------
def test_no_schedule_means_no_kad(trt, monkeypatch):
    _on(monkeypatch, trt)
    d = trt.Drafter(None)
    ex = trt.PyExecutor(d, [Req("a")])
    for _ in range(3):
        n, batch = _round(ex)
        assert n == 12
        assert not hasattr(batch, "_yee_kad_v")
    assert d.calls == 0
    k = trt.tk.kad()
    assert k.state("a") is None and k.snapshot()["v_steps"] == 0


# 8 ------------------------------------------------------------------------------------------------
def test_width_error_keeps_schedule_value(trt, monkeypatch, capsys):
    _on(monkeypatch, trt)
    ex = trt.PyExecutor(trt.Drafter({1: 12}), [Req("a")])
    k = trt.tk.kad(ex)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(k, "widths", boom)
    n, _ = _round(ex)
    assert n == 12
    assert k.snapshot()["errors"] == 1
    cap = capsys.readouterr()
    assert "boom" in cap.out + cap.err


# 9 ------------------------------------------------------------------------------------------------
def test_tp_ranks_agree_and_switch_policy_together(trt, monkeypatch):
    _on(monkeypatch, trt)
    tk = trt.tk
    bus = Bus()
    ex0 = trt.PyExecutor(trt.Drafter({1: 12}), [Req("a"), Req("b")], dist=Dist(0, bus))
    ex1 = trt.PyExecutor(trt.Drafter({1: 12}), [Req("a"), Req("b")], dist=Dist(1, bus))
    sync0, lead0 = tk._tp_sync(ex0)
    sync1, lead1 = tk._tp_sync(ex1)
    assert lead0 is True and lead1 is False
    k0 = core.Kad(tk.BASE, core.KMAX, enabled=True, leader=lead0, sync=sync0, reload_every=4)
    k1 = core.Kad(tk.BASE, core.KMAX, enabled=True, leader=lead1, sync=sync1, reload_every=4)
    policy = trt.dir / "kad_policy.json"
    lens0, lens1 = [], []
    for step in range(1, 9):
        if step == 3:  # written after step 2; both ranks pick it up at step 4 (step % 4 == 0)
            policy.write_text(json.dumps({"mode": "fixed", "fixed": 7}), encoding="utf-8")
        for k, ex, lens in ((k0, ex0, lens0), (k1, ex1, lens1)):  # rank 0 runs each step first
            tk._KAD = k
            n, _ = _round(ex, accept={"b": 0})  # a accepts everything, b nothing
            lens.append(n)
    assert lens0 == lens1 == [5, 7, 9, 7, 7, 7, 7, 7]
    assert k0.policy()["mode"] == k1.policy()["mode"] == "fixed"
    sha0 = k0.snapshot().get("policy_sha256")
    assert sha0 and sha0 == k1.snapshot().get("policy_sha256")


# 10 -----------------------------------------------------------------------------------------------
def test_install_twice_and_uninstall_restores(trt, monkeypatch):
    before = [cls.__dict__[n] for cls, n in _methods(trt)]
    _on(monkeypatch, trt)
    assert all(cls.__dict__[n] is not b for (cls, n), b in zip(_methods(trt), before))
    assert all(getattr(cls.__dict__[n], trt.tk.MARK, False) for cls, n in _methods(trt))
    assert trt.tk.install() == (True, "already installed")
    assert trt.tk.status()["installed"] is True
    assert trt.tk.uninstall() is True
    assert all(cls.__dict__[n] is b for (cls, n), b in zip(_methods(trt), before))
    assert trt.tk.uninstall() is False
