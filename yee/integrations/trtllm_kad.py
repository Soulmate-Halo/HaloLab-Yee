"""KAD for TensorRT-LLM (PyTorch backend): per-request ladders, one draft length per batch, max 12.

TensorRT-LLM runs one draft length per batch: CUDA graphs are keyed by (batch size, draft length) and
``CUDAGraphRunner.get_graph_key`` asserts that every request of a batch has the same draft length. So KAD runs
K_b = max k_i over the batch's generation requests (5, 7, 9, 11, 12 while everything is accepted,
max(5, accepted + 1) after a rejection).

The engine already changes the draft length per iteration for ``draft_len_schedule`` (batch size -> length):
``PyExecutor._prepare_and_schedule_batch`` calls ``drafter.get_draft_len_for_batch_size(len(active_requests))``
and ``drafter.update_max_total_draft_tokens(...)`` before scheduling. KAD rides on that path: the length of an
iteration becomes min(schedule value for the batch size, K_b), so the draft loop and the target's verify length
really shrink. After sampling (``PyExecutor._update_requests``) every generation request is observed with
v = the length its batch ran (tagged on the scheduled batch, so the overlap loop is handled) and
a = ``py_num_accepted_draft_tokens``. Requests with guided decoding keep the schedule value.

Needs ``draft_len_schedule`` in the speculative config, e.g. ``{1: 12}`` with ``max_draft_len: 12``; TensorRT-LLM
refuses a schedule for the static / chain draft loop (two-model EAGLE3 with the TRTLLM attention backend and
greedy drafts) and KAD is then not active. TP ranks schedule the same requests and sample the same tokens, so
their lengths agree; the policy is read by TP rank 0 and broadcast with ``dist.tp_broadcast``.
Switch: YEE_KAD=1, then ``from yee.integrations import trtllm_kad; trtllm_kad.install()`` in every worker
(the Yee autoload does it when YEE_KAD=1).
"""
from __future__ import annotations

import collections
import importlib
import os
import threading

from .. import kad as _core

BASE = "tensorrt-llm"
MARK = "__yee_kad__"
EXECUTOR_MOD = "tensorrt_llm._torch.pyexecutor.py_executor"
DRAFTER_MOD = "tensorrt_llm._torch.speculative.drafter"
VERIFIED = ("1.2.1",)
STALE_STEPS = 512

_KAD = None
_LOCK = threading.RLock()
_ORIG = []
_TLS = threading.local()
_SEEN = {}


def _tp_sync(executor):
    dist = getattr(executor, "dist", None)
    size = int(getattr(dist, "tp_size", 1) or 1) if dist is not None else 1
    if dist is None or size <= 1 or not hasattr(dist, "tp_broadcast"):
        return None, True
    leader = int(getattr(dist, "tp_rank", 0) or 0) == 0

    def sync(data):
        return dist.tp_broadcast(data if leader else None, root=0)

    return sync, leader


def kad(executor=None):
    global _KAD
    with _LOCK:
        if _KAD is None:
            sync, leader = _tp_sync(executor) if executor is not None else (None, True)
            _KAD = _core.Kad(BASE, _core.KMAX, enabled=True, leader=leader, sync=sync)
        return _KAD


def _err(what, e):
    k = _KAD
    if k is not None:
        k.error(what, e)


def _guided(req) -> bool:
    for name in ("py_guided_decoding_params", "guided_decoding_params", "guided_decoding"):
        if getattr(req, name, None) is not None:
            return True
    return False


def _rid(req):
    rid = getattr(req, "py_request_id", None)
    return rid if rid is not None else getattr(req, "request_id", id(req))


def _generation(reqs):
    out = []
    for r in reqs:
        fn = getattr(r, "is_generation_in_progress_state", None)
        try:
            gen = fn() if callable(fn) else bool(fn) if fn is not None else True
        except Exception:  # noqa: BLE001
            gen = True
        if gen:
            out.append(r)
    return out


def batch_length(drafter, reqs, base_len: int) -> int:
    """The draft length of one iteration: min(schedule value, K_b); the schedule value on error."""
    if base_len <= 0 or not reqs:
        return base_len
    k = kad()
    static = int(getattr(drafter, "_static_max_total_draft_tokens", 0) or 0)
    if static and k.kmax_run != static:
        k.set_kmax_run(static)
    step = k.begin_step()
    rids = [_rid(r) for r in reqs]
    for rid in rids:
        _SEEN[rid] = step
    if step % STALE_STEPS == 0:
        for rid in [r for r, s in _SEEN.items() if step - s > STALE_STEPS]:
            _SEEN.pop(rid, None)
            k.release(rid)
    if any(_guided(r) for r in reqs):
        k.note_batch([base_len] * len(rids), k_used=base_len)
        return base_len
    widths = k.widths(rids)
    kb = max(1, min(base_len, max(widths.values())))
    k.note_batch(widths, k_used=kb)
    return kb


def _wrap_get_len(orig):
    def get_draft_len_for_batch_size(self, batch_size):
        base_len = orig(self, batch_size)
        reqs = getattr(_TLS, "active", None)
        if reqs is None:
            return base_len
        try:
            n = batch_length(self, reqs, int(base_len))
        except Exception as e:  # noqa: BLE001  keep the schedule value
            _err("width", e)
            n = base_len
        _TLS.length = n
        return n

    get_draft_len_for_batch_size.__wrapped__ = orig
    return get_draft_len_for_batch_size


def _wrap_prepare(orig):
    def _prepare_and_schedule_batch(self, *args, **kwargs):
        if _KAD is None:
            kad(self)
        _TLS.active = _generation(list(getattr(self, "active_requests", None) or []))
        _TLS.length = None
        try:
            out = orig(self, *args, **kwargs)
        finally:
            length = _TLS.length
            _TLS.active = None
            _TLS.length = None
        if length is not None:
            batch = out[0] if isinstance(out, tuple) and out else out
            try:
                batch._yee_kad_v = int(length)  # read back when this batch's samples are applied
            except Exception:  # noqa: BLE001
                pass
        return out

    _prepare_and_schedule_batch.__wrapped__ = orig
    return _prepare_and_schedule_batch


def _wrap_update(orig):
    def _update_requests(self, sample_state, *args, **kwargs):
        out = orig(self, sample_state, *args, **kwargs)
        try:
            sched = getattr(sample_state, "scheduled_requests", None)
            v = getattr(sched, "_yee_kad_v", None)
            if v is not None and _KAD is not None:
                k = _KAD
                for req in list(getattr(sched, "generation_requests", None) or []):
                    a = getattr(req, "py_num_accepted_draft_tokens", None)
                    if a is not None:
                        k.observe(_rid(req), int(v), int(a))
        except Exception as e:  # noqa: BLE001
            _err("observe", e)
        return out

    _update_requests.__wrapped__ = orig
    return _update_requests


def _skip(reason):
    print("%s %s not installed: %s" % (_core.TAG, BASE, reason), flush=True)
    return False, reason


def _patch(cls, name, wrapper):
    cur = cls.__dict__[name]
    if getattr(cur, MARK, False):
        return
    new = wrapper(cur)
    setattr(new, MARK, True)
    setattr(cls, name, new)
    _ORIG.append((cls, name, cur))


def install(enabled=None):
    """Patch PyExecutor / Drafter. Returns (installed, reason)."""
    on = _core.env_enabled() if enabled is None else bool(enabled)
    if not on:
        return False, "YEE_KAD is off"
    try:
        emod = importlib.import_module(EXECUTOR_MOD)
        dmod = importlib.import_module(DRAFTER_MOD)
    except Exception as e:  # noqa: BLE001
        return _skip("TensorRT-LLM PyTorch backend not importable: %r" % (e,))
    executor = getattr(emod, "PyExecutor", None)
    drafter = getattr(dmod, "Drafter", None)
    need_e = ("_prepare_and_schedule_batch", "_update_requests")
    need_d = ("get_draft_len_for_batch_size", "update_max_total_draft_tokens")
    if executor is None or any(n not in vars(executor) for n in need_e):
        return _skip("PyExecutor lacks %s" % (", ".join(need_e),))
    if drafter is None or any(n not in vars(drafter) for n in need_d):
        return _skip("Drafter lacks %s" % (", ".join(need_d),))
    with _LOCK:
        if getattr(vars(executor)["_prepare_and_schedule_batch"], MARK, False):
            return True, "already installed"
        _patch(drafter, "get_draft_len_for_batch_size", _wrap_get_len)
        _patch(executor, "_prepare_and_schedule_batch", _wrap_prepare)
        _patch(executor, "_update_requests", _wrap_update)
    try:
        version = getattr(importlib.import_module("tensorrt_llm"), "__version__", "?")
    except Exception:  # noqa: BLE001
        version = "?"
    print("%s %s installed on TensorRT-LLM %s%s (pid %d); active when speculative_config.draft_len_schedule is set"
          % (_core.TAG, BASE, version, "" if version in VERIFIED else " (not in verified %s)" % (VERIFIED,),
             os.getpid()), flush=True)
    return True, "patched PyExecutor and Drafter"


def uninstall() -> bool:
    with _LOCK:
        done = bool(_ORIG)
        while _ORIG:
            cls, name, orig = _ORIG.pop()
            setattr(cls, name, orig)
        return done


def status() -> dict:
    out = {"base": BASE, "installed": bool(_ORIG), "verified": list(VERIFIED)}
    if _KAD is not None:
        out["kad"] = _KAD.snapshot()
    return out


__all__ = ["BASE", "VERIFIED", "batch_length", "install", "uninstall", "status", "kad"]
_ = collections
