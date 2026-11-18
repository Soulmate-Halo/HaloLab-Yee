"""KAD for SGLang (Mia build): per-request adaptive draft count, max 12, on top of its adaptive runtime.

The Mia SGLang (0.0.0.dev0) has an adaptive speculative runtime for EAGLE / EAGLE3 (topk=1): with
``--speculative-adaptive`` every candidate step count gets its own draft / verify CUDA graphs
(``adaptive_runtime_state.AdaptiveController``), and ``EAGLEWorkerV2`` switches between them per batch
(``activate_step_by_batch`` at the start of the decode branch of ``forward_batch_generation``). Its own policy
routes by batch size and the batch's mean acceptance.

KAD keeps that machinery and replaces the policy:

* before a decode batch runs, every request gets its ladder width k_i (5, 7, 9, 11, 12 while everything is
  accepted, max(5, accepted + 1) after a rejection); the batch runs K_b = max k_i, rounded up to the smallest
  captured step count (draft steps and verify length really follow K_b, so low-acceptance batches cost less);
* a batch with a grammar request runs the launch K; on any error the batch keeps the native choice;
* after verification (``BatchResultProcessor._resolve_spec_v2_tokens`` -> ``on_verify_complete_cpu``) every
  request's ladder is updated with v = the step count its batch ran and a = its accepted drafts. Overlapped
  scheduling is handled by tagging each forward result with the step count it ran.

TP ranks see the same batches and the same accept counts, so their widths agree; the policy file is read by
TP rank 0 and broadcast over the TP CPU group every YEE_KAD_RELOAD_EVERY steps. Only EAGLE / EAGLE3 / NEXTN /
MTP / STANDALONE chains run through EAGLEWorkerV2; DSPARK / DFLASH / NGRAM / UNO / multi-layer EAGLE keep a fixed
width per batch (native). Upstream SGLang has no adaptive runtime: not installed there.

Launch (draft count 12): ``--speculative-algorithm EAGLE3 --speculative-num-steps 12
--speculative-num-draft-tokens 13 --speculative-eagle-topk 1 --speculative-adaptive
--speculative-adaptive-config <yee_kad_sglang_adaptive.json>`` with candidate steps [5, 7, 9, 11, 12].
Switch: YEE_KAD=1 (the Yee autoload calls install() in every SGLang process).
"""
from __future__ import annotations

import importlib
import os
import threading

from .. import kad as _core

BASE = "sglang"
MARK = "__yee_kad__"
WORKER_MOD = "sglang.srt.speculative.eagle_worker_v2"
WORKER_CLS = "EAGLEWorkerV2"
ADAPTIVE_MOD = "sglang.srt.speculative.adaptive_runtime_state"
RESULT_MOD = "sglang.srt.managers.scheduler_components.batch_result_processor"
RESULT_METHOD = "_resolve_spec_v2_tokens"
KAD_STEPS = (5, 7, 9, 11, 12)
ADAPTIVE_CONFIG = {"1": {"candidate_steps": list(KAD_STEPS)}}
STALE_STEPS = 512

_KAD = None
_LOCK = threading.RLock()
_ORIG = []
_TLS = threading.local()
_SEEN = {}


# ------------------------------------------------------------------------------------------- KAD state
def _tp_sync():
    """A sync callable for yee.kad (TP rank 0 broadcasts the policy bytes), or None for one rank."""
    try:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return None, True
        from sglang.srt.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
        if getattr(tp, "world_size", 1) <= 1:
            return None, True
        group, src = tp.cpu_group, tp.ranks[0]
        leader = dist.get_rank() == src

        def sync(data):
            box = [data if leader else None]
            dist.broadcast_object_list(box, src=src, group=group)
            return box[0]

        return sync, leader
    except Exception as e:  # noqa: BLE001  single process or unknown layout
        print("%s %s TP policy sync unavailable (%r); every rank keeps the default policy" % (_core.TAG, BASE, e),
              flush=True)
        return (lambda data: None), True


def kad():
    global _KAD
    with _LOCK:
        if _KAD is None:
            sync, leader = _tp_sync()
            _KAD = _core.Kad(BASE, _core.KMAX, enabled=True, leader=leader, sync=sync)
        return _KAD


def _err(what, e):
    k = _KAD
    if k is not None:
        k.error(what, e)


# ------------------------------------------------------------------------------------------- width
def candidate_steps(controller) -> list:
    """The step counts the adaptive controller has graphs for (sorted)."""
    states = getattr(controller, "_states", None)
    if isinstance(states, dict) and states:
        return sorted(int(s) for s in states)
    params = getattr(controller, "params", None)
    return sorted(int(s) for s in (getattr(params, "candidate_steps", None) or []))


def round_up(k: int, steps: list) -> int:
    """The smallest captured step count >= k (the largest when none is)."""
    ups = [s for s in steps if s >= k]
    return min(ups) if ups else max(steps)


def _structured(req) -> bool:
    return getattr(req, "grammar", None) is not None


def choose_steps(worker, reqs) -> int | None:
    """The step count KAD runs for a decode batch, or None to keep the native choice."""
    ctl = getattr(worker, "adaptive_controller", None)
    if ctl is None or not reqs:
        return None
    steps = candidate_steps(ctl)
    if not steps:
        return None
    k = kad()
    if k.kmax_run != max(steps):
        k.set_kmax_run(max(steps))
    step = k.begin_step()
    rids = [r.rid for r in reqs]
    for rid in rids:
        _SEEN[rid] = step
    if step % STALE_STEPS == 0:
        for rid in [r for r, s in _SEEN.items() if step - s > STALE_STEPS]:
            _SEEN.pop(rid, None)
            k.release(rid)
    if any(_structured(r) for r in reqs):
        s = max(steps)
        k.note_batch([s] * len(rids), k_used=s)
        return s
    widths = k.widths(rids)
    s = round_up(max(widths.values()), steps)
    k.note_batch(widths, k_used=s)
    return s


# ------------------------------------------------------------------------------------------- hooks
def _wrap_forward(orig):
    def forward_batch_generation(self, batch, *args, **kwargs):
        steps = None
        try:
            mode = getattr(batch, "forward_mode", None)
            if mode is not None and mode.is_decode():
                steps = choose_steps(self, list(getattr(batch, "reqs", None) or []))
        except Exception as e:  # noqa: BLE001  keep the native choice
            _err("width", e)
            steps = None
        self._yee_kad_pending = steps
        try:
            out = orig(self, batch, *args, **kwargs)
        finally:
            self._yee_kad_pending = None
        if steps is not None:
            try:
                out._yee_kad_steps = steps  # carried with the result through the overlap queue
            except Exception:  # noqa: BLE001
                pass
        return out

    forward_batch_generation.__wrapped__ = orig
    return forward_batch_generation


def _wrap_activate(orig):
    def activate_step_by_batch(self, batch_size):
        steps = getattr(self, "_yee_kad_pending", None)
        ctl = getattr(self, "adaptive_controller", None)
        if steps is None or ctl is None:
            return orig(self, batch_size)
        try:
            if int(steps) != int(self.speculative_num_steps):
                ctl._activate(int(steps))
        except Exception as e:  # noqa: BLE001
            _err("activate", e)
            return orig(self, batch_size)
        return None

    activate_step_by_batch.__wrapped__ = orig
    return activate_step_by_batch


def _wrap_verify(orig):
    def on_verify_complete_cpu(self, num_correct_drafts_per_req, batch_size=0):
        ctx = getattr(_TLS, "ctx", None)
        if ctx is None or getattr(self, "adaptive_controller", None) is None:
            return orig(self, num_correct_drafts_per_req, batch_size)
        reqs, steps = ctx
        try:
            k = kad()
            v = int(steps if steps is not None else self.speculative_num_steps)
            for req, a in zip(reqs, list(num_correct_drafts_per_req)):
                k.observe(req.rid, v, int(a))
        except Exception as e:  # noqa: BLE001
            _err("observe", e)
        return None  # KAD replaces the controller's own batch policy

    on_verify_complete_cpu.__wrapped__ = orig
    return on_verify_complete_cpu


def _wrap_resolve(orig):
    def _resolve_spec_v2_tokens(self, result, batch, *args, **kwargs):
        prev = getattr(_TLS, "ctx", None)
        _TLS.ctx = (list(getattr(batch, "reqs", None) or []), getattr(result, "_yee_kad_steps", None))
        try:
            return orig(self, result, batch, *args, **kwargs)
        finally:
            _TLS.ctx = prev

    _resolve_spec_v2_tokens.__wrapped__ = orig
    return _resolve_spec_v2_tokens


def _skip(reason):
    print("%s %s not installed: %s" % (_core.TAG, BASE, reason), flush=True)
    return False, reason


def _result_class(mod):
    for value in vars(mod).values():
        if isinstance(value, type) and RESULT_METHOD in vars(value):
            return value
    return None


def _patch(cls, name, wrapper):
    cur = cls.__dict__[name]
    if getattr(cur, MARK, False):
        return
    new = wrapper(cur)
    setattr(new, MARK, True)
    setattr(cls, name, new)
    _ORIG.append((cls, name, cur))


def install(enabled=None):
    """Patch the Mia SGLang adaptive runtime. Returns (installed, reason)."""
    on = _core.env_enabled() if enabled is None else bool(enabled)
    if not on:
        return False, "YEE_KAD is off"
    try:
        importlib.import_module(ADAPTIVE_MOD)
    except Exception as e:  # noqa: BLE001
        return _skip("no adaptive speculative runtime (%s: %r); KAD needs the Mia SGLang build" % (ADAPTIVE_MOD, e))
    try:
        wmod = importlib.import_module(WORKER_MOD)
        rmod = importlib.import_module(RESULT_MOD)
    except Exception as e:  # noqa: BLE001
        return _skip("import failed: %r" % (e,))
    worker = getattr(wmod, WORKER_CLS, None)
    result_cls = _result_class(rmod)
    need = ("forward_batch_generation", "activate_step_by_batch", "on_verify_complete_cpu")
    if worker is None or any(n not in vars(worker) for n in need):
        return _skip("%s.%s lacks %s" % (WORKER_MOD, WORKER_CLS, ", ".join(need)))
    if result_cls is None:
        return _skip("no class with %s in %s" % (RESULT_METHOD, RESULT_MOD))
    with _LOCK:
        if getattr(vars(worker)["forward_batch_generation"], MARK, False):
            return True, "already installed"
        _patch(worker, "forward_batch_generation", _wrap_forward)
        _patch(worker, "activate_step_by_batch", _wrap_activate)
        _patch(worker, "on_verify_complete_cpu", _wrap_verify)
        _patch(result_cls, RESULT_METHOD, _wrap_resolve)
    print("%s %s installed on %s.%s (pid %d); needs --speculative-adaptive with candidate steps %s"
          % (_core.TAG, BASE, WORKER_MOD, WORKER_CLS, os.getpid(), list(KAD_STEPS)), flush=True)
    return True, "patched %s and %s.%s" % (WORKER_CLS, result_cls.__name__, RESULT_METHOD)


def uninstall() -> bool:
    with _LOCK:
        done = bool(_ORIG)
        while _ORIG:
            cls, name, orig = _ORIG.pop()
            setattr(cls, name, orig)
        return done


def status() -> dict:
    out = {"base": BASE, "installed": bool(_ORIG), "kad_steps": list(KAD_STEPS)}
    if _KAD is not None:
        out["kad"] = _KAD.snapshot()
    return out


__all__ = ["BASE", "KAD_STEPS", "ADAPTIVE_CONFIG", "candidate_steps", "round_up", "choose_steps", "install",
           "uninstall", "status", "kad"]
