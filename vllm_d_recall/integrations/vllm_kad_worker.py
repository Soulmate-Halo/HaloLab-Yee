"""KAD worker side: any vLLM draft head drafts a per-request block width.

The scheduler side (``vllm_kad``) attaches ``_xy_kw = {"v": 1, "w": {req_id: int}}`` to every
``SchedulerOutput``; ``w[req_id]`` is the number of drafts the request needs from this step.  The
runner patch calls ``on_execute_model(scheduler_output)`` at the start of
``GPUModelRunner.execute_model``; the ``propose`` of the same batch (inside ``sample_tokens``, which
strictly alternates with ``execute_model`` on a worker) consumes the map and drafts ``w[i]`` tokens
for request ``i``.

Three kinds of draft head are covered:

* DSpark goes through the varlen path (``make_varlen_propose``): one varlen forward of the draft
  head with a per-request query block (``[Q[i], Q[i] + w[i])``, ``Q`` the exclusive prefix sum of
  ``w``); the sequential Markov head runs ``max(w)`` columns and the columns ``>= w[i]`` of request
  ``i`` stay inert (``sample_idx_mapping = -1``: no draft logits are written).
* The autoregressive heads (eagle, eagle3, mtp, gemma4 mtp) and the plain draft model go through
  the steps path (``make_steps_propose``): the draft head runs only ``max(w)`` decoding steps
  instead of ``K``, so the step count follows the widest request of the batch.
* DFlash, DFlash2, LiLiCorr and the multi-module MTP produce ``K_max`` drafts natively; the
  scheduler shortens them per request, so they keep the native propose.

Safety: whenever a path cannot run (configuration it does not cover, no map, a dummy run, any
exception) this module calls the native ``propose``, which drafts ``K_max`` tokens for every
request, which always covers what the scheduler asked for.

Only the standard library is imported at module level; torch / numpy / vllm and the Triton kernel
(``vllm_kad_kernel``) are imported on first use.
"""
from __future__ import annotations

import inspect
import os
import time

TAG = "[vllm-d-recall]"
KW_TAG = "[vllm-d-recall] kad(worker)"
ATTR_SO = "_xy_kw"
ENV_LOG_S = "VLLM_D_RECALL_KAD_LOG_S"

# The block-width map of the batch the worker is executing (set by on_execute_model).
_w = {"map": None}

_state = {
    "mods": None,
    "enabled_fn": None,
    "log_s": None,
    "log_t": 0.0,
    "err_printed": 0,
    "errors": 0,
    "varlen_steps": 0,
    "fallback": {},
    "sum_w": 0,
    "sum_reqs": 0,
    "max_w_hist": {},
    "steps_runs": 0,
    "steps_sum_n": 0,
    "steps_sum_reqs": 0,
    "steps_hist": {},
    "cgm_none": None,
}


# ------------------------------------------------------------------------------------------- helpers
def _mods() -> dict:
    """torch / numpy / vllm pieces and the kernel module, imported once."""
    m = _state["mods"]
    if m is None:
        import numpy as np
        import torch
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

        from . import vllm_kad_kernel

        m = {
            "np": np,
            "torch": torch,
            "CGM": CUDAGraphMode,
            "BED": BatchExecutionDescriptor,
            "slots_by_layer": build_slot_mappings_by_layer,
            "kern": vllm_kad_kernel,
        }
        _state["mods"] = m
    return m


def _enabled() -> bool:
    fn = _state["enabled_fn"]
    if fn is None:
        try:
            from .vllm_kad import enabled as fn
        except Exception:  # noqa: BLE001  scheduler half missing: treat as disabled
            def fn():
                return False
        _state["enabled_fn"] = fn
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001
        return False


def _log_s() -> float:
    v = _state["log_s"]
    if v is None:
        raw = (os.environ.get(ENV_LOG_S) or "").strip()
        try:
            v = float(raw) if raw else 30.0
        except ValueError:
            v = 30.0
        v = max(0.0, v)
        _state["log_s"] = v
    return v


def _log_error(what: str, e: BaseException) -> None:
    _state["errors"] += 1
    if _state["err_printed"] < 5:
        _state["err_printed"] += 1
        print("%s %s failed, fallback: %r" % (KW_TAG, what, e), flush=True)


def _count_fallback(reason: str) -> None:
    fb = _state["fallback"]
    fb[reason] = fb.get(reason, 0) + 1


def _count_varlen(num_reqs: int, total: int, max_w: int) -> None:
    _state["varlen_steps"] += 1
    _state["sum_w"] += int(total)
    _state["sum_reqs"] += int(num_reqs)
    hist = _state["max_w_hist"]
    hist[int(max_w)] = hist.get(int(max_w), 0) + 1


def _count_steps(n: int, num_reqs: int) -> None:
    """One steps-path batch: ``n`` decoding steps for ``num_reqs`` requests."""
    _state["steps_runs"] += 1
    _state["steps_sum_n"] += int(n)
    _state["steps_sum_reqs"] += int(num_reqs)
    hist = _state["steps_hist"]
    hist[int(n)] = hist.get(int(n), 0) + 1


def counts() -> dict:
    fb = dict(_state["fallback"])
    sum_reqs = _state["sum_reqs"]
    steps_runs = _state["steps_runs"]
    return {
        "varlen_steps": _state["varlen_steps"],
        "fallback": fb,
        "fallback_total": sum(fb.values()),
        "sum_w": _state["sum_w"],
        "sum_reqs": sum_reqs,
        "mean_w": (float(_state["sum_w"]) / sum_reqs) if sum_reqs else 0.0,
        "max_w_hist": {str(k): v for k, v in sorted(_state["max_w_hist"].items())},
        "steps_runs": steps_runs,
        "mean_n": (float(_state["steps_sum_n"]) / steps_runs) if steps_runs else 0.0,
        "steps_hist": {str(k): v for k, v in sorted(_state["steps_hist"].items())},
        "errors": _state["errors"],
    }


def reset() -> None:
    """Tests: forget the map, the counters and the cached configuration."""
    _w["map"] = None
    _state["enabled_fn"] = None
    _state["log_s"] = None
    _state["log_t"] = 0.0
    _state["err_printed"] = 0
    _state["errors"] = 0
    _state["varlen_steps"] = 0
    _state["fallback"] = {}
    _state["sum_w"] = 0
    _state["sum_reqs"] = 0
    _state["max_w_hist"] = {}
    _state["steps_runs"] = 0
    _state["steps_sum_n"] = 0
    _state["steps_sum_reqs"] = 0
    _state["steps_hist"] = {}


def _maybe_log() -> None:
    log_s = _log_s()
    if log_s <= 0:
        return
    now = time.monotonic()
    if now - _state["log_t"] < log_s:
        return
    _state["log_t"] = now
    c = counts()
    print("%s varlen=%d fallback=%d mean_w=%.2f reasons=%s max_w_hist=%s errors=%d pid=%d"
          " steps=%d mean_n=%.2f steps_hist=%s"
          % (KW_TAG, c["varlen_steps"], c["fallback_total"], c["mean_w"], c["fallback"],
             c["max_w_hist"], c["errors"], os.getpid(), c["steps_runs"], c["mean_n"],
             c["steps_hist"]), flush=True)


# ------------------------------------------------------------------------------------------- runner hook
def on_execute_model(scheduler_output) -> None:
    """Keep the block-width map of the batch about to execute (None when absent or malformed)."""
    obj = getattr(scheduler_output, ATTR_SO, None)
    if isinstance(obj, dict) and obj.get("v") == 1 and isinstance(obj.get("w"), dict):
        _w["map"] = obj["w"]
    else:
        _w["map"] = None


# ------------------------------------------------------------------------------------------- plan
def compute_plan(req_ids, wmap, k):
    """Per-request block width (clamped to [1, k], missing -> k), its exclusive prefix sum,
    the total query tokens and the widest block."""
    k = int(k)
    w = []
    for rid in req_ids:
        try:
            v = int(wmap.get(rid, k))
        except (TypeError, ValueError):
            v = k
        w.append(1 if v < 1 else (k if v > k else v))
    q = [0]
    acc = 0
    for v in w:
        acc += v
        q.append(acc)
    return w, q, acc, (max(w) if w else 0)


def _static_reason(spec) -> str:
    """Configurations the varlen path does not cover ('' when it does)."""
    k = getattr(spec, "num_speculative_steps", None)
    if not hasattr(spec, "sample_from_anchor"):
        # A block head without the anchor layout (DFlash, DFlash2, LiLiCorr): it drafts K_max
        # natively and the scheduler shortens the list per request.
        return "native_block"
    if not getattr(spec, "sample_from_anchor", False) or getattr(spec, "num_query_per_req", None) != k:
        return "layout"
    model = getattr(spec, "model", None)
    if model is None or not hasattr(model, "markov_embed"):
        return "no_markov"
    if getattr(spec, "pcp_manager", None) is not None:
        return "pcp"
    if int(getattr(spec, "dp_size", 1) or 1) != 1:
        return "dp"
    if int(getattr(spec, "dcp_size", 1) or 1) != 1:
        return "dcp"
    if (getattr(spec, "enable_adaptive_verification", False)
            or getattr(spec, "use_confidence_head", False)
            or getattr(spec, "acceptance_estimator", None) is not None):
        return "adaptive"
    if getattr(spec, "draft_watermarker", None) is not None:
        return "watermark"
    # PIECEWISE target -> the draft runs eagerly (DFlashSpeculator.init_cudagraph_manager);
    # a FULL draft graph replays the fixed-width layout, so only NONE is covered.
    mgr = getattr(spec, "query_cudagraph_manager", None)
    mode = getattr(mgr, "cudagraph_mode", None)
    if mode is None or mode != _mods()["CGM"].NONE:
        return "cudagraph"
    return ""


def _decide(spec, input_batch, dummy_run, is_profile, mm_inputs):
    """(None, plan) when the varlen path runs, else (reason, None)."""
    if not _enabled():
        return "disabled", None
    if dummy_run or is_profile:
        return "dummy", None
    # The map belongs to exactly one batch: take it, so that a second propose without a new
    # execute_model can only fall back.
    wmap = _w["map"]
    _w["map"] = None
    reason = _static_reason(spec)
    if reason:
        return reason, None
    if mm_inputs is not None:
        return "mm", None
    if wmap is None:
        return "no_map", None
    num_reqs = int(input_batch.num_reqs)
    if num_reqs <= 0:
        return "empty", None
    k = int(spec.num_speculative_steps)
    plan = compute_plan(input_batch.req_ids[:num_reqs], wmap, k)
    if min(plan[0]) >= k:
        return "full", None
    return None, plan


# ------------------------------------------------------------------------------------------- steps path
def _cgm_none():
    """``CUDAGraphMode.NONE``; imported alone (the kernel / triton imports stay untouched)."""
    cached = _state["cgm_none"]
    if cached is None:
        from vllm.config.compilation import CUDAGraphMode

        cached = _state["cgm_none"] = CUDAGraphMode.NONE
    return cached


def _steps_static_reason(spec) -> str:
    """Configurations the steps path does not cover ('' when it does)."""
    k = getattr(spec, "num_speculative_steps", None)
    if not isinstance(k, int) or k < 2:
        return "k1"
    if getattr(spec, "pcp_manager", None) is not None:
        return "pcp"
    if int(getattr(spec, "dp_size", 1) or 1) != 1:
        # Every rank has its own block width table, so a different step count per rank would
        # misalign the collective communication.
        return "dp"
    if (getattr(spec, "enable_adaptive_verification", False)
            or getattr(spec, "use_acceptance_estimator", False)
            or getattr(spec, "use_confidence_head", False)
            or getattr(spec, "acceptance_estimator", None) is not None):
        return "adaptive"
    if getattr(spec, "draft_watermarker", None) is not None:
        return "watermark"
    mgr = getattr(spec, "decode_cudagraph_manager", None)
    if mgr is not None:
        mode = getattr(mgr, "cudagraph_mode", None)
        if mode is None:
            return "cudagraph"
        # A fused decode graph records K-1 decode steps in one graph: it cannot run fewer.
        if mode != _cgm_none() and getattr(spec, "use_fused_multi_step_decode", False):
            return "cudagraph"
    return ""


def _decide_steps(spec, input_batch, dummy_run, is_profile):
    """(None, n) when the steps path runs with n decoding steps, else (reason, None)."""
    if not _enabled():
        return "disabled", None
    if dummy_run or is_profile:
        return "dummy", None
    # The map belongs to exactly one batch: take it, so that a second propose without a new
    # execute_model can only fall back.
    wmap = _w["map"]
    _w["map"] = None
    reason = _steps_static_reason(spec)
    if reason:
        return reason, None
    if wmap is None:
        return "no_map", None
    num_reqs = int(input_batch.num_reqs)
    if num_reqs <= 0:
        return "empty", None
    k = int(spec.num_speculative_steps)
    plan = compute_plan(input_batch.req_ids[:num_reqs], wmap, k)
    max_w = int(plan[3])
    if max_w >= k:
        return "full", None
    return None, max_w


def kad_path_of(spec) -> str:
    """Which KAD path a speculator belongs to: 'varlen', 'steps' or 'native'."""
    names = [c.__name__ for c in type(spec).__mro__]
    if "DSparkSpeculator" in names:
        return "varlen"
    if "AutoRegressiveSpeculator" in names or "PlainDraftModelSpeculator" in names:
        return "steps"
    return "native"


# ------------------------------------------------------------------------------------------- varlen path
def _sample_varlen(spec, num_reqs: int, max_w: int, head_hidden, q_gpu) -> None:
    """DSparkSpeculator._sample_sequential(_topk) with n_spec = max_w and per-request anchors."""
    num_sample = num_reqs * max_w
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base_logits = spec.model.compute_draft_logits(sample_hidden)
    idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, max_w)
    sample_pos = spec.sample_pos[:num_sample].view(num_reqs, max_w)
    # Anchor (bonus) token of request i = the input id at its query offset 0, i.e. at Q[i].
    prev = spec.input_buffers.input_ids[q_gpu[:num_reqs].long()]
    topk = getattr(spec, "_draft_topk", None)
    if topk is None:
        base_logits = base_logits.view(num_reqs, max_w, base_logits.shape[-1])
        for i in range(max_w):
            markov_embed = spec.model.markov_embed(prev)
            bias = spec.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            sampled = spec._sample_logits(logits_i, idx_map[:, i], sample_pos[:, i], i)
            spec.draft_tokens[:num_reqs, i] = sampled
            prev = sampled
        return
    base_logits = base_logits.view(num_reqs, max_w, -1)
    base_values, draft_indices = base_logits.topk(topk, dim=-1)
    base_logits.fill_(float("-inf"))
    for i in range(max_w):
        markov_embed = spec.model.markov_embed(prev)
        logits_i = spec.model.apply_markov_bias_gathered(
            markov_embed,
            base_logits[:, i],
            base_values[:, i],
            draft_indices[:, i],
        )
        sampled = spec._sample_logits(logits_i, idx_map[:, i], sample_pos[:, i], i)
        spec.draft_tokens[:num_reqs, i] = sampled
        prev = sampled


def _varlen_propose(spec, input_batch, last_hidden_states, aux_hidden_states, num_sampled,
                    num_rejected, last_sampled, next_prefill_tokens, temperature, seeds, plan):
    """DFlashSpeculator.propose (non-dummy, eager) with a per-request query block."""
    m = _mods()
    torch = m["torch"]
    np = m["np"]
    w, q, total, max_w = plan
    num_reqs = int(input_batch.num_reqs)
    num_target_tokens = input_batch.num_tokens
    with torch.inference_mode():
        ub = input_batch.seq_lens_cpu_upper_bound[:num_reqs]
        w_cpu = torch.tensor(w, dtype=torch.int32)
        ub_w = ub + w_cpu.to(ub.dtype)
        spec.draft_max_seq_len = min(int(ub_w.max()), spec.max_model_len)

        if aux_hidden_states:
            hidden_states = spec.model.combine_hidden_states(torch.cat(aux_hidden_states, dim=-1))
        else:
            hidden_states = last_hidden_states
        spec.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])
        spec.prepare_context_anchor(input_batch, num_rejected)

        q_np = np.asarray(q, dtype=np.int32)
        dev = spec.device
        w_gpu = w_cpu.pin_memory().to(dev, non_blocking=True)
        q_gpu = torch.from_numpy(q_np).pin_memory().to(dev, non_blocking=True)
        kern = m["kern"]
        for i, gid in enumerate(spec.draft_kv_cache_group_ids):
            kern.prepare_dspark_varlen_inputs(
                spec, i, gid, input_batch, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds, w_gpu, q_gpu, max_w, total,
            )

        batch_desc = m["BED"](cg_mode=m["CGM"].NONE, num_tokens=total, num_reqs=num_reqs)
        spec._precompute_context_kv(0, num_target_tokens, False)
        attn_metadata = spec._build_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=batch_desc,
            query_start_loc_np=q_np,
            seq_lens_cpu_upper_bound=ub_w,
            step=0,
            causal=spec._group_causal,
        )
        slot_mappings = m["slots_by_layer"](
            spec.block_tables.slot_mappings[:, :total], spec.kv_cache_config
        )
        spec._prepare_eplb_forward(total)
        head_hidden = spec._run_model(total, attn_metadata, slot_mappings, None, m["CGM"].NONE)
        _sample_varlen(spec, num_reqs, max_w, head_hidden, q_gpu)
        return spec.draft_tokens[:num_reqs]


# ------------------------------------------------------------------------------------------- patching
def make_varlen_propose(orig_propose):
    """A drop-in ``DFlashSpeculator.propose`` that drafts per-request block widths (DSpark)."""
    orig = orig_propose

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        try:
            reason, plan = _decide(self, input_batch, dummy_run, is_profile, mm_inputs)
        except Exception as e:  # noqa: BLE001
            reason, plan = "error", None
            _log_error("check", e)
        if reason is None:
            try:
                out = _varlen_propose(
                    self, input_batch, last_hidden_states, aux_hidden_states, num_sampled,
                    num_rejected, last_sampled, next_prefill_tokens, temperature, seeds, plan,
                )
            except Exception as e:  # noqa: BLE001  the native propose rewrites every buffer
                reason = "error"
                _log_error("varlen", e)
            else:
                _count_varlen(len(plan[0]), plan[2], plan[3])
                try:
                    _maybe_log()
                except Exception:  # noqa: BLE001
                    pass
                return out
        _count_fallback(reason)
        try:
            _maybe_log()
        except Exception:  # noqa: BLE001
            pass
        return orig(
            self,
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync=dp_sync,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
        )

    propose.__recall_kad_wrapped__ = True
    propose.__wrapped__ = orig
    propose.__name__ = getattr(orig, "__name__", "propose")
    propose.__qualname__ = getattr(orig, "__qualname__", "propose")
    propose.__doc__ = getattr(orig, "__doc__", None)
    return propose


def make_steps_propose(orig_propose):
    """A drop-in ``propose`` that runs only ``max(w)`` decoding steps (AR and draft model heads)."""
    orig = orig_propose

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        try:
            reason, n = _decide_steps(self, input_batch, dummy_run, is_profile)
        except Exception as e:  # noqa: BLE001
            reason, n = "error", None
            _log_error("check", e)
        if reason is None:
            num_reqs = int(input_batch.num_reqs)
            k = self.num_speculative_steps
            self.num_speculative_steps = n
            ok = False
            try:
                orig(
                    self,
                    input_batch,
                    attn_metadata,
                    slot_mappings,
                    last_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                    temperature,
                    seeds,
                    dp_sync=dp_sync,
                    dummy_run=dummy_run,
                    skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                    mm_inputs=mm_inputs,
                    is_profile=is_profile,
                )
                ok = True
            except Exception as e:  # noqa: BLE001  the native propose runs the full K steps
                reason = "error"
                _log_error("steps", e)
            finally:
                self.num_speculative_steps = k
            if ok:
                _count_steps(n, num_reqs)
                try:
                    _maybe_log()
                except Exception:  # noqa: BLE001
                    pass
                # The native propose returns only the first column when K == 1: the full width is
                # returned here, the scheduler only sends w[i] columns for verification.
                return self.draft_tokens[:num_reqs]
        _count_fallback(reason)
        try:
            _maybe_log()
        except Exception:  # noqa: BLE001
            pass
        return orig(
            self,
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync=dp_sync,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
        )

    propose.__recall_kad_wrapped__ = True
    propose.__wrapped__ = orig
    propose.__name__ = getattr(orig, "__name__", "propose")
    propose.__qualname__ = getattr(orig, "__qualname__", "propose")
    propose.__doc__ = getattr(orig, "__doc__", None)
    try:
        propose.__signature__ = inspect.signature(orig)
    except (TypeError, ValueError):  # pragma: no cover
        pass
    return propose
