"""Recall copy drafts for SGLang's EAGLE family (EAGLE, EAGLE3, NEXTN/MTP, STANDALONE, multi-layer EAGLE, UNO).

Injection point (ysy37 dsv41-head, sglang 0.0.0.dev0):
  speculative/eagle_worker_common.py:316 ``build_eagle_verify_input(batch, draft_input, parent_list,
  top_scores_index, draft_tokens, draft_probs, *, target_worker, topk, num_steps, num_draft_tokens, ...)``
  is the only place an EagleVerifyInput is built from draft tokens; it is called by
  eagle_worker_v2.py:651 (EagleDraftWorker.draft), multi_layer_eagle_worker_v2.py:464 and uno_worker_v2.py:550.
  Line 373 hands ``draft_input.bonus_tokens`` and ``draft_tokens`` to ``build_tree_kernel_efficient`` which
  starts with ``torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1)``.

With topk == 1 the tree is a chain (base_spec_worker.py:119 asserts num_draft_tokens == num_steps + 1):
``draft_tokens`` is [bs, num_steps], column i the (i+1)-th id after the bonus, and the bonus sits at position
``seq_lens`` - exactly DSpark's layout (anchor = bonus, gamma = num_steps). So the wrapper swaps aligned copies
into ``draft_tokens`` before the tree kernel runs; masks/positions come out unchanged because a chain's are
independent of the token values. topk > 1 (a real tree) is left alone: replacing a leaf path would need the
draft scores to stay consistent with the tokens, which this hook cannot guarantee, so copies stay off there.

Also off: rejection-sampling mode (``draft_probs`` is not None), grammar batches (sampled batches only while RECALL_ACCEL_NONGREEDY is on), idle batches,
CUDA-graph capture. Accounting is sglang_dspark's result-processor patch (accepted = commit - 1).
"""
from __future__ import annotations

import inspect
import sys

from . import sglang_common as sc
from . import sglang_dspark as sd

COMMON = "sglang.srt.speculative.eagle_worker_common"
CALLERS = (
    "sglang.srt.speculative.eagle_worker_v2",
    "sglang.srt.speculative.multi_layer_eagle_worker_v2",
    "sglang.srt.speculative.uno_worker_v2",
)
PROCESSOR = "sglang.srt.managers.scheduler_components.batch_result_processor"
BUILD_PARAMS = ("batch", "draft_input", "parent_list", "top_scores_index", "draft_tokens", "draft_probs",
                "target_worker", "topk", "num_steps", "num_draft_tokens", "tree_mask_mode", "device")

REQUIRED_SYMBOLS = (
    (COMMON, None, ("build_eagle_verify_input",)),
    ("sglang.srt.speculative.eagle_worker_v2", "EagleDraftWorker", ("draft", "draft_forward")),
    ("sglang.srt.speculative.eagle_worker_v2", "EAGLEWorkerV2", ("forward_batch_generation",)),
    ("sglang.srt.speculative.eagle_utils", None, ("build_tree_kernel_efficient",)),
    (PROCESSOR, "SchedulerBatchResultProcessor", ("process_batch_result_decode",)),
    ("sglang.srt.distributed", None, ("get_tp_group",)),
)
SIGNATURES = ((COMMON, "build_eagle_verify_input", BUILD_PARAMS),)


def participates(a) -> bool:
    """Same answer on every TP rank (depends on batch / config only)."""
    batch = a.get("batch")
    if batch is None or sc.capturing():
        return False
    fm = getattr(batch, "forward_mode", None)
    if fm is not None and getattr(fm, "is_idle", lambda: False)():
        return False
    reqs = getattr(batch, "reqs", None) or []
    bs = len(reqs)
    if not (0 < bs <= sc.MAX_BS):
        return False
    topk, steps, ndt = int(a.get("topk", 0)), int(a.get("num_steps", 0)), int(a.get("num_draft_tokens", 0))
    if topk != 1:
        sc.warn_once("eagle-topk", "EAGLE topk=%d is a tree: copies stay off (topk=1 chains only)" % topk)
        return False
    if steps < 1 or ndt != steps + 1:
        return False
    if a.get("draft_probs") is not None:
        sc.warn_once("eagle-rs", "EAGLE rejection-sampling mode (draft_probs): copies stay off")
        return False
    dt = a.get("draft_tokens")
    if dt is None or dt.dim() != 2 or tuple(dt.shape) != (bs, steps):
        return False
    bonus = getattr(a.get("draft_input"), "bonus_tokens", None)
    if bonus is None or bonus.numel() != bs:
        return False
    return sc.copy_batch(batch)


def merge(a):
    batch = a["batch"]
    dt = a["draft_tokens"]
    gamma = int(a["num_steps"])
    plan, _, _ = sc.plan_broadcast(batch.reqs, gamma, dt.device)
    return sd.device_merge(plan, batch.seq_lens, a["draft_input"].bonus_tokens.reshape(-1), dt, None, gamma)


def wrap_build(orig):
    if getattr(orig, "__recall_wrapped__", False):
        return orig
    try:
        sig = inspect.signature(orig)
    except (TypeError, ValueError):
        sig = None

    def build_eagle_verify_input(*args, **kw):
        try:
            if sig is not None:
                a = sig.bind(*args, **kw).arguments
                if participates(a):
                    merge(a)
        except Exception as e:  # noqa: BLE001  never break serving
            sd._counts["plan_errors"] += 1
            if sd._counts["plan_errors"] <= 3:
                print("[recall] eagle hook error (model drafts kept): %r" % (e,), flush=True)
        return orig(*args, **kw)

    build_eagle_verify_input.__wrapped__ = orig
    build_eagle_verify_input.__recall_wrapped__ = True
    build_eagle_verify_input.__name__ = getattr(orig, "__name__", "build_eagle_verify_input")
    if sig is not None:
        build_eagle_verify_input.__signature__ = sig
    return build_eagle_verify_input


def _patch_common(module) -> None:
    module.build_eagle_verify_input = wrap_build(module.build_eagle_verify_input)
    print("[recall] patched %s.build_eagle_verify_input (EAGLE family, topk=1)" % module.__name__, flush=True)


def _patch_caller(module) -> None:
    fn = getattr(module, "build_eagle_verify_input", None)
    if fn is not None and not getattr(fn, "__recall_wrapped__", False):
        module.build_eagle_verify_input = wrap_build(fn)
        print("[recall] rebound build_eagle_verify_input in %s" % module.__name__, flush=True)


TARGETS = dict({COMMON: _patch_common, PROCESSOR: sd._patch_processor_module},
               **{name: _patch_caller for name in CALLERS})


def install(targets=None) -> None:
    sd.install(TARGETS if targets is None else targets)


__all__ = ["install", "TARGETS", "REQUIRED_SYMBOLS", "SIGNATURES", "participates", "merge", "wrap_build"]
_ = sys  # keep the import for tests that patch sys.modules
