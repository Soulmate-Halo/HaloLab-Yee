"""Recall copy drafts for SGLang's DFLASH block drafter (DFlashWorkerV2, spec v2 / overlap).

Injection point (ysy37 dsv41-head, sglang 0.0.0.dev0, speculative/dflash_worker_v2.py):
  2578  draft_tokens = self._draft_block_tokens_buf[:bs]
  2579  draft_tokens[:, 0].copy_(block_ids[:, 0])          # the bonus / anchor
  2580  draft_tokens[:, 1:].copy_(draft_next)              # block_size - 1 drafts
  2591  verify_input_ids = draft_tokens.reshape(-1)
  2592  verify_input = DFlashVerifyInput(draft_token=verify_input_ids, positions=..., draft_token_num=block_size, ...)
  2670  candidates = draft_tokens                          # what _accept_block compares with the target's argmax
``DFlashVerifyInput`` is a module-level name there (line 54 ``from sglang.srt.speculative.dflash_info import
DFlashVerifyInput``) used only at 2281 (idle) and 2592, so the module's name is replaced by a function that merges
the copies into ``draft_token`` (a view of the same buffer ``candidates`` reads) and then builds the real object.
The batch comes from a thin wrapper around ``DFlashWorkerV2.forward_batch_generation``.

Layout = DSpark's: anchor at position ``batch.seq_lens`` (= prefix_lens, line 2354), gamma = block_size - 1.
Off for: grammar batches (sampled batches only while RECALL_ACCEL_NONGREEDY is on), the selector sampler (``_selector_sample`` set), idle batches, capture.
"""
from __future__ import annotations

import sys
import threading

from . import sglang_common as sc
from . import sglang_dspark as sd

WORKER = "sglang.srt.speculative.dflash_worker_v2"
PROCESSOR = "sglang.srt.managers.scheduler_components.batch_result_processor"
_ctx = threading.local()

REQUIRED_SYMBOLS = (
    (WORKER, "DFlashWorkerV2", ("forward_batch_generation", "_accept_block")),
    (WORKER, None, ("DFlashVerifyInput", "_is_all_greedy")),
    ("sglang.srt.speculative.dflash_info", "DFlashVerifyInput", ()),
    (PROCESSOR, "SchedulerBatchResultProcessor", ("process_batch_result_decode",)),
    ("sglang.srt.distributed", None, ("get_tp_group",)),
)
SIGNATURES = ((WORKER, "DFlashWorkerV2.forward_batch_generation", ("self", "batch")),)
SOURCE_MARKERS = ((WORKER, "DFlashWorkerV2", (
    "draft_tokens = self._draft_block_tokens_buf[:bs]", "draft_tokens[:, 0].copy_(block_ids[:, 0])",
    "verify_input_ids = draft_tokens.reshape(-1)", "candidates = draft_tokens", "prefix_lens = batch.seq_lens")),)


def participates(worker, batch, draft_token, n: int) -> bool:
    if worker is None or batch is None or draft_token is None or sc.capturing():
        return False
    fm = getattr(batch, "forward_mode", None)
    if fm is not None and getattr(fm, "is_idle", lambda: False)():
        return False
    reqs = getattr(batch, "reqs", None) or []
    bs = len(reqs)
    if not (0 < bs <= sc.MAX_BS) or n < 2 or draft_token.numel() != bs * n:
        return False
    if getattr(worker, "_selector_sample", None) is not None:
        return False
    mod = sys.modules.get(WORKER)
    fn = getattr(mod, "_is_all_greedy", None) if mod is not None else None
    if fn is not None and not fn(getattr(batch, "sampling_info", None)) and not sc.nongreedy_enabled():
        return False
    return sc.copy_batch(batch)


def merge(batch, draft_token, n: int):
    bs = len(batch.reqs)
    view = draft_token.view(bs, n)
    gamma = n - 1
    plan, _, _ = sc.plan_broadcast(batch.reqs, gamma, draft_token.device)
    return sd.device_merge(plan, batch.seq_lens, view[:, 0], view[:, 1:], None, gamma)


def wrap_verify_input(cls):
    if getattr(cls, "__recall_wrapped__", False):
        return cls

    def DFlashVerifyInput(*a, **kw):  # noqa: N802  stands in for the class name in that module only
        try:
            cur = getattr(_ctx, "cur", None)
            dt = kw.get("draft_token")
            n = int(kw.get("draft_token_num") or 0)
            if cur is not None and participates(cur[0], cur[1], dt, n):
                merge(cur[1], dt, n)
        except Exception as e:  # noqa: BLE001
            sd._counts["plan_errors"] += 1
            if sd._counts["plan_errors"] <= 3:
                print("[recall] dflash hook error (model drafts kept): %r" % (e,), flush=True)
        return cls(*a, **kw)

    DFlashVerifyInput.__wrapped__ = cls
    DFlashVerifyInput.__recall_wrapped__ = True
    return DFlashVerifyInput


def _patch_worker(module) -> None:
    cls = module.DFlashWorkerV2
    orig = cls.forward_batch_generation
    if getattr(orig, "__recall_wrapped__", False):
        return

    def forward_batch_generation(self, batch, *a, **kw):
        prev = getattr(_ctx, "cur", None)
        _ctx.cur = (self, batch)
        try:
            return orig(self, batch, *a, **kw)
        finally:
            _ctx.cur = prev

    forward_batch_generation.__wrapped__ = orig
    forward_batch_generation.__recall_wrapped__ = True
    cls.forward_batch_generation = forward_batch_generation
    module.DFlashVerifyInput = wrap_verify_input(module.DFlashVerifyInput)
    print("[recall] patched %s (DFLASH block drafts)" % module.__name__, flush=True)


TARGETS = {WORKER: _patch_worker, PROCESSOR: sd._patch_processor_module}


def install(targets=None) -> None:
    sd.install(TARGETS if targets is None else targets)
