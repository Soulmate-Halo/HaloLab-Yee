"""Recall copy drafts for SGLang's NGRAM speculator (NGRAMWorker, CPU trie drafts + tree verify).

Injection point (ysy37 dsv41-head, sglang 0.0.0.dev0, speculative/ngram_worker.py):
  242-312  ``_prepare_draft_tokens(batch) -> (req_drafts, mask)``: numpy, ``bs * draft_token_num`` ids
           (each request's block starts with its current last id, the tree root) and a
           ``bs * n * n`` attention mask (row q may attend column k); asserted at 309.
  341-358  the caller copies both to the GPU and rebuilds positions / retrieve_* from the mask
           (``reconstruct_indices_from_tree_mask``), so any valid tree - a chain included - verifies.
  253      overlap: the last round's accepted ids are not in ``req.output_ids`` yet; the worker spliced them
           from ``self.prev_token_ids`` / ``self.prev_accept_lens`` (stride n) - known exactly on the CPU here.

The wrapper replaces a greedy request's whole block by a chain: root (kept), then the record's continuation, with
a lower-triangular mask. A row is used only when the CPU-known length L and the spliced ids line up exactly with
the planned row (W[0..o] == [last known id] + prev ids, o = len(prev)), so nothing depends on a lagging view.
rank0 plans, one broadcast (the step already syncs to the CPU), every rank merges identically.
"""
from __future__ import annotations

import numpy as np

from . import sglang_common as sc
from . import sglang_dspark as sd

WORKER = "sglang.srt.speculative.ngram_worker"
PROCESSOR = "sglang.srt.managers.scheduler_components.batch_result_processor"

REQUIRED_SYMBOLS = (
    (WORKER, "NGRAMWorker", ("_prepare_draft_tokens", "_prepare_for_speculative_decoding", "forward_batch_generation")),
    (PROCESSOR, "SchedulerBatchResultProcessor", ("process_batch_result_decode",)),
    ("sglang.srt.distributed", None, ("get_tp_group",)),
)
SIGNATURES = ((WORKER, "NGRAMWorker._prepare_draft_tokens", ("self", "batch")),)
SOURCE_MARKERS = ((WORKER, "NGRAMWorker", (
    "req_drafts, mask = self._prepare_draft_tokens(batch)", "reconstruct_indices_from_tree_mask(",
    "self.prev_token_ids", "self.prev_accept_lens", "total_draft_token_num == bs * self.draft_token_num")),)


def _prev_tokens(worker, i: int, n: int):
    ids = getattr(worker, "prev_token_ids", None) or []
    lens = getattr(worker, "prev_accept_lens", None) or []
    if i >= len(lens):
        return []
    k = int(lens[i])
    return [int(t) for t in ids[i * n: i * n + k]]


def row_ok(row, prev, root: int, gamma: int) -> bool:
    """CPU check for one plan row [flag, L, W...] against the ids known exactly right now."""
    if not row[0]:
        return False
    W = row[2:]
    o = len(prev)
    if o > sd._omax(gamma) or o + gamma >= len(W):
        return False
    if any(int(W[1 + j]) != int(t) for j, t in enumerate(prev)):
        return False
    return int(W[o]) == int(root)


def chain_mask(n: int, dtype) -> np.ndarray:
    return np.tril(np.ones((n, n), dtype=np.int64)).astype(dtype)


def cpu_merge(arr, worker, req_drafts, mask, n: int):
    """Every rank: swap aligned rows for chains (in place). Returns the per-row used flags."""
    bs = arr.shape[0]
    gamma = n - 1
    m3 = mask.reshape(bs, n, n)
    tri = None
    used = []
    for i in range(bs):
        prev = _prev_tokens(worker, i, n)
        ok = row_ok(arr[i], prev, int(req_drafts[i * n]), gamma)
        if ok:
            o = len(prev)
            W = arr[i, 2:]
            req_drafts[i * n + 1: i * n + n] = W[o + 1: o + 1 + gamma].astype(req_drafts.dtype)
            if tri is None:
                tri = chain_mask(n, m3.dtype)
            m3[i] = tri
        used.append(ok)
    return used


def participates(worker, batch, req_drafts, mask) -> bool:
    if batch is None or req_drafts is None or mask is None:
        return False
    reqs = getattr(batch, "reqs", None) or []
    bs = len(reqs)
    n = int(getattr(worker, "draft_token_num", 0) or 0)
    if not (0 < bs <= sc.MAX_BS) or n < 2:
        return False
    if not isinstance(req_drafts, np.ndarray) or not isinstance(mask, np.ndarray):
        return False
    if req_drafts.size != bs * n or mask.size != bs * n * n:
        return False
    return sc.copy_batch(batch)


def on_prepared(worker, batch, req_drafts, mask) -> None:
    import torch

    reqs = batch.reqs
    n = int(worker.draft_token_num)
    gamma = n - 1
    dev = getattr(worker, "device", None) or "cpu"

    def accept_rows(arr):  # rank0 vetoes rows that will not line up, so the accounting says "own" for them
        return [row_ok(arr[i], _prev_tokens(worker, i, n), int(req_drafts[i * n]), gamma) for i in range(len(reqs))]

    plan, _, _ = sc.plan_broadcast(reqs, gamma, torch.device(dev) if isinstance(dev, str) else dev,
                                   accept_rows=accept_rows)
    arr = plan.cpu().numpy()
    if arr[:, 0].any():
        cpu_merge(arr, worker, req_drafts, mask, n)


def wrap_prepare(orig):
    if getattr(orig, "__recall_wrapped__", False):
        return orig

    def _prepare_draft_tokens(self, batch):
        req_drafts, mask = orig(self, batch)
        try:
            if participates(self, batch, req_drafts, mask):
                on_prepared(self, batch, req_drafts, mask)
        except Exception as e:  # noqa: BLE001
            sd._counts["plan_errors"] += 1
            if sd._counts["plan_errors"] <= 3:
                print("[recall] ngram hook error (trie drafts kept): %r" % (e,), flush=True)
        return req_drafts, mask

    _prepare_draft_tokens.__wrapped__ = orig
    _prepare_draft_tokens.__recall_wrapped__ = True
    return _prepare_draft_tokens


def _patch_worker(module) -> None:
    cls = module.NGRAMWorker
    cls._prepare_draft_tokens = wrap_prepare(cls._prepare_draft_tokens)
    print("[recall] patched %s.NGRAMWorker._prepare_draft_tokens" % module.__name__, flush=True)


TARGETS = {WORKER: _patch_worker, PROCESSOR: sd._patch_processor_module}


def install(targets=None) -> None:
    sd.install(TARGETS if targets is None else targets)
