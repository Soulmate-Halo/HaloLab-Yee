"""Pieces every SGLang adapter shares with the DSpark one (sglang_dspark.py owns the engine, the FIFO of planned
rows and the result-processor accounting, so all adapters account the same way).

A plan row is ``[flag, L, W0..W(W-1)]``: rank0 builds it from the scheduler's Req objects (CPU-known length L, the
last known id W0 and the record's continuation after it), one NCCL broadcast gives it to every TP rank, and
``sglang_dspark.device_merge`` swaps the copies in on the GPU wherever the step's anchor lines up.
"""
from __future__ import annotations

import collections
import os

from . import sglang_dspark as sd

MAX_BS = sd.MAX_BS
_warned: set = set()


class _SingleGroup:
    rank_in_group = 0
    world_size = 1

    def broadcast(self, t, src=0):
        return t


def warn_once(key: str, msg: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    print("[recall] %s" % (msg,), flush=True)


def tp_group():
    g = sd._state["group"]
    if g is None:
        try:
            from sglang.srt.distributed import get_tp_group

            g = get_tp_group()
        except Exception:  # noqa: BLE001  tests / single process
            g = _SingleGroup()
        sd._state["group"] = g
    return g


def greedy_batch(batch) -> bool:
    """All-greedy, no grammar: the only batches copies take part in (same on every TP rank)."""
    if batch is None or getattr(batch, "has_grammar", False):
        return False
    si = getattr(batch, "sampling_info", None)
    return si is not None and bool(getattr(si, "is_all_greedy", False))


def nongreedy_enabled(environ=None) -> bool:
    """RECALL_ACCEL_NONGREEDY (default 1): sampled batches take copies too; 0 = all-greedy batches only."""
    env = os.environ if environ is None else environ
    return str(env.get("RECALL_ACCEL_NONGREEDY", "1")).strip().lower() not in ("0", "false", "no", "off")


def copy_batch(batch) -> bool:
    """The batches copies take part in when the head verifies TARGET-ONLY (EAGLE/NEXTN chain, NGRAM, DFLASH
    without its selector: ``tree_speculative_sampling_target_only`` with zero draft probs). A copied id is then a
    point-mass proposal accepted with probability p(x) and the residual is p without x, so any temperature /
    top-k / top-p / penalty keeps the target's distribution. Grammar batches stay off (same on every TP rank)."""
    if batch is None or getattr(batch, "has_grammar", False):
        return False
    si = getattr(batch, "sampling_info", None)
    if si is None:
        return False
    if bool(getattr(si, "is_all_greedy", False)):
        return True
    return nongreedy_enabled()


def capturing() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


_rings: dict = {}  # width -> [pinned (RING, MAX_BS, width) tensor, next slot]; one ring per width so an
#                   adaptive gamma never frees a buffer an in-flight non_blocking copy still reads


def _pinned(width: int):
    import torch

    r = _rings.get(width)
    if r is None:
        r = _rings[width] = [torch.zeros((sd.RING, MAX_BS, width), dtype=torch.int64, pin_memory=True), 0]
    slot = r[1]
    r[1] = (slot + 1) % sd.RING
    return r[0][slot]


def plan_broadcast(reqs, gamma: int, device, accept_rows=None):
    """rank0 plans one row per request, every rank receives it. Returns (plan tensor, rank0, entries).

    ``accept_rows(arr) -> list[bool]`` (rank0, optional) can veto rows before the broadcast; vetoed rows are
    zeroed and recorded as "own" so the accounting stays exact.
    """
    import torch

    bs = len(reqs)
    width = 2 + gamma + sd._pad(gamma)
    group = tp_group()
    rank0 = int(getattr(group, "rank_in_group", 0)) == 0
    plan = torch.zeros((bs, width), dtype=torch.int64, device=device)
    entries = None
    if rank0:
        eng = sd._get_engine(gamma)
        try:
            if eng is not None:
                arr, entries = sd.build_plan(eng, reqs, gamma, paused=sd._check_pause(eng))
                if accept_rows is not None and arr[:, 0].any():
                    keep = accept_rows(arr)
                    for i, k in enumerate(keep):
                        if arr[i, 0] and not k:
                            arr[i, :] = 0
                            e = entries[i]
                            entries[i] = (e[0], 0, e[2], None, 0)
                if arr.shape == (bs, width) and arr[:, 0].any():
                    src = torch.from_numpy(arr)
                    if plan.device.type == "cuda":
                        host = _pinned(width)
                        host[:bs].copy_(src)
                        plan.copy_(host[:bs], non_blocking=True)
                    else:
                        plan.copy_(src)
                    sd._counts["plan_rows"] += int(arr[:, 0].sum())
        except Exception as e:  # noqa: BLE001  never skip the broadcast below
            sd._counts["plan_errors"] += 1
            if sd._counts["plan_errors"] <= 3:
                print("[recall] plan error (zero plan this step): %r" % (e,), flush=True)
            entries = None
            plan.zero_()
        if eng is not None:
            if entries is None:
                entries = [(r.rid, 0, 0, None, 0) for r in reqs]
            for e in entries:
                dq = sd._fifo.get(e[0])
                if dq is None:
                    dq = sd._fifo[e[0]] = collections.deque()
                dq.append(e)
        sd._counts["steps"] += 1
        sd._counts["rows"] += bs
    if int(getattr(group, "world_size", 1)) > 1:
        group.broadcast(plan, src=0)
    return plan, rank0, entries


def enabled() -> bool:
    return sd.enabled()


def env_adapter() -> str:
    return (os.environ.get("RECALL_ACCEL_ADAPTER") or "auto").strip().lower()
