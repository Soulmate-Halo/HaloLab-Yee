"""Yee hot-expert verify budget for the DSV4.1 routed MoE on Mia's ``moe_b12x_next``.

Decode MoE on GB10 is bandwidth bound: a step costs about one read of every distinct expert the step's rows
route to. When armed, every call (one per MoE layer per forward) weighs each expert by the routing weight it
receives from all rows of the step, keeps the smallest hottest set covering ``p`` of that weight (and at most
``kcap`` experts), points every dropped (row, slot) at an expert the row keeps with weight 0, and rescales the
row's kept weights so the row sum is unchanged. The selection follows the live batch on every step, so it adapts
to the traffic by itself; the control file only sets how aggressive it is.

It replaces the ``ep_remap`` launch of the deterministic path (``DSV41_MOE_B12X_NEXT_DETERMINISTIC=1``,
``_Geometry.direct`` False) with one single-CTA Triton kernel that writes the same ``ids_buf``/``w_buf``. Disarmed
(``armed=0``, or ``p>=1`` with ``kcap>=`` distinct experts) the output is bit-identical to ``ep_remap``.

Environment (``YEE_EXPERT_BUDGET=1`` arms the import hook; Yee 4.2 turns it on by default, see defaults.py):
  YEE_EXPERT_BUDGET_E        experts of the geometry to patch (default 384; the 128-expert draft is untouched)
  YEE_EXPERT_BUDGET_MAXM     rows above this take the stock forward (default 512)
  YEE_EXPERT_BUDGET_CTRL     control file (default /state/yee_expert_budget.json, ~/.cache/yee/ without /state)
  YEE_EXPERT_BUDGET_STATS    stats jsonl, appended by global rank 0 (default /state/yee_expert_budget_stats.jsonl)
  YEE_EXPERT_BUDGET_STATS_S  stats interval in seconds (default 10)
  YEE_EXPERT_BUDGET_P / _KCAP / _ARMED  initial control (module default 1.0 / 4096 / 0; the 4.2 startup defaults set 0.9 / 4096 / 1)
  YEE_RANK_CHECK_EB_S        seconds to wait for every TP rank to report the same Yee before arming (default 300)
Control file: {"seq": int, "armed": 0|1, "p": (0, 1], "kcap": int >= 0, "at_calls": int (optional)}; read when
``seq`` changes. With ``at_calls`` the new values sit in a pending slot and the kernel itself switches at that call
index; every TP rank makes the same sequence of calls, so all ranks switch on the same layer of the same step and
this is safe under live traffic. ``at_calls`` = rank 0's latest ``calls`` (stats file) + a margin of a few hundred
forwards (40 calls each). Without ``at_calls`` the value is written at once, which is only safe while idle (each
rank polls its own node's file at a different moment).
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time

_TAG = "[yee-eb]"
_TRUE = ("1", "true", "on", "yes")
QSCALE = 4096          # routing weight quantum (weights clamped to [0, 16]: M<=512 keeps every sum in int32)
WMAX = 16.0
NBUCKET = 17           # stats rows: bucket = min(ceil(M / 8), 16)
NSTAT = 8              # calls, rows, uniq_before, uniq_after, mass_total, mass_kept, forced_rows, armed_calls
NIT = 28               # binary-search iterations (mass < 2**25)
BIG = 2147483647


def _env_int(name, default):
    try:
        return int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return int(default)


def _env_float(name, default):
    try:
        return float(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return float(default)


TARGET_E = _env_int("YEE_EXPERT_BUDGET_E", 384)
MAXM = _env_int("YEE_EXPERT_BUDGET_MAXM", 512)
def _default_path(name):
    """/state/<name> when /state is a writable directory (Mia containers), else ~/.cache/yee/<name>."""
    if os.path.isdir("/state") and os.access("/state", os.W_OK):
        return os.path.join("/state", name)
    return os.path.join(os.path.expanduser("~"), ".cache", "yee", name)


CTRL_PATH = os.environ.get("YEE_EXPERT_BUDGET_CTRL") or _default_path("yee_expert_budget.json")
STATS_PATH = os.environ.get("YEE_EXPERT_BUDGET_STATS") or _default_path("yee_expert_budget_stats.jsonl")
STATS_S = max(1.0, _env_float("YEE_EXPERT_BUDGET_STATS_S", 10.0))

_STATE = {"installed": False, "patched": False, "thread": None, "warned": set(), "seq": None, "pending": None,
          "ctrl": (_env_float("YEE_EXPERT_BUDGET_P", 1.0), _env_int("YEE_EXPERT_BUDGET_KCAP", 4096),
                   1 if _env_int("YEE_EXPERT_BUDGET_ARMED", 0) else 0)}
_REG = []              # armed geometries
_KERNEL = [None]
_LOCK = threading.Lock()


def _np2(n):
    n = int(n)
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _print(msg):
    print("%s %s" % (_TAG, msg), flush=True)


def _warn_once(key, exc):
    if key not in _STATE["warned"]:
        _STATE["warned"].add(key)
        _print("%s error (further ones muted): %r" % (key, exc))


# --------------------------------------------------------------------------------------------
# control
# --------------------------------------------------------------------------------------------
def parse_ctrl(d):
    """Validate a control dict; returns (seq, armed, p, kcap, at_calls) or None (at_calls None = immediate)."""
    if not isinstance(d, dict):
        return None
    seq, armed, p, kcap = d.get("seq"), d.get("armed", 0), d.get("p", 1.0), d.get("kcap", 4096)
    at = d.get("at_calls")
    if isinstance(seq, bool) or not isinstance(seq, int):
        return None
    if isinstance(armed, bool):
        armed = int(armed)
    if armed not in (0, 1):
        return None
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not (0.0 < float(p) <= 1.0):
        return None
    if isinstance(kcap, bool) or not isinstance(kcap, int) or not (0 <= kcap <= 1 << 20):
        return None
    if at is not None and (isinstance(at, bool) or not isinstance(at, int) or not (0 <= at < NEVER)):
        return None
    return seq, armed, float(p), int(kcap), at


# --------------------------------------------------------------------------------------------
# reference semantics (torch; the kernel must match it)
# --------------------------------------------------------------------------------------------
def _thresholds(mass, total, nsel, p32, kcap, armed):
    """Integer binary searches of the kernel, in Python (mass: list of int)."""
    import numpy as np

    if not armed:
        return 1
    mmax = max(mass) if mass else 0
    need = np.float32(p32) * np.float32(total)
    lo, hi = 1, max(mmax, 1)
    for _ in range(NIT):
        if lo >= hi:
            break
        mid = (lo + hi + 1) // 2
        s = sum(v for v in mass if v >= mid)
        if np.float32(s) >= need:
            lo = mid
        else:
            hi = mid - 1
    t_cov = 1 if (np.float32(p32) >= np.float32(1.0) or total == 0) else lo
    lo2, hi2 = 1, mmax + 1
    for _ in range(NIT):
        if lo2 >= hi2:
            break
        mid = (lo2 + hi2) // 2
        c = sum(1 for v in mass if v >= mid)
        if c <= kcap:
            hi2 = mid
        else:
            lo2 = mid + 1
    t_cap = 1 if kcap >= nsel else lo2
    return max(t_cov, t_cap)


def budget_reference(ids, w, E, p, kcap, armed):
    """Returns (out_ids int32 [M,K], out_w float32 [M,K], info). ``p`` is taken as float32, like the kernel."""
    import numpy as np
    import torch

    dev = ids.device
    ids64 = ids.to(torch.int64)
    wf = w.to(torch.float32)
    M, K = ids64.shape
    ok = (ids64 >= 0) & (ids64 < E)
    wq = torch.floor(wf.clamp(0.0, WMAX) * QSCALE + 0.5).to(torch.int64).clamp_min(1)
    wq = torch.where(ok, wq, torch.zeros_like(wq))
    mass_t = torch.zeros(E, dtype=torch.int64, device=dev)
    mass_t.index_add_(0, torch.where(ok, ids64, torch.zeros_like(ids64)).reshape(-1), wq.reshape(-1))
    mass = [int(v) for v in mass_t.cpu().tolist()]
    total = sum(mass)
    nsel = sum(1 for v in mass if v > 0)
    p32 = float(np.float32(p))
    t = _thresholds(mass, total, nsel, p32, int(kcap), bool(armed))
    keep_e = (mass_t >= t) & (mass_t > 0)
    kidx = torch.nonzero(keep_e).flatten()
    aidx = torch.nonzero(mass_t > 0).flatten()
    gmin = int(kidx[0]) if kidx.numel() else (int(aidx[0]) if aidx.numel() else 0)
    safe = torch.where(ok, ids64, torch.zeros_like(ids64))
    kept = ok & keep_e[safe]
    row_valid = ok.any(dim=1)
    row_kept = kept.any(dim=1)
    forced = row_valid & ~row_kept
    if bool(forced.any()):
        wv = torch.where(ok, wf, torch.full_like(wf, -1.0))
        wmax = wv.max(dim=1, keepdim=True).values
        cols = torch.arange(K, device=dev).expand(M, K)
        ismax = ok & (wv == wmax)
        fcol = torch.where(ismax, cols, torch.full_like(cols, K)).min(dim=1).values
        kept = kept | (forced[:, None] & (cols == fcol[:, None]))
    rmin = torch.where(kept, ids64, torch.full_like(ids64, BIG)).min(dim=1).values
    fill = torch.where(rmin < E, rmin, torch.full_like(rmin, gmin))
    zero = torch.zeros_like(wf)
    wsum = torch.where(ok, wf, zero).sum(dim=1)
    ksum = torch.where(kept, wf, zero).sum(dim=1)
    one = torch.ones_like(ksum)
    scale = torch.where(ksum == wsum, one, wsum / torch.where(ksum > 0, ksum, one))
    scale = torch.where(ksum > 0, scale, one)
    out_ids = torch.where(kept, ids64, fill[:, None].expand(M, K)).to(torch.int32)
    out_w = torch.where(kept, wf * scale[:, None], zero)
    info = {"calls": 1, "rows": int(M), "uniq_before": nsel, "uniq_after": int(torch.unique(out_ids).numel()),
            "mass_total": total, "mass_kept": int(mass_t[keep_e].sum()), "forced_rows": int(forced.sum()),
            "armed": int(bool(armed)), "t": int(t)}
    return out_ids, out_w, info


def ep_remap_reference(ids, w, E):
    """Production ``moe_b12x_next.ep_remap_reference`` with offset 0, num_local E."""
    import torch

    lid = ids.to(torch.int64)
    ok = (lid >= 0) & (lid < E)
    big = torch.full_like(lid, BIG)
    best = torch.where(ok, lid, big).min()
    fb = best if int(best) < E else torch.zeros_like(best)
    row_min = torch.where(ok, lid, big).min(dim=1).values
    fill = torch.where(row_min < E, row_min, fb)
    out_ids = torch.where(ok, lid, fill[:, None].expand_as(lid)).to(torch.int32)
    out_w = torch.where(ok, w.float(), torch.zeros_like(w, dtype=torch.float32))
    return out_ids, out_w


def bucket_of(m):
    return min((int(m) + 7) // 8, NBUCKET - 1)


# --------------------------------------------------------------------------------------------
# Triton kernel
# --------------------------------------------------------------------------------------------
def _kernel():
    if _KERNEL[0] is None:
        import triton
        import triton.language as tl

        @triton.jit(do_not_specialize=["M", "SI0", "SI1", "SW0", "SW1"])
        def kernel(IDS, W, OUT_IDS, OUT_W, CTRL, SCHED, STATS, SCR, M, SI0, SI1, SW0, SW1,
                   K: tl.constexpr, KP: tl.constexpr, E: tl.constexpr, EP: tl.constexpr, BR: tl.constexpr,
                   QS: tl.constexpr, WCLAMP: tl.constexpr, NIT: tl.constexpr, NB: tl.constexpr):
            offs_e = tl.arange(0, EP)
            emask = offs_e < E
            cols = tl.arange(0, KP)
            cmask = cols < K
            # rank-synchronized switch: every TP rank runs the same sequence of calls, so switching at
            # the same call index keeps the ranks on the same expert set. CTRL = [cur p,kcap,armed,_, new ...]
            calls = tl.load(SCHED)
            at = tl.load(SCHED + 1)
            sw = calls >= at
            p = tl.where(sw, tl.load(CTRL + 4), tl.load(CTRL))
            kcapf = tl.where(sw, tl.load(CTRL + 5), tl.load(CTRL + 1))
            armed = tl.where(sw, tl.load(CTRL + 6), tl.load(CTRL + 2))
            tl.store(CTRL, p, mask=sw)
            tl.store(CTRL + 1, kcapf, mask=sw)
            tl.store(CTRL + 2, armed, mask=sw)
            tl.store(SCHED + 1, tl.full((), 4611686018427387904, tl.int64), mask=sw)
            tl.store(SCHED, calls + 1)
            # phase 0: clear scratch (mass | keep | histogram of the output ids)
            zeros_e = tl.zeros([EP], dtype=tl.int32)
            tl.store(SCR + offs_e, zeros_e)
            tl.store(SCR + EP + offs_e, zeros_e)
            tl.store(SCR + 2 * EP + offs_e, zeros_e)
            tl.debug_barrier()
            # phase 1: per-expert routing mass
            for r0 in range(0, M, BR):
                rows = r0 + tl.arange(0, BR)
                m2 = (rows[:, None] < M) & cmask[None, :]
                lid = tl.load(IDS + rows[:, None] * SI0 + cols[None, :] * SI1, mask=m2, other=-1).to(tl.int32)
                w = tl.load(W + rows[:, None] * SW0 + cols[None, :] * SW1, mask=m2, other=0.0).to(tl.float32)
                ok = m2 & (lid >= 0) & (lid < E)
                wc = tl.minimum(tl.maximum(w, 0.0), WCLAMP)
                wq = tl.maximum((wc * QS + 0.5).to(tl.int32), 1)
                tl.atomic_add(SCR + tl.where(ok, lid, 0), wq, mask=ok)
            tl.debug_barrier()
            # phase 2: thresholds
            mass = tl.load(SCR + offs_e, mask=emask, other=0, volatile=True)
            total = tl.sum(mass, axis=0)
            nsel = tl.sum((mass > 0).to(tl.int32), axis=0)
            mmax = tl.max(mass, axis=0)
            kcap = kcapf.to(tl.int32)
            need = p * total.to(tl.float32)
            lo = tl.full((), 1, tl.int32)
            hi = tl.maximum(mmax, 1)
            for _ in range(NIT):
                mid = (lo + hi + 1) // 2
                s = tl.sum(tl.where(mass >= mid, mass, 0), axis=0)
                sf = s.to(tl.float32)
                act = lo < hi
                lo = tl.where(act & (sf >= need), mid, lo)
                hi = tl.where(act & (sf < need), mid - 1, hi)
            t_cov = tl.where((p >= 1.0) | (total == 0), 1, lo)
            lo2 = tl.full((), 1, tl.int32)
            hi2 = mmax + 1
            for _ in range(NIT):
                mid2 = (lo2 + hi2) // 2
                c = tl.sum((mass >= mid2).to(tl.int32), axis=0)
                act2 = lo2 < hi2
                hi2 = tl.where(act2 & (c <= kcap), mid2, hi2)
                lo2 = tl.where(act2 & (c > kcap), mid2 + 1, lo2)
            t_cap = tl.where(kcap >= nsel, 1, lo2)
            t = tl.where(armed > 0.5, tl.maximum(t_cov, t_cap), 1)
            keep_e = (mass >= t) & (mass > 0) & emask
            tl.store(SCR + EP + offs_e, keep_e.to(tl.int32), mask=emask)
            gk = tl.min(tl.where(keep_e, offs_e, 2147483647), axis=0)
            ga = tl.min(tl.where((mass > 0) & emask, offs_e, 2147483647), axis=0)
            gmin = tl.where(gk < E, gk, tl.where(ga < E, ga, 0))
            mass_kept = tl.sum(tl.where(keep_e, mass, 0), axis=0)
            tl.debug_barrier()
            # phase 3: outputs
            forced_n = tl.full((), 0, tl.int32)
            for r0 in range(0, M, BR):
                rows = r0 + tl.arange(0, BR)
                m2 = (rows[:, None] < M) & cmask[None, :]
                lid = tl.load(IDS + rows[:, None] * SI0 + cols[None, :] * SI1, mask=m2, other=-1).to(tl.int32)
                w = tl.load(W + rows[:, None] * SW0 + cols[None, :] * SW1, mask=m2, other=0.0).to(tl.float32)
                ok = m2 & (lid >= 0) & (lid < E)
                kp = tl.load(SCR + EP + tl.where(ok, lid, 0), mask=ok, other=0, volatile=True)
                kept = ok & (kp > 0)
                row_valid = tl.max(ok.to(tl.int32), axis=1)
                row_kept = tl.max(kept.to(tl.int32), axis=1)
                forced = (row_valid > 0) & (row_kept < 1)
                wv = tl.where(ok, w, -1.0)
                wmx = tl.max(wv, axis=1)
                ismax = ok & (wv == wmx[:, None])
                fcol = tl.min(tl.where(ismax, cols[None, :], KP), axis=1)
                kept = kept | (forced[:, None] & (cols[None, :] == fcol[:, None]))
                forced_n += tl.sum(forced.to(tl.int32), axis=0)
                rmin = tl.min(tl.where(kept, lid, 2147483647), axis=1)
                fill = tl.where(rmin < E, rmin, gmin)
                wsum = tl.sum(tl.where(ok, w, 0.0), axis=1)
                ksum = tl.sum(tl.where(kept, w, 0.0), axis=1)
                scale = tl.where(ksum == wsum, 1.0, wsum / tl.where(ksum > 0.0, ksum, 1.0))
                scale = tl.where(ksum > 0.0, scale, 1.0)
                oid = tl.where(kept, lid, fill[:, None])
                ow = tl.where(kept, w * scale[:, None], 0.0)
                offs = rows[:, None] * K + cols[None, :]
                tl.store(OUT_IDS + offs, oid, mask=m2)
                tl.store(OUT_W + offs, ow, mask=m2)
                tl.atomic_add(SCR + 2 * EP + oid, tl.full([BR, KP], 1, tl.int32), mask=m2)
            tl.debug_barrier()
            # phase 4: stats row
            hist = tl.load(SCR + 2 * EP + offs_e, mask=emask, other=0, volatile=True)
            after = tl.sum((hist > 0).to(tl.int32), axis=0)
            bucket = tl.minimum((M + 7) // 8, NB - 1)
            o8 = tl.arange(0, 8)
            armed_i = (armed > 0.5).to(tl.int64)
            vals = tl.where(o8 == 0, 1,
                   tl.where(o8 == 1, M,
                   tl.where(o8 == 2, nsel,
                   tl.where(o8 == 3, after,
                   tl.where(o8 == 4, total,
                   tl.where(o8 == 5, mass_kept,
                   tl.where(o8 == 6, forced_n, 0))))))).to(tl.int64)
            vals = tl.where(o8 == 7, armed_i, vals)
            tl.atomic_add(STATS + bucket * 8 + o8, vals)

        _KERNEL[0] = kernel
    return _KERNEL[0]


NEVER = 1 << 62


def new_ctrl(p, kcap, armed, device):
    """[cur p, kcap, armed, 0, new p, kcap, armed, 0] (float32) and sched [calls, at] (int64, at=NEVER: none)."""
    import torch

    v = [float(p), float(kcap), float(armed), 0.0]
    return (torch.tensor(v + v, dtype=torch.float32, device=device),
            torch.tensor([0, NEVER], dtype=torch.int64, device=device))


def budget_triton(ids, w, out_ids, out_w, ctrl, sched, stats, scratch, E):
    """One launch, no host sync, graph-capturable. ids/w: any strided [M, K]; out_*: contiguous [>=M, K]."""
    m, k = ids.shape
    if m == 0:
        return
    if tuple(w.shape) != (m, k) or not (out_ids.is_contiguous() and out_w.is_contiguous()) \
            or out_ids.shape[0] < m or out_w.shape[0] < m or scratch.numel() < 3 * _np2(E) \
            or tuple(stats.shape) != (NBUCKET, NSTAT) or ctrl.numel() != 8 or sched.numel() != 2:
        raise RuntimeError("%s budget_triton shapes ids %s w %s out %s/%s" % (
            _TAG, tuple(ids.shape), tuple(w.shape), tuple(out_ids.shape), tuple(out_w.shape)))
    _kernel()[(1,)](ids, w, out_ids, out_w, ctrl, sched, stats, scratch, m,
                    ids.stride(0), ids.stride(1), w.stride(0), w.stride(1),
                    K=k, KP=_np2(k), E=int(E), EP=_np2(E), BR=32, QS=QSCALE, WCLAMP=WMAX, NIT=NIT, NB=NBUCKET,
                    num_warps=4)


# --------------------------------------------------------------------------------------------
# moe_b12x_next patch
# --------------------------------------------------------------------------------------------
def _arm(geom):
    import torch

    dev = geom.device
    p, kcap, armed = _STATE["ctrl"]
    with torch.cuda.device(dev):
        ctrl, sched = new_ctrl(p, kcap, armed, dev)
        stats = torch.zeros(NBUCKET, NSTAT, dtype=torch.int64, device=dev)
        scratch = torch.zeros(3 * _np2(geom.E), dtype=torch.int32, device=dev)
        ids = torch.zeros(8, int(geom.topk), dtype=torch.int32, device=dev)
        w = torch.full((8, int(geom.topk)), 0.1, dtype=torch.float32, device=dev)
        oi, ow = torch.empty_like(ids), torch.empty_like(w)
        c0, s0 = new_ctrl(1.0, 4096, 0, dev)    # identity check below only holds unarmed (initial ctrl may be armed)
        budget_triton(ids, w, oi, ow, c0, s0, stats, scratch, geom.E)    # JIT now, not inside graph capture
        torch.cuda.synchronize(dev)
        if not bool((oi == 0).all()) or not bool((ow == w).all()):
            raise RuntimeError("warm-up output differs from ep_remap")
        bad = _quick_check(dev, int(geom.E), int(geom.topk))
        if bad:
            raise RuntimeError("self-check vs reference failed: " + bad)
        stats.zero_()
    geom._yb = {"ctrl": ctrl, "sched": sched, "stats": stats, "scratch": scratch}
    with _LOCK:
        _REG.append(geom)
    _start_thread()
    _print("armed geometry E=%d topk=%d device=%s ctrl=p:%.4f kcap:%d armed:%d"
           % (geom.E, geom.topk, dev, p, kcap, armed))


_RANK = {"ok": None}


def _ranks_agree():
    """4.2: arm only when every TP rank of this group runs the same Yee (store-based, no collective), so all ranks
    pick the same experts. Checked once per process, at the first target geometry (every rank builds it)."""
    ok = _RANK["ok"]
    if ok is not None:
        return ok
    try:
        try:
            from . import rank_agree
        except ImportError:
            from yee.integrations import rank_agree
        ranks, me = rank_agree.tp_ranks()
        if ranks is None:
            ok, why = True, "not distributed"
        else:
            ok, why = rank_agree.agree("expert_budget", ranks, me, _env_float("YEE_RANK_CHECK_EB_S", 300.0))
    except Exception as exc:  # noqa: BLE001
        ok, why = False, "rank check error: %r" % (exc,)
    _RANK["ok"] = ok
    _print("rank check %s: %s" % ("on" if ok else "off", why))
    return ok


def _patch_module(mod):
    G = getattr(mod, "_Geometry", None)
    if G is None or not hasattr(G, "forward") or not hasattr(G, "_run"):
        _print("moe_b12x_next has no _Geometry.forward/_run; not patched")
        return
    if getattr(G, "_yee_eb_patched", False):
        return
    orig_init, orig_fwd = G.__init__, G.forward

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        self._yb = None
        try:
            if int(getattr(self, "E", -1)) == TARGET_E and _ranks_agree():
                _arm(self)
        except Exception as exc:  # noqa: BLE001
            self._yb = None
            _print("geometry disabled: %r" % (exc,))

    def forward(self, impl, x, topk_ids, topk_weights, out):
        yb = getattr(self, "_yb", None)
        if yb is None:
            return orig_fwd(self, impl, x, topk_ids, topk_weights, out)
        m = int(x.shape[0])
        ib = self.ids_buf
        if (self.direct or m == 0 or m > MAXM or ib is None or m > ib.shape[0] or m > self.chunk
                or topk_ids.dim() != 2 or int(topk_ids.shape[1]) != int(self.topk)
                or tuple(topk_weights.shape) != tuple(topk_ids.shape)):
            return orig_fwd(self, impl, x, topk_ids, topk_weights, out)
        ids, w = ib[:m], self.w_buf[:m]
        budget_triton(topk_ids, topk_weights, ids, w, yb["ctrl"], yb["sched"], yb["stats"], yb["scratch"], self.E)
        self._run(self.cap_for(m), x, ids, w, out, impl)
        return out

    G.__init__ = __init__
    G.forward = forward
    G._yee_eb_patched = True
    _STATE["patched"] = True
    _print("patched moe_b12x_next._Geometry (target E=%d)" % TARGET_E)


class _Loader(importlib.abc.Loader):
    def __init__(self, orig):
        self._orig = orig

    def create_module(self, spec):
        return self._orig.create_module(spec)

    def exec_module(self, module):
        self._orig.exec_module(module)
        try:
            _patch_module(module)
        except Exception as exc:  # noqa: BLE001
            _print("patch failed: %r" % (exc,))


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "moe_b12x_next" or _STATE["patched"]:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


def install():
    """Idempotent; never raises."""
    try:
        if _STATE["installed"]:
            return True
        _STATE["installed"] = True
        mod = sys.modules.get("moe_b12x_next")
        if mod is not None:
            _patch_module(mod)
        else:
            sys.meta_path.insert(0, _Finder())
        _print("installed E=%d maxm=%d ctrl=%s stats=%s" % (TARGET_E, MAXM, CTRL_PATH, STATS_PATH))
        return True
    except Exception as exc:  # noqa: BLE001
        _print("install failed: %r" % (exc,))
        return False


# --------------------------------------------------------------------------------------------
# control / stats thread
# --------------------------------------------------------------------------------------------
def _rank():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        pass
    return None


def _apply_ctrl(seq, armed, p, kcap, at_calls=None):
    """at_calls None: write the current slot now (only safe while idle: ranks poll at different moments).
    at_calls given: write the 'new' slot, then the switch index; the kernel flips at call ``at_calls`` on every rank."""
    import torch

    with _LOCK:
        geoms = list(_REG)
    v = torch.tensor([p, float(kcap), float(armed), 0.0], dtype=torch.float32)
    for g in geoms:
        with torch.cuda.device(g.device):
            if at_calls is None:
                g._yb["sched"][1:2].fill_(NEVER)
                g._yb["ctrl"][0:4].copy_(v)
                g._yb["ctrl"][4:8].copy_(v)
            else:
                g._yb["ctrl"][4:8].copy_(v)
                g._yb["sched"][1:2].copy_(torch.tensor([int(at_calls)], dtype=torch.int64))
    _STATE["seq"] = seq
    if at_calls is None:
        _STATE["ctrl"] = (p, kcap, armed)
        _print("ctrl seq=%d armed=%d p=%.4f kcap=%d applied rank=%s geoms=%d (immediate)"
               % (seq, armed, p, kcap, _rank(), len(geoms)))
    else:
        _STATE["pending"] = (seq, p, kcap, armed, int(at_calls))
        _print("ctrl seq=%d armed=%d p=%.4f kcap=%d scheduled at_calls=%d rank=%s geoms=%d"
               % (seq, armed, p, kcap, int(at_calls), _rank(), len(geoms)))


def _device_state():
    """(cur p, kcap, armed, calls, pending_at) of the first armed geometry, read back from the device."""
    with _LOCK:
        geoms = list(_REG)
    if not geoms:
        return None
    c = geoms[0]._yb["ctrl"].cpu().tolist()
    s = geoms[0]._yb["sched"].cpu().tolist()
    return float(c[0]), int(c[1]), int(c[2]), int(s[0]), (None if int(s[1]) >= NEVER else int(s[1]))


def _watch_pending():
    pend = _STATE.get("pending")
    if not pend:
        return
    d = _device_state()
    if d is None or d[4] is not None:
        return
    seq, p, kcap, armed, at = pend
    _STATE["pending"] = None
    _STATE["ctrl"] = (d[0], d[1], d[2])
    _print("ctrl seq=%d active armed=%d p=%.4f kcap=%d at_calls=%d calls=%d rank=%s"
           % (seq, d[2], d[0], d[1], at, d[3], _rank()))


def _poll_ctrl(last):
    try:
        st = os.stat(CTRL_PATH)
    except OSError:
        return last
    key = (st.st_mtime_ns, st.st_size)
    if key == last:
        return last
    try:
        with open(CTRL_PATH, encoding="utf-8") as fh:
            parsed = parse_ctrl(json.load(fh))
    except Exception as exc:  # noqa: BLE001
        _warn_once("ctrl-read", exc)
        return key
    if parsed is None:
        _print("ctrl file %s rejected (bad fields)" % CTRL_PATH)
        return key
    seq, armed, p, kcap, at_calls = parsed
    if seq != _STATE["seq"]:
        _apply_ctrl(seq, armed, p, kcap, at_calls)
    return key


def _dump_stats():
    if _rank() != 0:
        return
    with _LOCK:
        geoms = list(_REG)
    if not geoms:
        return
    tot = None
    for g in geoms:
        s = g._yb["stats"].cpu()
        tot = s if tot is None else tot + s
    d = _device_state()
    p, kcap, armed = (d[0], d[1], d[2]) if d else _STATE["ctrl"]
    line = {"t": round(time.time(), 3), "seq": _STATE["seq"], "armed": armed, "p": p, "kcap": kcap,
            "calls": d[3] if d else None, "pending_at": d[4] if d else None, "stats": tot.tolist()}
    with open(STATS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


def _thread_main():
    last, next_dump = None, time.time() + STATS_S
    while True:
        try:
            last = _poll_ctrl(last)
            _watch_pending()
        except Exception as exc:  # noqa: BLE001
            _warn_once("ctrl", exc)
        if time.time() >= next_dump:
            next_dump = time.time() + STATS_S
            try:
                _dump_stats()
            except Exception as exc:  # noqa: BLE001
                _warn_once("stats", exc)
        time.sleep(1.0)


def _start_thread():
    with _LOCK:
        if _STATE["thread"] is not None:
            return
        t = threading.Thread(target=_thread_main, name="yee-eb", daemon=True)
        _STATE["thread"] = t
    t.start()


# --------------------------------------------------------------------------------------------
# GPU self-test (run on the serving node: python expert_budget.py --selftest)
# --------------------------------------------------------------------------------------------
def _case(gen, dev, M, K, E, kind):
    import torch

    scores = torch.rand(M, E, generator=gen, device="cpu")
    if kind == "hot":
        hot = torch.randperm(E, generator=gen)[:30]
        scores[:, hot] += 1.0
    elif kind == "corr":
        base = torch.rand(1, E, generator=gen)
        scores = 0.7 * base + 0.3 * scores
    ids = scores.topk(K, dim=1).indices.to(torch.int32)
    w = torch.rand(M, K, generator=gen) + 0.05
    w = w / w.sum(dim=1, keepdim=True) * 1.5
    if kind in ("invalid", "hot") and M > 2:
        mask = torch.rand(M, K, generator=gen) < 0.1
        ids[mask] = -1
        ids[0, :] = -1
        if M > 3:
            ids[2, 1] = E + 7
    return ids.to(dev), w.to(dev)


def _check_case(ids, w, E, p, kcap, armed, stats, scratch):
    """Kernel vs reference for one case; returns an error string or None."""
    import torch

    M, K = ids.shape
    dev = ids.device
    ctrl, sched = new_ctrl(p, kcap, armed, dev)
    stats.zero_()
    scratch.fill_(777)
    oi = torch.full((M + 3, K), -9, dtype=torch.int32, device=dev)
    ow = torch.full((M + 3, K), -9.0, dtype=torch.float32, device=dev)
    budget_triton(ids, w, oi, ow, ctrl, sched, stats, scratch, E)
    torch.cuda.synchronize(dev)
    ri, rw, info = budget_reference(ids, w, E, p, kcap, armed)
    ki, kw = oi[:M], ow[:M]
    if not torch.equal(ki, ri):
        return "ids differ at %s (t=%s)" % (torch.nonzero(ki != ri)[:4].tolist(), info.get("t"))
    if bool((oi[M:] != -9).any()) or bool((ow[M:] != -9.0).any()):
        return "wrote past M"
    lossless = (not armed) or (info["mass_kept"] == info["mass_total"] and info["forced_rows"] == 0)
    if lossless:
        ei, ew = ep_remap_reference(ids, w, E)
        if not (torch.equal(kw, rw) and torch.equal(ki, ei) and torch.equal(kw, ew)):
            return "lossless case not bit-identical (max |dw|=%g)" % float((kw - ew).abs().max())
    else:
        rel = ((kw - rw).abs() / rw.abs().clamp_min(1e-6)).max()
        if float(rel) > 2e-6:
            return "weights rel err %g" % float(rel)
    want = [info["calls"], info["rows"], info["uniq_before"], info["uniq_after"], info["mass_total"],
            info["mass_kept"], info["forced_rows"], info["armed"]]
    got = stats[bucket_of(M)].tolist()
    if got != want or int(stats.sum()) != sum(want):
        return "stats %s != %s" % (got, want)
    if sched.cpu().tolist() != [1, NEVER]:
        return "sched %s" % sched.cpu().tolist()
    return None


def _check_switch(dev, E, K, gen):
    """Scheduled switch: params A for calls < at, B from call ``at`` on; slot and sched updated by the kernel."""
    import torch

    ids, w = _case(gen, dev, 48, K, E, "corr")
    stats = torch.zeros(NBUCKET, NSTAT, dtype=torch.int64, device=dev)
    scratch = torch.zeros(3 * _np2(E), dtype=torch.int32, device=dev)
    A, B = (1.0, 4096, 0), (0.8, 4096, 1)
    ctrl, sched = new_ctrl(*A, dev)
    # explicit float32: model construction runs under torch.set_default_dtype(bfloat16), 0.8 would become 0.80078125
    ctrl[4:8].copy_(torch.tensor([B[0], float(B[1]), float(B[2]), 0.0], dtype=torch.float32))
    sched[1:2].fill_(3)
    ra = budget_reference(ids, w, E, *A)
    rb = budget_reference(ids, w, E, *B)
    for call in range(6):
        oi = torch.empty(48, K, dtype=torch.int32, device=dev)
        ow = torch.empty(48, K, dtype=torch.float32, device=dev)
        budget_triton(ids, w, oi, ow, ctrl, sched, stats, scratch, E)
        torch.cuda.synchronize(dev)
        ref = ra if call < 3 else rb
        if not torch.equal(oi, ref[0]) or float(((ow - ref[1]).abs() / ref[1].abs().clamp_min(1e-6)).max()) > 2e-6:
            return "switch call %d used the wrong params" % call
    c, s = ctrl.cpu().tolist(), sched.cpu().tolist()
    if [round(c[0], 4), int(c[1]), int(c[2])] != [B[0], B[1], B[2]] or s != [6, NEVER]:
        return "switch state ctrl=%s sched=%s" % (c, s)
    return None


def _quick_check(dev, E, K, seed=4100):
    """A few kernel-vs-reference cases on the serving device at load time (eager, before graph capture)."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    stats = torch.zeros(NBUCKET, NSTAT, dtype=torch.int64, device=dev)
    scratch = torch.zeros(3 * _np2(E), dtype=torch.int32, device=dev)
    combos = [(1.0, 4096, 0), (0.9, 4096, 1), (0.8, 4096, 1), (1.0, 16, 1), (0.95, 24, 1), (0.6, 4, 1)]
    for i, M in enumerate([1, 8, 16, 40, 64, 128, 300, 8, 56, 96, 128, 24]):
        kind = ["uniform", "hot", "corr", "invalid"][i % 4]
        ids, w = _case(gen, dev, M, K, E, kind)
        p, kcap, armed = combos[i % len(combos)]
        bad = _check_case(ids, w, E, p, kcap, armed, stats, scratch)
        if bad:
            return "case %d M=%d %s p=%s kcap=%s armed=%s: %s" % (i, M, kind, p, kcap, armed, bad)
    return _check_switch(dev, E, K, gen)


def selftest(n=240, seed=715):
    import torch

    if not torch.cuda.is_available():
        print("SELFTEST SKIP no cuda")
        return 2
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(seed)
    E, K = 384, 6
    stats = torch.zeros(NBUCKET, NSTAT, dtype=torch.int64, device=dev)
    scratch = torch.full((3 * _np2(E),), 12345, dtype=torch.int32, device=dev)
    ps = [1.0, 0.97, 0.9, 0.8, 0.6]
    kcaps = [4096, 64, 24, 12, 4, 0]
    kinds = ["uniform", "hot", "corr", "invalid"]
    for i in range(n):
        M = int(torch.randint(1, 129, (1,), generator=gen)) if i % 4 else int(torch.randint(129, 513, (1,), generator=gen))
        kind = kinds[i % len(kinds)]
        p, kcap, armed = ps[i % len(ps)], kcaps[(i // 5) % len(kcaps)], (i // 3) % 2
        ids, w = _case(gen, dev, M, K, E, kind)
        if i % 7 == 3:          # strided views
            ids = torch.cat([ids, torch.zeros(M, 2, dtype=torch.int32, device=dev)], dim=1)[:, :K]
            w = torch.cat([w, torch.zeros(M, 2, dtype=torch.float32, device=dev)], dim=1)[:, :K]
        bad = _check_case(ids, w, E, p, kcap, armed, stats, scratch)
        if bad:
            print("SELFTEST FAIL case=%d M=%d kind=%s p=%s kcap=%s armed=%s: %s" % (i, M, kind, p, kcap, armed, bad))
            return 1
    bad = _check_switch(dev, E, K, gen)
    if bad:
        print("SELFTEST FAIL " + bad)
        return 1
    # CUDA graph: capture once; immediate slot writes and a scheduled switch both reach the replays
    M = 64
    ids, w = _case(gen, dev, M, K, E, "corr")
    oi = torch.empty(M, K, dtype=torch.int32, device=dev)
    ow = torch.empty(M, K, dtype=torch.float32, device=dev)
    ctrl, sched = new_ctrl(1.0, 4096, 0, dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        budget_triton(ids, w, oi, ow, ctrl, sched, stats, scratch, E)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        budget_triton(ids, w, oi, ow, ctrl, sched, stats, scratch, E)
    plan = [((1.0, 4096, 0), None), ((0.9, 4096, 1), None), ((0.8, 16, 1), None), ((0.95, 4096, 1), "sched"),
            ((1.0, 4096, 1), None)]
    for (p, kcap, armed), mode in plan:
        v = torch.tensor([p, float(kcap), float(armed), 0.0], dtype=torch.float32)
        if mode == "sched":
            ctrl[4:8].copy_(v)
            sched[1:2].copy_(sched[0:1] + 1)        # flips on the second replay from now
            g.replay()
            torch.cuda.synchronize()
            prev = budget_reference(ids, w, E, 0.8, 16, 1)
            if not torch.equal(oi, prev[0]):
                print("SELFTEST FAIL graph: scheduled switch fired one call early")
                return 1
        else:
            ctrl[0:4].copy_(v)
            ctrl[4:8].copy_(v)
        g.replay()
        torch.cuda.synchronize()
        ri, rw, info = budget_reference(ids, w, E, p, kcap, armed)
        if not torch.equal(oi, ri) or float(((ow - rw).abs() / rw.abs().clamp_min(1e-6)).max()) > 2e-6:
            print("SELFTEST FAIL graph replay p=%s kcap=%s armed=%s mode=%s" % (p, kcap, armed, mode))
            return 1
    # timing (informational, eager)
    ctrl, sched = new_ctrl(0.9, 4096, 1, dev)
    for M in (8, 64, 128):
        ids, w = _case(gen, dev, M, K, E, "corr")
        oi = torch.empty(M, K, dtype=torch.int32, device=dev)
        ow = torch.empty(M, K, dtype=torch.float32, device=dev)
        for _ in range(20):
            budget_triton(ids, w, oi, ow, ctrl, sched, stats, scratch, E)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(200):
            budget_triton(ids, w, oi, ow, ctrl, sched, stats, scratch, E)
        t1.record()
        torch.cuda.synchronize()
        print("SELFTEST TIMING M=%d %.1f us/launch (eager)" % (M, t0.elapsed_time(t1) * 1000.0 / 200))
    print("SELFTEST PASS n=%d (+switch, +graph replay)" % n)
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)