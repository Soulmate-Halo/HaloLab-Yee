"""Per-step verify length for DSpark without changing the verify layout. Default OFF.

The target verify runs the fixed [bs, 6] rectangle (anchor + 5 drafts). With DSV41_VERIFY_CAP set,
each request gets a live length L = 1 + k (k = drafts actually verified):

  * acceptance is capped at k drafts through the engine's own cutoff (CapCorrectLen + bonus from
    row k), so the committed tokens are exactly those of a verify with k drafts;
  * the router ids of the dead rows (k+1..5) are replaced by the anchor row's ids, so the dead rows
    add no experts to the per-layer set the MoE streams (the saving; MoE is bandwidth-bound);
  * dead rows are never committed (commit <= k + 1), and live rows only attend to earlier rows, so
    their wrong MoE output cannot reach a committed token.

k comes from DSV41_VERIFY_CAP:
  "1".."5"     fixed k for every step (5 = all live, measures the machinery's own cost)
  "conf:T"     k = number of leading drafts whose running product of the draft confidence head's
               per-position survival stays >= T (at least DSV41_VERIFY_CAP_MIN, default 1).
               The head's value for position i is computed from drafts before i only, so the
               cut is a stopping rule: it never looks at the token it drops (exact for sampling).
Sampled rows go through block_verify, which treats dead positions as q = 0, p(X) = 0.

r653 addition (vcap v1, conf mode only): a runtime control file (DSV41_VCAP_CTL, default
/state/vcap_ctl.json). While it does not exist nothing changes. When it exists, rank 0 records every
step's live lengths, confidence, recall rows and accepted drafts (async, no host sync) and
  mode "legacy"  keeps the conf:T decision above (same k), observation only;
  mode "adapt"   takes per-request, per-position thresholds and recall caps computed on the CPU by
                 vcap_policy.VcapPolicy. The decision still uses only drafts before each position and
                 past steps, so it is a stopping rule as above. Rank 0 decides and broadcasts as before.
Stats go to DSV41_VCAP_STATS, finished requests to DSV41_VCAP_REQLOG, the applied mode to DSV41_VCAP_ACK.
"""
import collections
import json
import os
import time

import torch

SPEC = os.environ.get("DSV41_VERIFY_CAP", "").strip()
ENABLED = SPEC not in ("", "0", "off")
# verify rows per request = DSPARK block size + 1 (anchor); 6 for the stock block of 5
STRIDE = int(os.environ.get("DSV41_VERIFY_CAP_STRIDE") or int(os.environ.get("DSPARK_BLOCK_SIZE", "5")) + 1)
MAX_BS = 64
E_TARGET, K_TARGET = 384, 6

_state = {"live": None, "in_verify": False, "conf": None, "fixed": None, "thr": None,
          "kmin": int(os.environ.get("DSV41_VERIFY_CAP_MIN", "1")), "logged": False}
if ENABLED:
    if SPEC.startswith("conf:"):
        # one threshold for every draft, or one per draft 2..5 ("conf:0.25,0.25,0.03,0.06");
        # draft 1 is always verified (DSV41_VERIFY_CAP_MIN, default 1)
        vals = [float(v) for v in SPEC.split(":", 1)[1].split(",")]
        _state["thr"] = vals[0] if len(vals) == 1 else [0.0] + vals
    else:
        _state["fixed"] = int(SPEC)

# ------------------------------------------------------------------ r653 vcap state
VCAP_VERSION = "vcap-v1"
RING = 8
_VC = {
    "ctl": os.environ.get("DSV41_VCAP_CTL", "/state/vcap_ctl.json"),
    "ack": os.environ.get("DSV41_VCAP_ACK", "/state/vcap_ack.json"),
    "stats": os.environ.get("DSV41_VCAP_STATS", "/state/vcap_stats.jsonl"),
    "reqlog": os.environ.get("DSV41_VCAP_REQLOG", "/state/vcap_req.jsonl"),
    "active": False, "mode": "legacy", "tag": "", "cfg": {}, "policy": None, "mtime": None,
    "next_check": 0.0, "rank0": None, "disabled": False, "ready": False, "rids": None, "t_prep": None,
    "pending": None, "is_rec": None, "thr_dev": None, "krec_dev": None, "kadd_dev": None,
    "h2d": None, "h2d_i": 0, "obs": None, "queue": collections.deque(), "last_stats": 0.0,
    "steps": 0, "prints": collections.Counter(), "last_out": None,
}
if ENABLED and _state["thr"] is not None:
    print(f"[verify_cap] vcap v1 ready (ctl={_VC['ctl']}, stride={STRIDE})", flush=True)


def _vc_print(key, msg, limit=3):
    if _VC["prints"][key] < limit:
        _VC["prints"][key] += 1
        print(f"[verify_cap] {msg}", flush=True)


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def live_buf(device=None):
    buf = _state["live"]
    if buf is None:
        assert not _capturing(), "verify_cap: live buffer must exist before capture"
        buf = torch.full((MAX_BS,), STRIDE, dtype=torch.int64,
                         device=device if device is not None else torch.cuda.current_device())
        if _state["fixed"] is not None:
            buf.fill_(1 + _state["fixed"])
        _state["live"] = buf
    return buf


def cutoff(bs):
    return live_buf()[:bs] if ENABLED else None


def _tp_group():
    group = _state.get("tp_group")
    if group is None:
        from sglang.srt.distributed import get_tp_group
        group = _state["tp_group"] = get_tp_group()
    return group


def _vc_k_adapt(confidence, cum, bs):
    """Per-request k on the device from the CPU thresholds prepared for this step."""
    conf = confidence[:bs].float().clamp(0, 1)
    c = cum[:bs]
    P = c.shape[1]
    is_rec = (conf >= 1.0).all(dim=1)
    k_head = (c >= _VC["thr_dev"][:bs, :P]).to(torch.int64).cumprod(dim=1).sum(dim=1)
    k = torch.where(is_rec, _VC["krec_dev"][:bs], k_head) + _VC["kadd_dev"][:bs]
    pol = _VC["policy"]
    kmin, kmax = pol.kbounds() if pol is not None else (_state["kmin"], STRIDE - 1)
    k = k.clamp(min=kmin, max=min(kmax, STRIDE - 1))
    _VC["is_rec"] = is_rec
    return k


def set_live_from_confidence(confidence, bs):
    """confidence [bs, 5] per-position survival (position i given drafts < i)."""
    if _state["thr"] is None or confidence is None:
        return
    buf = live_buf(confidence.device)
    _state["last_conf"] = confidence
    cum = torch.cumprod(confidence.float().clamp(0, 1), dim=1)
    k = None
    if _VC["ready"] and _VC["mode"] == "adapt" and not _VC["disabled"]:
        try:
            k = _vc_k_adapt(confidence, cum, bs)
        except Exception as exc:  # noqa: BLE001
            _vc_fail("adapt k", exc)
            k = None
    if k is None:
        thr = _state["thr"]
        if isinstance(thr, list):
            if _state.get("thr_t") is None or _state["thr_t"].device != cum.device:
                t = (thr + [thr[-1]] * cum.shape[1])[:cum.shape[1]]
                _state["thr_t"] = torch.tensor(t, dtype=torch.float32, device=cum.device)
            thr = _state["thr_t"]
        k = (cum >= thr).to(torch.int64).cumprod(dim=1).sum(dim=1).clamp(min=_state["kmin"], max=STRIDE - 1)
        if _VC["active"] and _VC["rids"] is not None and not _VC["disabled"]:
            try:
                _VC["is_rec"] = (confidence[:bs].float() >= 1.0).all(dim=1)
            except Exception as exc:  # noqa: BLE001
                _vc_fail("is_rec", exc)
    buf[:bs].copy_(k + 1)
    if _VC["active"] and _VC["rids"] is not None and not _VC["disabled"] and _VC["is_rec"] is not None:
        _VC["pending"] = (k[:bs], cum[:bs], _VC["is_rec"][:bs])
    _VC["is_rec"] = None
    # Rank-invariant by construction: the head's input comes out of an all-reduce that may differ
    # in the last bits between ranks, and a live length that differs across ranks would give one
    # verify row two expert sets. Rank 0 decides, as for the engine's own accept (SpecTpSync).
    group = _tp_group()
    if group.world_size > 1:
        group.broadcast(buf[:bs], src=0)


# ------------------------------------------------------------------ r653 vcap helpers (rank 0 only)
def _vc_is_rank0():
    r = _VC["rank0"]
    if r is None:
        r = _VC["rank0"] = (_tp_group().rank_in_group == 0)
    return r


def _vc_fail(where, exc):
    _VC["disabled"] = True
    _VC["ready"] = False
    _VC["pending"] = None
    _vc_print("fail", f"vcap disabled after error in {where}: {exc!r}; staying on conf:{SPEC[5:]}", limit=1)
    try:
        _vc_write_json(_VC["ack"], {"mode": "failed", "tag": _VC["tag"], "version": VCAP_VERSION,
                                    "t": time.time(), "step": _VC["steps"], "error": repr(exc)[:300]})
    except Exception:  # noqa: BLE001
        pass


def _vc_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def _vc_append(path, rows):
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _vc_default_thr():
    thr = _state["thr"]
    if isinstance(thr, list):
        return float(thr[1]) if len(thr) > 1 else 0.1
    return float(thr) if thr is not None else 0.1


def _vc_flush(now):
    pol = _VC["policy"]
    if pol is None:
        return
    rec = pol.stats()
    rec.update({"t": round(now, 3), "tag": _VC["tag"], "mode": _VC["mode"], "version": VCAP_VERSION})
    _vc_append(_VC["stats"], [rec])
    _vc_append(_VC["reqlog"], pol.drain_finished())
    _VC["last_stats"] = now


def _vc_poll_ctl(now):
    if now < _VC["next_check"]:
        return
    _VC["next_check"] = now + 0.5
    try:
        st = os.stat(_VC["ctl"])
    except OSError:
        if _VC["active"]:
            _vc_flush(now)
            _VC["active"] = False
            _VC["mode"] = "legacy"
            _VC["mtime"] = None
            _vc_print("off", "control file gone: back to plain conf mode")
        return
    if st.st_mtime_ns == _VC["mtime"]:
        return
    _VC["mtime"] = st.st_mtime_ns
    try:
        with open(_VC["ctl"], "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        if not isinstance(cfg, dict):
            raise ValueError("control file is not an object")
    except Exception as exc:  # noqa: BLE001
        _vc_print("badctl", f"control file ignored: {exc!r}")
        return
    mode = cfg.get("mode", "legacy")
    if mode not in ("legacy", "adapt"):
        mode = "legacy"
    if _VC["active"]:
        _vc_flush(now)
    cfg = dict(cfg)
    cfg.setdefault("thr", _vc_default_thr())
    if _VC["policy"] is None:
        from vcap_policy import VcapPolicy
        cfg.setdefault("kmin", _state["kmin"])
        _VC["policy"] = VcapPolicy(STRIDE - 1, cfg, seed=0)
    else:
        _VC["policy"].configure(cfg)
    _VC["active"], _VC["mode"], _VC["tag"], _VC["cfg"] = True, mode, str(cfg.get("tag", "")), cfg
    _VC["last_stats"] = now
    _vc_write_json(_VC["ack"], {"mode": mode, "tag": _VC["tag"], "version": VCAP_VERSION, "t": now,
                                "step": _VC["steps"]})
    _vc_print("mode", f"control: mode={mode} tag={_VC['tag']}", limit=1000000)


def _vc_rings(device, P):
    if _VC["h2d"] is not None:
        return
    pin = device.type == "cuda"
    mk = (lambda *s, dtype: torch.zeros(*s, dtype=dtype, pin_memory=True)) if pin else \
        (lambda *s, dtype: torch.zeros(*s, dtype=dtype))
    _VC["thr_dev"] = torch.zeros((MAX_BS, P), dtype=torch.float32, device=device)
    _VC["krec_dev"] = torch.zeros((MAX_BS,), dtype=torch.int64, device=device)
    _VC["kadd_dev"] = torch.zeros((MAX_BS,), dtype=torch.int64, device=device)
    _VC["h2d"] = [{"thr": mk(MAX_BS, P, dtype=torch.float32), "krec": mk(MAX_BS, dtype=torch.int64),
                   "kadd": mk(MAX_BS, dtype=torch.int64), "ev": None} for _ in range(RING)]
    _VC["obs"] = [{"k": mk(MAX_BS, dtype=torch.int64), "cum": mk(MAX_BS, P, dtype=torch.float32),
                   "rec": mk(MAX_BS, dtype=torch.bool), "acc": mk(MAX_BS, dtype=torch.int64), "ev": None,
                   "busy": False} for _ in range(RING)]


def _vc_h2d(thr, krec, kadd, bs, device):
    slot = _VC["h2d"][_VC["h2d_i"]]
    _VC["h2d_i"] = (_VC["h2d_i"] + 1) % RING
    if slot["ev"] is not None and not slot["ev"].query():
        slot["ev"].synchronize()                      # 8 steps old: practically never waits
    P = thr.shape[1]
    slot["thr"][:bs, :P].numpy()[:] = thr
    slot["krec"][:bs].numpy()[:] = krec
    slot["kadd"][:bs].numpy()[:] = kadd
    nb = device.type == "cuda"
    _VC["thr_dev"][:bs, :P].copy_(slot["thr"][:bs, :P], non_blocking=nb)
    _VC["krec_dev"][:bs].copy_(slot["krec"][:bs], non_blocking=nb)
    _VC["kadd_dev"][:bs].copy_(slot["kadd"][:bs], non_blocking=nb)
    if nb:
        ev = torch.cuda.Event()
        ev.record()
        slot["ev"] = ev


def _vc_d2h(rids, t_prep, k, cum, is_rec, acc):
    bs = len(rids)
    slot = None
    for s in _VC["obs"]:
        if not s["busy"]:
            slot = s
            break
    if slot is None:
        _VC["policy"].count("dropped_obs")
        return
    P = cum.shape[1]
    nb = k.device.type == "cuda"
    slot["k"][:bs].copy_(k, non_blocking=nb)
    slot["cum"][:bs, :P].copy_(cum.float(), non_blocking=nb)
    slot["rec"][:bs].copy_(is_rec, non_blocking=nb)
    slot["acc"][:bs].copy_(acc.to(torch.int64), non_blocking=nb)
    ev = None
    if nb:
        ev = torch.cuda.Event()
        ev.record()
    slot["ev"], slot["busy"] = ev, True
    _VC["queue"].append((slot, list(rids), t_prep, bs, P))


def _vc_drain():
    pol = _VC["policy"]
    q = _VC["queue"]
    while q:
        slot, rids, t_prep, bs, P = q[0]
        if slot["ev"] is not None and not slot["ev"].query():
            break
        q.popleft()
        try:
            pol.observe(rids, slot["k"][:bs].numpy().copy(), slot["cum"][:bs, :P].numpy().copy(),
                        slot["rec"][:bs].numpy().copy(), slot["acc"][:bs].numpy().copy(), t_prep)
        finally:
            slot["busy"] = False


def _vc_before_step(batch, bs, device):
    """run_non_compact entry on every rank; work only on rank 0 (outside CUDA graphs)."""
    _VC["ready"] = False
    _VC["rids"] = None
    _VC["pending"] = None
    if _VC["disabled"]:
        return
    try:
        if not _vc_is_rank0():
            return
        now = time.time()
        _VC["steps"] += 1
        _vc_poll_ctl(now)
        if not _VC["active"]:
            return
        pol = _VC["policy"]
        t0 = time.perf_counter()
        _vc_rings(device, STRIDE - 1)
        _vc_drain()
        reqs = getattr(batch, "reqs", None)
        if reqs is None or len(reqs) != bs:
            pol.count("len_mismatch")
            return
        rids = [str(r.rid) for r in reqs]
        temps = []
        for r in reqs:
            sp = getattr(r, "sampling_params", None)
            temps.append(float(getattr(sp, "temperature", 0.0) or 0.0))
        need = _VC["mode"] == "adapt"
        out = pol.prepare_step(rids, temps, now, need=need)
        _VC["rids"], _VC["t_prep"] = rids, now
        if need and out is not None:
            thr, krec, kadd = out
            _vc_h2d(thr, krec, kadd, bs, device)
            _VC["last_out"] = out
            _VC["ready"] = True
        pol.note_cost((time.perf_counter() - t0) * 1e6)
        if now - _VC["last_stats"] >= float(pol.cfg.get("stats_s", 10.0)):
            _vc_flush(now)
    except Exception as exc:  # noqa: BLE001
        _vc_fail("before_step", exc)


def _vc_after_accept(outs):
    pend = _VC["pending"]
    _VC["pending"] = None
    if pend is None or _VC["disabled"] or _VC["rids"] is None:
        return
    try:
        k, cum, is_rec = pend
        bs = len(_VC["rids"])
        _vc_d2h(_VC["rids"], _VC["t_prep"], k[:bs], cum[:bs], is_rec[:bs], outs.correct_len[:bs])
    except Exception as exc:  # noqa: BLE001
        _vc_fail("after_accept", exc)


def _src_rows(m, device):
    """Row each verify row takes its router output from (itself if live, else its anchor).
    Computed once per forward and reused by all 43 layers (inside the graph: once per replay)."""
    src = _state.get("src")
    if src is not None and src.shape[0] == m:
        return src
    live = live_buf(device)
    rows = torch.arange(m, device=device)
    req, pos = rows // STRIDE, rows % STRIDE
    dead = pos >= live[req.clamp(max=MAX_BS - 1)]
    src = torch.where(dead, req * STRIDE, rows)
    _state["src"] = src
    if _capturing() and m not in _state.setdefault("captured", set()):
        _state["captured"].add(m)
        print(f"[verify_cap] remap captured into the verify graph for M={m}", flush=True)
    return src


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _remap_kernel(W, I, LIVE, M, SW: tl.constexpr, SI: tl.constexpr, STRIDE_R: tl.constexpr,
                      K: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        rmask = r < M
        req = r // STRIDE_R
        pos = r % STRIDE_R
        live = tl.load(LIVE + req, rmask, STRIDE_R)
        dead = rmask & (pos >= live)
        anchor = req * STRIDE_R
        for k in tl.static_range(K):
            w = tl.load(W + anchor * SW + k, dead, 0.0)
            i = tl.load(I + anchor * SI + k, dead, 0)
            tl.store(W + r * SW + k, w, dead)
            tl.store(I + r * SI + k, i, dead)
except Exception:                                   # no triton: fall back to index_select
    _remap_kernel = None


def remap_dead_rows(weights, indices):
    """weights/indices [M, 6] for M = bs * 6 verify rows (request-major). In place when possible:
    one kernel per layer copies the anchor row's ids and weights over the dead rows."""
    m = indices.shape[0]
    if (_remap_kernel is not None and weights.is_contiguous() and indices.is_contiguous()
            and weights.shape[1] == K_TARGET and m <= 1024):
        if _capturing() and m not in _state.setdefault("captured", set()):
            _state["captured"].add(m)
            print(f"[verify_cap] in-place remap captured into the verify graph for M={m}", flush=True)
        _remap_kernel[(1,)](weights, indices, live_buf(indices.device), m, weights.stride(0),
                            indices.stride(0), STRIDE, K_TARGET, triton.next_power_of_2(m))
        return weights, indices
    src = _src_rows(m, indices.device)
    return weights.index_select(0, src), indices.index_select(0, src)


def install_gate(module):
    """sglang.kernels.ops.moe.moe_fused_gate"""
    if not ENABLED or getattr(module, "_dsv41_verify_cap", False):
        return
    module._dsv41_verify_cap = True
    original = module.moe_fused_gate
    fused = None
    if (os.environ.get("DSV41_LAUNCHER") == "tp4"
            and os.environ.get("DSV41_ROUTER_LIVE", "0").strip() not in ("0", "", "off", "false")):
        # DSV41_ROUTER_LIVE: the router loads the anchor's scores for dead rows (no remap launch)
        from router_live import build
        fused = build(module)
        print("[verify_cap] dead-row remap folded into the router kernel", flush=True)

    def wrapped(scores, bias, topk, *a, **kw):
        m = scores.shape[0] if scores.dim() == 2 else 0
        verify = (_state["in_verify"] and scores.shape[-1] == E_TARGET and topk == K_TARGET
                  and m and m % STRIDE == 0 and m // STRIDE <= MAX_BS)
        if verify and fused is not None:
            return fused(scores, bias, topk, *a, live=live_buf(scores.device), live_stride=STRIDE, **kw)
        weights, indices = original(scores, bias, topk, *a, **kw)
        if verify:
            return remap_dead_rows(weights, indices)
        return weights, indices

    module.moe_fused_gate = wrapped
    print(f"[verify_cap] armed ({SPEC}): dead verify rows route to the anchor's experts", flush=True)


def install_model(dsv4_module):
    """sglang.srt.models.deepseek_v4: mark target-verify forwards (eager and graph capture)."""
    if not ENABLED:
        return
    cls = dsv4_module.DeepseekV4ForCausalLM
    if getattr(cls, "_dsv41_verify_cap", False):
        return
    cls._dsv41_verify_cap = True
    orig = cls.forward

    def forward(self, input_ids, positions, forward_batch, *a, **kw):
        if _state["live"] is None and not _capturing():
            live_buf(input_ids.device)
        prev = _state["in_verify"]
        _state["src"] = None
        _state["in_verify"] = (type(self).__name__ == "DeepseekV4ForCausalLM"
                               and forward_batch.forward_mode.is_target_verify())
        try:
            return orig(self, input_ids, positions, forward_batch, *a, **kw)
        finally:
            _state["in_verify"] = prev

    cls.forward = forward


class _Cut:
    def __init__(self, verify_lens):
        self.verify_lens = verify_lens


def install_verify(verify_module):
    """sglang.srt.speculative.dspark_components.dspark_verify"""
    if not ENABLED:
        return
    ep = verify_module.DsparkVerifyEpilogue
    if getattr(ep, "_dsv41_verify_cap", False):
        return
    ep._dsv41_verify_cap = True
    orig_accept = ep._accept

    def _accept(self, *, candidates, logits, draft_tokens, seq_lens, cutoff_verify_lens=None):
        if cutoff_verify_lens is None:
            cutoff_verify_lens = cutoff(candidates.shape[0])
        return orig_accept(self, candidates=candidates, logits=logits, draft_tokens=draft_tokens,
                           seq_lens=seq_lens, cutoff_verify_lens=cutoff_verify_lens)

    ep._accept = _accept

    orig_adt = verify_module.accept_draft_tokens

    def accept_draft_tokens(*, candidates, cutoff_layout=None, **kw):
        if cutoff_layout is None:
            cutoff_layout = _Cut(cutoff(candidates.shape[0]))
        return orig_adt(candidates=candidates, cutoff_layout=cutoff_layout, **kw)

    verify_module.accept_draft_tokens = accept_draft_tokens

    ex = verify_module.TargetVerifyExecutor
    orig_run = ex.run_non_compact

    def run_non_compact(self, *, batch, draft_input, verify_ids_2d, **kw):
        bs = verify_ids_2d.shape[0]
        if _state["thr"] is not None:
            _vc_before_step(batch, bs, verify_ids_2d.device)
            set_live_from_confidence(_state["conf"], bs)
            if not _state["logged"]:
                _state["logged"] = True
                print(f"[verify_cap] confidence {'present' if _state['conf'] is not None else 'MISSING'}",
                      flush=True)
        _state["conf"] = None
        return orig_run(self, batch=batch, draft_input=draft_input, verify_ids_2d=verify_ids_2d, **kw)

    ex.run_non_compact = run_non_compact

    # DSV41_VERIFY_CAP_LOG=/state/vcap-log.bin (rank 0): per verify step and request, the confidence
    # vector, the live length used and the accepted drafts. Costs a device sync per step: measurement only.
    log_path = os.environ.get("DSV41_VERIFY_CAP_LOG", "")
    orig_aaf = ex.accept_and_finalize
    if log_path and _state["thr"] is not None:
        _state["log"] = None

        def accept_and_finalize(self, *a, **kw):
            outs = orig_aaf(self, *a, **kw)
            _vc_after_accept(outs)
            conf = _state.get("last_conf")
            if conf is not None:
                import numpy as np
                if _state["log"] is None:
                    from sglang.srt.distributed import get_tp_group
                    _state["log"] = open(log_path, "ab") if get_tp_group().rank_in_group == 0 else False
                if _state["log"]:
                    bs = conf.shape[0]
                    rec = torch.cat([conf.float(), live_buf()[:bs].float().unsqueeze(1),
                                     outs.correct_len[:bs].float().unsqueeze(1)], dim=1)
                    _state["log"].write(rec.cpu().numpy().astype(np.float32).tobytes())
                    _state["log"].flush()
            return outs

        ex.accept_and_finalize = accept_and_finalize
    elif _state["thr"] is not None:
        def accept_and_finalize(self, *a, **kw):
            outs = orig_aaf(self, *a, **kw)
            if _VC["pending"] is not None:
                _vc_after_accept(outs)
            return outs

        ex.accept_and_finalize = accept_and_finalize


def install_draft(draft_module):
    """sglang.srt.speculative.dspark_components.dspark_draft: keep a folded proposal's confidence."""
    if not ENABLED or _state["thr"] is None:
        return
    cls = draft_module.DraftBlockProposer
    orig = cls.propose

    def propose(self, *a, **kw):
        out = orig(self, *a, **kw)
        if getattr(out, "confidence", None) is not None:
            _state["conf"] = out.confidence
        return out

    cls.propose = propose


def install_planner(planner_module):
    """sglang.srt.speculative.dspark_components.dspark_planner: keep the step's confidence."""
    if not ENABLED or _state["thr"] is None:
        return
    cls = planner_module.DSparkVerifyPlanner
    orig = cls.compute_confidence_tensor

    def compute_confidence_tensor(self, **kw):
        out = orig(self, **kw)
        _state["conf"] = out
        return out

    cls.compute_confidence_tensor = compute_confidence_tensor


def install_dspark(dspark_module):
    """sglang.srt.models.deepseek_v4_dspark: the engine builds the draft's confidence head only in
    the ragged-verify modes; build it in static mode too when the conf policy needs it."""
    if not ENABLED or _state["thr"] is None:
        return
    orig = dspark_module.build_dspark_v4_confidence_head
    mode_fn = dspark_module.read_ragged_verify_mode
    from sglang.srt.speculative.ragged_verify import RaggedVerifyMode

    def build(*a, **kw):
        dspark_module.read_ragged_verify_mode = lambda: RaggedVerifyMode.CAP_ACCEPT
        try:
            head = orig(*a, **kw)
        finally:
            dspark_module.read_ragged_verify_mode = mode_fn
        print(f"[verify_cap] draft confidence head built: {head is not None}", flush=True)
        return head

    dspark_module.build_dspark_v4_confidence_head = build
