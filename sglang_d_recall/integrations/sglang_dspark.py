"""Recall copy drafts inside SGLang's DSpark spec-v2 worker (TP4, overlap scheduling).

Per decode step:
  * rank0 builds one plan row per request from the scheduler's Req objects. Under overlap the CPU has only
    processed results up to step k-2 when step k launches, so a row carries the CPU-known length L, the last
    known token and the record's continuation after it (W = gamma + 7 ids in all).
  * every TP rank receives the plan through one NCCL broadcast (same call pattern as Mia's verify_cap).
  * a few GPU ops find the step's anchor in the row (o = seq_len - L + 1 in [0, 6] and W[o] == anchor) and,
    where it lines up, replace DSpark's gamma drafts with W[o+1 : o+1+gamma] - in the proposal tensor and in
    the verify epilogue's draft buffer that the in-graph accept reads - and set those rows' confidence to 1
    so verify_cap verifies them at full length.
  * the target model verifies the copied tokens exactly as it verifies DSpark's, so greedy replies are
    unchanged. Only all-greedy batches without grammar take part.
  * when the step's result is processed, rank0 repeats the alignment test on the CPU (it is the same
    arithmetic) and reports landed tokens to the engine; finished requests add their reply to the record.

Top level imports stay light (no torch / sglang) so the pure functions can be unit tested.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.abc
import json
import os
import sys
import time

import numpy as np

from ..core import ENV_NEW, ENV_OLD, _env

O_MAX = 6  # gamma=5 values, kept for the tests; live code uses _omax(gamma) / _pad(gamma)
PAD = 7
MAX_BS = 64
RING = 8
_TRUE = ("1", "true", "yes", "on")

_state = {
    "engine": None, "engine_failed": False, "group": None, "pinned": None, "slot": 0, "worker": None,
    "paused": False, "pause_t": 0.0, "log_t": 0.0, "installed": False,
}
_fifo: dict = {}  # rid -> deque of (rid, flag, L, W list | None, real); one entry per launched step
_counts = collections.Counter()


def enabled() -> bool:
    return _env(ENV_NEW["on"], ENV_OLD["on"], "0").strip().lower() in _TRUE


def _omax(gamma: int) -> int:
    """Anchor offset range: CPU lag of 0..1 step, and a step commits 1..gamma+1 ids."""
    return int(gamma) + 1


def _pad(gamma: int) -> int:
    """W = gamma + _pad(gamma): any offset in [0, _omax] followed by gamma copied ids."""
    return int(gamma) + 2


# ------------------------------------------------------------------------------------------- pure logic
def build_plan(engine, reqs, gamma: int, paused: bool = False):
    """rank0: plan rows [flag, L, W0..W(W-1)] and FIFO entries for one launched decode step."""
    W = gamma + _pad(gamma)
    bs = len(reqs)
    plan = np.zeros((bs, 2 + W), dtype=np.int64)
    entries = []
    for i, req in enumerate(reqs):
        rid = req.rid
        origin = req.origin_input_ids
        out = req.output_ids
        n_out = len(out)
        L = len(origin) + n_out
        flag, toks, real = 0, None, 0
        if engine is not None and L > 0 and not req.finished():
            k = engine.stream_out_len(rid)
            if k > n_out:  # the request was rolled back: start its stream again
                engine.end_stream(rid, finished=False)
                k = -1
            if k < 0:
                engine.begin_stream(rid, origin, cached_len=int(getattr(req, "cached_tokens", 0) or 0))
                k = 0
            if k < n_out:
                engine.extend_stream(rid, out[k:])
            if not paused:
                prop = engine.propose(rid, n=W - 1)
                if prop is not None:
                    plan[i, 0] = 1
                    plan[i, 1] = L
                    plan[i, 2] = int(out[-1]) if n_out else int(origin[-1])
                    plan[i, 3:2 + W] = prop.tokens[:W - 1]
                    flag, real = 1, int(prop.real)
                    toks = plan[i, 2:].tolist()
        entries.append((rid, flag, L, toks, real))
    return plan, entries


def device_merge(plan, seq_lens, anchors, draft_tokens, confidence, gamma: int, epilogue_buf=None):
    """All ranks, GPU only (no host sync): swap aligned copies in for the drafts; returns the row mask."""
    import torch

    flag = plan[:, 0]
    L = plan[:, 1]
    Wt = plan[:, 2:]
    o = seq_lens.to(torch.int64) - L + 1
    om = _omax(gamma)
    ok = (flag > 0) & (o >= 0) & (o <= om)
    oc = o.clamp(0, om)
    ok = ok & (Wt.gather(1, oc.unsqueeze(1)).squeeze(1) == anchors.to(torch.int64))
    idx = oc.unsqueeze(1) + 1 + torch.arange(gamma, device=Wt.device, dtype=torch.int64).unsqueeze(0)
    ext = Wt.gather(1, idx).to(draft_tokens.dtype)
    merged = torch.where(ok.unsqueeze(1), ext, draft_tokens)
    draft_tokens.copy_(merged)
    if epilogue_buf is not None:
        bs = draft_tokens.shape[0]
        epilogue_buf[: bs * gamma].view(bs, gamma).copy_(merged)
    if confidence is not None:
        confidence.masked_fill_(ok.unsqueeze(1), 1.0)
    return ok


def account(engine, entry, Lb: int, La: int, token_at, gamma: int) -> str:
    """rank0, when the step's result is processed. Lb/La: known length before/after the result."""
    rid, flag, L, W, real = entry
    commit = La - Lb
    if not flag:
        if commit > 0:
            engine.observe_own(rid, commit - 1)
        return "own"
    o = Lb - L  # equals the GPU's seq_len - L + 1: the anchor is the id at index Lb - 1
    if 0 <= o <= _omax(gamma) and Lb >= 1 and W[o] == int(token_at(Lb - 1)):
        avail = max(0, min(gamma, real - o))
        landed = min(commit - 1, avail) if commit > 0 else 0
        engine.feedback(rid, landed)
        return "used"
    engine.feedback(rid, 0, used=False)
    if commit > 0:
        engine.observe_own(rid, commit - 1)
    return "fallback"


def after_step(engine, req, Lb: int, gamma: int) -> None:
    rid = req.rid
    dq = _fifo.get(rid)
    entry = dq.popleft() if dq else None
    if getattr(req, "is_retracted", False):
        engine.drop_pending(rid)
        _fifo.pop(rid, None)
        return
    origin = req.origin_input_ids
    out = req.output_ids
    n0 = len(origin)
    La = n0 + len(out)
    if entry is not None:
        kind = account(engine, entry, Lb, La, lambda i: origin[i] if i < n0 else out[i - n0], gamma)
        _counts[kind] += 1
        if kind == "used":
            _counts["landed_rows"] += 1
    if req.finished():
        name = type(getattr(req, "finished_reason", None)).__name__.upper()
        engine.end_stream(rid, out, finished="ABORT" not in name)
        _fifo.pop(rid, None)


# ------------------------------------------------------------------------------------------- engine
def _digest_from_dir(d: str):
    """(digest, vocab_size, found) for a tokenizer directory: the digest identifies the record."""
    h = hashlib.sha256()
    vocab = 0
    found = False
    for name in ("tokenizer.json", "tokenizer_config.json"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        found = True
        with open(p, "rb") as f:
            data = f.read()
        h.update(name.encode())
        h.update(data)
        if name == "tokenizer.json":
            try:
                js = json.loads(data)
                vocab = len((js.get("model") or {}).get("vocab") or {}) + len(js.get("added_tokens") or [])
            except ValueError:
                pass
    return h.hexdigest(), vocab, found


def _identity() -> dict:
    d = _env(ENV_NEW["tokenizer_dir"], ENV_OLD["tokenizer_dir"], "/models/DeepSeek-V4.1-Flash")
    digest, vocab, found = _digest_from_dir(d)
    if not found:
        print("[recall] warning: no tokenizer files under %s; identity is a placeholder" % d, flush=True)
    return {"tokenizer_digest": digest, "vocab_size": vocab,
            "model": _env(ENV_NEW["model_name"], ENV_OLD["model_name"], "deepseek-v4.1-flash")}


def _get_engine(gamma: int):
    eng = _state["engine"]
    if eng is not None or _state["engine_failed"]:
        return eng
    try:
        from sglang_d_recall.core import engine_from_env

        eng = engine_from_env(gamma, _identity())
        if eng is None:
            raise RuntimeError("%s is not on in this process" % (ENV_NEW["on"],))
        _state["engine"] = eng
        st = eng.record.stats()
        print("[recall] engine ready dir=%s backend=%s ids=%d state=%s gamma=%d"
              % (st["dir"], st.get("backend", "disk"), st["ids"], st["state"], gamma), flush=True)
    except Exception as e:  # noqa: BLE001
        _state["engine_failed"] = True
        print("[recall] engine init failed, copies stay off: %r" % (e,), flush=True)
    return _state["engine"]


def _check_pause(eng) -> bool:
    now = time.monotonic()
    if now - _state["pause_t"] >= 2.0:
        _state["pause_t"] = now
        d = getattr(eng.record, "dir", None)
        _state["paused"] = bool(d) and os.path.exists(os.path.join(d, "paused"))
    return _state["paused"]


def _maybe_log(eng) -> None:
    now = time.monotonic()
    if now - _state["log_t"] < 30.0:
        return
    _state["log_t"] = now
    st = eng.stats()
    print("[recall] steps=%d rows=%d plan_rows=%d used=%d fallback=%d own=%d landed=%d offers=%d probes=%d "
          "record_ids=%d streams=%d paused=%s errors=%d"
          % (_counts["steps"], _counts["rows"], _counts["plan_rows"], _counts["used"], _counts["fallback"],
             _counts["own"], st["landed_tokens"], st["offers"], st["probes"], st["ids"], st["active_streams"],
             _state["paused"], _counts["plan_errors"] + _counts["account_errors"]), flush=True)


# ------------------------------------------------------------------------------------------- hooks
def _participates(kw, out) -> bool:
    """Identical on every TP rank: depends only on state all ranks share."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    batch = kw.get("batch")
    bs = kw.get("bs")
    if batch is None or bs is None or not (0 < int(bs) <= MAX_BS):
        return False
    if getattr(batch, "has_grammar", False):
        return False
    si = kw.get("sampling_info")
    if si is not None and not getattr(si, "is_all_greedy", False):
        return False
    dt = getattr(getattr(out, "draft_block", None), "draft_tokens", None)
    return dt is not None and dt.dim() == 2 and dt.shape[0] == int(bs)


def _pinned(width: int):
    import torch

    buf = _state["pinned"]
    if buf is None or buf.shape[2] != width:
        buf = _state["pinned"] = torch.zeros((RING, MAX_BS, width), dtype=torch.int64, pin_memory=True)
    slot = _state["slot"]
    _state["slot"] = (slot + 1) % RING
    return buf[slot]


def _on_propose(worker, kw, out) -> None:
    import torch

    batch = kw["batch"]
    bs = int(kw["bs"])
    dt = out.draft_block.draft_tokens
    gamma = int(dt.shape[1])
    width = 2 + gamma + _pad(gamma)
    group = _state["group"]
    if group is None:
        from sglang.srt.distributed import get_tp_group

        group = _state["group"] = get_tp_group()
    rank0 = group.rank_in_group == 0
    plan = torch.zeros((bs, width), dtype=torch.int64, device=dt.device)
    if rank0:
        eng = _get_engine(gamma)
        entries = None
        try:
            if eng is not None:
                arr, entries = build_plan(eng, batch.reqs, gamma, paused=_check_pause(eng))
                if arr.shape == (bs, width) and arr[:, 0].any():
                    host = _pinned(width)
                    host[:bs].copy_(torch.from_numpy(arr))
                    plan.copy_(host[:bs], non_blocking=True)
                    _counts["plan_rows"] += int(arr[:, 0].sum())
        except Exception as e:  # noqa: BLE001  never skip the broadcast below
            _counts["plan_errors"] += 1
            if _counts["plan_errors"] <= 3:
                print("[recall] plan error (zero plan this step): %r" % (e,), flush=True)
            entries = None
            plan.zero_()
        if eng is not None:
            if entries is None:
                entries = [(r.rid, 0, 0, None, 0) for r in batch.reqs]
            for e in entries:
                dq = _fifo.get(e[0])
                if dq is None:
                    dq = _fifo[e[0]] = collections.deque()
                dq.append(e)
        _counts["steps"] += 1
        _counts["rows"] += bs
    group.broadcast(plan, src=0)
    epi = getattr(worker, "_verify_epilogue", None) if worker is not None else None
    device_merge(plan, batch.seq_lens, out.draft_block_ids[:, 0], dt, out.confidence, gamma,
                 epilogue_buf=getattr(epi, "draft_tokens_buf", None))


def _make_propose(worker, inner):
    def propose(*args, **kw):
        out = inner(*args, **kw)
        if _participates(kw, out):
            _on_propose(worker, kw, out)
        return out

    propose.__recall_wrapped__ = True
    return propose


def _patch_worker_module(module) -> None:
    cls = module.DSparkWorkerV2
    orig_init = cls.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        prop = getattr(self, "_proposer", None)
        if prop is None:
            print("[recall] DSparkWorkerV2 has no _proposer; copies stay off", flush=True)
            return
        _state["worker"] = self
        try:
            prop.propose = _make_propose(self, prop.propose)
            print("[recall] DSpark proposer wrapped (instance)", flush=True)
        except (AttributeError, TypeError):
            pcls = type(prop)
            if not getattr(pcls.propose, "__recall_wrapped__", False):
                inner = pcls.propose

                def propose(self_, *a2, **kw2):
                    out = inner(self_, *a2, **kw2)
                    if _participates(kw2, out):
                        _on_propose(_state["worker"], kw2, out)
                    return out

                propose.__recall_wrapped__ = True
                pcls.propose = propose
            print("[recall] DSpark proposer wrapped (class)", flush=True)

    cls.__init__ = __init__
    print("[recall] patched %s" % module.__name__, flush=True)


def _patch_processor_module(module) -> None:
    cls = module.SchedulerBatchResultProcessor
    orig = cls.process_batch_result_decode

    def process_batch_result_decode(self, batch, result):
        eng = _state["engine"]
        if eng is None or batch.spec_algorithm.is_none():
            return orig(self, batch, result)
        reqs = list(batch.reqs)
        before = [len(r.origin_input_ids) + len(r.output_ids) for r in reqs]
        ret = orig(self, batch, result)
        try:
            for req, Lb in zip(reqs, before):
                after_step(eng, req, Lb, eng.gamma)
            eng.tick()
            _maybe_log(eng)
        except Exception as e:  # noqa: BLE001
            _counts["account_errors"] += 1
            if _counts["account_errors"] <= 3:
                print("[recall] accounting error: %r" % (e,), flush=True)
        return ret

    cls.process_batch_result_decode = process_batch_result_decode
    print("[recall] patched %s" % module.__name__, flush=True)


def _patch_dspark_model_module(module) -> None:
    """r580: DSPARK_BLOCK_SIZE above the checkpoint block (5). The model's compute_confidence reshapes by
    the checkpoint gamma; take gamma from the drafts it is handed instead (identical at gamma=5)."""
    cls = module.DeepseekV4ForCausalLMDSpark
    orig = cls.compute_confidence

    def compute_confidence(self, *, anchor_tokens, sampled_tokens, x_post_hc):
        g = int(sampled_tokens.shape[1])
        if g != self.gamma:
            print("[recall] draft compute_confidence gamma %d -> %d (runtime)" % (self.gamma, g), flush=True)
            self.gamma = g
        return orig(self, anchor_tokens=anchor_tokens, sampled_tokens=sampled_tokens, x_post_hc=x_post_hc)

    cls.compute_confidence = compute_confidence
    print("[recall] patched %s" % module.__name__, flush=True)


TARGETS = {
    "sglang.srt.speculative.dspark_components.dspark_worker_v2": _patch_worker_module,
    "sglang.srt.managers.scheduler_components.batch_result_processor": _patch_processor_module,
    "sglang.srt.models.deepseek_v4_dspark": _patch_dspark_model_module,
}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, fn):
        self.inner = inner
        self.fn = fn

    def create_module(self, spec):
        cm = getattr(self.inner, "create_module", None)
        return cm(spec) if cm is not None else None

    def exec_module(self, module):
        self.inner.exec_module(module)
        try:
            self.fn(module)
        except Exception as e:  # noqa: BLE001
            print("[recall] patch of %s failed: %r" % (module.__name__, e), flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    """Wraps the loader of whatever spec the other finders return, so Mia's own loader still runs first."""

    def __init__(self, targets):
        self.targets = targets

    def find_spec(self, fullname, path=None, target=None):
        fn = self.targets.get(fullname)
        if fn is None:
            return None
        for f in list(sys.meta_path):
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None:
                    spec.loader = _Loader(spec.loader, fn)
                return spec
        return None


def install(targets=None) -> None:
    targets = dict(TARGETS if targets is None else targets)
    for name, fn in list(targets.items()):
        mod = sys.modules.get(name)
        if mod is not None:  # imported before us: patch in place
            fn(mod)
            targets.pop(name)
    if targets:
        sys.meta_path.insert(0, _Finder(targets))
    _state["installed"] = True
    print("[recall] hooks installed for %d module(s) (pid %d)" % (len(targets), os.getpid()), flush=True)
