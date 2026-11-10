"""Recall copy drafts inside vLLM's V2 GPU model runner (``vllm/v1/worker/gpu``), DSpark / DFlash speculators.

Where it sits (vLLM 0.28.1rc1, V2 runner):
  * ``GPUModelRunner.execute_model`` samples, calls ``postprocess_sampled`` (which writes the accepted ids into
    ``req_states.all_token_ids`` - a UVA buffer in pinned host memory - and the true lengths into
    ``req_states.total_len.gpu``), then ``self.speculator.propose(...)``, and stores the returned
    ``[num_reqs, num_speculative_steps]`` drafts in ``req_states.draft_tokens``; the next step verifies them.
  * ``DSparkSpeculator`` subclasses ``DFlashSpeculator`` and does not override ``propose``, so wrapping
    ``DFlashSpeculator.propose`` covers both methods.

Per participating step (no structured output, not a dummy / profile / graph capture run; all requests greedy,
or - RECALL_ACCEL_NONGREEDY, default on - any sampling while the speculator keeps no draft logits: with
``draft_logits`` None the rejection sampler treats a draft as one-hot (log q = 0), so a copied id is accepted with
p(x) and the residual is p without x, the target distribution after penalties/temperature/top-k/top-p is kept):
  * rank0 knows each request's ids only up to a CPU-visible length L (a non-blocking D2H snapshot of total_len
    taken at an earlier step whose CUDA event has completed; the ids below L are final in the UVA buffer).
  * rank0 builds a plan row [flag, L, W0 = id at L-1, W1..] with the record's continuation after the known ids.
  * every TP rank receives the plan (one NCCL broadcast) and a few GPU ops align it with the step's true
    length T (o = T - L, W[o] must equal last_sampled) and swap W[o+1 : o+1+gamma] in for the model drafts.
    No host sync is added; a row that does not line up keeps the model's drafts.
  * the target model verifies the copied ids like any drafts, so greedy output is unchanged.
  * when the next snapshot of a request completes, rank0 repeats the alignment test on the CPU (same
    arithmetic) and reports the landed ids to the engine; finished requests add their reply to the record.

Top level imports stay light (no torch / vllm) so the pure functions can be unit tested.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.abc
import inspect
import json
import os
import sys
import time

import numpy as np

from ..core import ENV_NEW, ENV_OLD, _env

LAG_STEPS = 3  # CPU snapshots may trail the step being launched by up to this many steps
MAX_BS = 512
RING = 16  # pinned snapshot slots in flight
_TRUE = ("1", "true", "yes", "on")
TAG = "[recall-accel]"

PROPOSE_PARAMS = (
    "self", "input_batch", "attn_metadata", "slot_mappings", "last_hidden_states", "aux_hidden_states",
    "num_sampled", "num_rejected", "last_sampled", "next_prefill_tokens", "temperature", "seeds", "dp_sync",
    "dummy_run", "skip_attn_for_dummy_run", "mm_inputs", "is_profile",
)

_state = {
    "engine": None, "engine_failed": False, "group": None, "runner": None, "paused": False, "pause_t": 0.0,
    "log_t": 0.0, "installed": False, "pending": collections.deque(), "slot": 0, "pinned": None,
    "err_printed": 0,
}
_hist: dict = {}  # rid -> (np.int64 ids known on the CPU, prompt_len)
_prev: dict = {}  # rid -> (entry, T at the step the entry was made); entry = (flag, L, W list | None, real)
_counts = collections.Counter()


def enabled() -> bool:
    return _env(ENV_NEW["on"], ENV_OLD["on"], "0").strip().lower() in _TRUE


def omax(gamma: int) -> int:
    """Largest anchor offset: LAG_STEPS steps of CPU lag, each step commits 1..gamma+1 ids."""
    return LAG_STEPS * (int(gamma) + 1)


def width(gamma: int) -> int:
    """W = omax + 1 + gamma ids: any offset in [0, omax] followed by gamma copied ids."""
    return omax(gamma) + 1 + int(gamma)


def _log_error(what: str, e: BaseException) -> None:
    _counts["errors"] += 1
    if _state["err_printed"] < 5:
        _state["err_printed"] += 1
        print("%s %s, this step keeps the model drafts: %r" % (TAG, what, e), flush=True)


# ------------------------------------------------------------------------------------------- pure logic
def sync_stream(engine, rid, T: int, plen: int, read) -> None:
    """Bring the engine's stream of ``rid`` up to length T. ``read(a, b)`` returns ids [a, b) as int64."""
    plen = max(0, min(int(plen), int(T)))
    n_out = int(T) - plen
    k = engine.stream_out_len(rid)
    if k > n_out:  # rolled back (preempted and resumed with fewer ids): start the stream again
        engine.end_stream(rid, finished=False)
        k = -1
    if k < 0:
        engine.begin_stream(rid, read(0, plen))
        k = 0
    if k < n_out:
        engine.extend_stream(rid, read(plen + k, int(T)))


def build_plan(engine, rids, gamma: int, paused: bool = False, hist=None):
    """rank0: plan rows [flag, L, W0..W(W-1)] and one entry per row for the step being launched.

    ``hist[rid]`` = (L, prompt_len, id at L-1): what the CPU knows of the request.
    """
    hist = _hist if hist is None else hist
    Wn = width(gamma)
    bs = len(rids)
    plan = np.zeros((bs, 2 + Wn), dtype=np.int64)
    entries = []
    for i, rid in enumerate(rids):
        h = hist.get(rid)
        if engine is None or h is None or paused or h[0] <= 0:
            entries.append((0, 0, None, 0))
            continue
        L = int(h[0])
        prop = engine.propose(rid, n=Wn - 1)
        if prop is None:
            entries.append((0, L, None, 0))
            continue
        plan[i, 0] = 1
        plan[i, 1] = L
        plan[i, 2] = int(h[2])
        plan[i, 3:2 + Wn] = prop.tokens[:Wn - 1]
        entries.append((1, L, plan[i, 2:].tolist(), int(prop.real)))
    return plan, entries


def device_merge(plan, true_len, anchors, draft_tokens, gamma: int, confidence=None):
    """All ranks, device ops only: swap aligned copies in for the drafts (in place); returns the row mask.

    plan [bs, 2+W] int64; true_len [bs] = total_len after this step's accepted ids; anchors [bs] = the last
    accepted id (at index true_len - 1); draft_tokens [bs, gamma].
    """
    import torch

    flag = plan[:, 0]
    L = plan[:, 1]
    Wt = plan[:, 2:]
    o = true_len.to(torch.int64) - L
    om = omax(gamma)
    ok = (flag > 0) & (o >= 0) & (o <= om)
    oc = o.clamp(0, om)
    ok = ok & (Wt.gather(1, oc.unsqueeze(1)).squeeze(1) == anchors.to(torch.int64))
    idx = oc.unsqueeze(1) + 1 + torch.arange(gamma, device=Wt.device, dtype=torch.int64).unsqueeze(0)
    ext = Wt.gather(1, idx).to(draft_tokens.dtype)
    draft_tokens.copy_(torch.where(ok.unsqueeze(1), ext, draft_tokens))
    if confidence is not None:
        confidence.masked_fill_(ok.unsqueeze(1), 1.0)
    return ok


def account(engine, rid, entry, Tb: int, Ta: int, token_at, gamma: int) -> str:
    """rank0: the drafts made with ``entry`` at true length Tb were verified; the length is now Ta."""
    flag, L, W, real = entry
    commit = int(Ta) - int(Tb)
    if not flag:
        if commit > 0:
            engine.observe_own(rid, commit - 1)
        return "own"
    o = int(Tb) - int(L)
    if 0 <= o <= omax(gamma) and Tb >= 1 and W[o] == int(token_at(Tb - 1)):
        avail = max(0, min(gamma, real - o))
        landed = min(commit - 1, avail) if commit > 0 else 0
        engine.feedback(rid, max(0, landed))
        return "used"
    engine.feedback(rid, 0, used=False)
    if commit > 0:
        engine.observe_own(rid, commit - 1)
    return "fallback"


def on_snapshot(engine, rid, T: int, plen: int, read, entry, gamma: int) -> None:
    """rank0: a snapshot of ``rid`` is complete: its true length was T when the step's drafts were made.
    ``read(a, b)`` returns the request's ids [a, b) (final for b <= T)."""
    T = int(T)
    prev = _prev.pop(rid, None)
    if prev is not None:
        pentry, Tb = prev
        if T >= Tb:
            kind = account(engine, rid, pentry, Tb, T, lambda i: int(read(i, i + 1)[0]), gamma)
            _counts[kind] += 1
        elif pentry[0]:
            engine.feedback(rid, 0, used=False)
    _hist[rid] = (T, int(plen), int(read(T - 1, T)[0]))
    sync_stream(engine, rid, T, plen, read)
    if entry is not None:
        _prev[rid] = (entry, T)


def forget_prev(engine, rids) -> None:
    """rank0: these requests ran a step without the hook: their last copies cannot be measured."""
    for rid in rids:
        p = _prev.pop(rid, None)
        if p is not None and p[0][0] and engine is not None:
            engine.feedback(rid, 0, used=False)


def finish(engine, rid, finished: bool, read=None) -> None:
    """rank0: the scheduler finished (or preempted) ``rid``. ``read`` as in on_snapshot (slot still owned)."""
    h = _hist.pop(rid, None)
    p = _prev.pop(rid, None)
    if engine is None:
        return
    if p is not None and p[0][0]:
        engine.feedback(rid, 0, used=False)
    out = None
    if h is not None and read is not None and h[0] > h[1]:
        out = read(h[1], h[0])
    engine.end_stream(rid, out, finished=bool(finished and out is not None and len(out)))
    _counts["finished" if finished else "preempted"] += 1


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


def _tokenizer_dir(runner=None) -> str:
    d = _env(ENV_NEW["tokenizer_dir"], ENV_OLD["tokenizer_dir"], "")
    if d:
        return d
    try:
        mc = runner.vllm_config.model_config
        for cand in (getattr(mc, "tokenizer", None), getattr(mc, "model", None)):
            if cand and os.path.isdir(str(cand)):
                return str(cand)
    except Exception:  # noqa: BLE001
        pass
    return "/models/DeepSeek-V4.1-Flash"


def _identity(runner=None) -> dict:
    d = _tokenizer_dir(runner)
    digest, vocab, found = _digest_from_dir(d)
    if not found:
        print("%s warning: no tokenizer files under %s; identity is a placeholder" % (TAG, d), flush=True)
    name = _env(ENV_NEW["model_name"], ENV_OLD["model_name"], "")
    if not name:
        try:
            name = str(runner.vllm_config.model_config.served_model_name or "")
        except Exception:  # noqa: BLE001
            name = ""
    return {"tokenizer_digest": digest, "vocab_size": vocab, "model": name or "deepseek-v4.1-flash"}


def _get_engine(gamma: int):
    eng = _state["engine"]
    if eng is not None or _state["engine_failed"]:
        return eng
    try:
        from ..core import engine_from_env

        eng = engine_from_env(gamma, _identity(_state["runner"]))
        if eng is None:
            raise RuntimeError("%s is not on in this process" % (ENV_NEW["on"],))
        _state["engine"] = eng
        st = eng.record.stats()
        print("%s engine ready dir=%s backend=%s ids=%d state=%s gamma=%d"
              % (TAG, st["dir"], st.get("backend", "disk"), st["ids"], st["state"], gamma), flush=True)
    except Exception as e:  # noqa: BLE001
        _state["engine_failed"] = True
        print("%s engine init failed, copies stay off: %r" % (TAG, e), flush=True)
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
    print("%s steps=%d rows=%d plan_rows=%d used=%d fallback=%d own=%d landed=%d offers=%d "
          "record_ids=%d streams=%d pending=%d paused=%s errors=%d"
          % (TAG, _counts["steps"], _counts["rows"], _counts["plan_rows"], _counts["used"], _counts["fallback"],
             _counts["own"], st["landed_tokens"], st["offers"], st["ids"], st["active_streams"],
             len(_state["pending"]), _state["paused"], _counts["errors"]), flush=True)


# ------------------------------------------------------------------------------------------- runner views
def _uva_np(staged):
    """CPU numpy view of a StagedWriteTensor created with uva_instead_of_gpu=True."""
    buf = getattr(staged, "_uva_buf", None)
    if buf is not None and getattr(buf, "np", None) is not None:
        return buf.np
    cpu = getattr(staged, "cpu", None)
    if cpu is not None:
        return cpu.numpy() if hasattr(cpu, "numpy") else np.asarray(cpu)
    return staged.gpu.cpu().numpy()  # CPU-only test doubles


def _all_greedy(runner, idx_np) -> bool:
    sampler = getattr(runner, "sampler", None)
    if sampler is None:
        return False
    temp = sampler.sampling_states.temperature.np
    return bool(np.all(temp[idx_np] == 0.0))


def _nongreedy_ok(spec, runner) -> bool:
    """Sampled requests may take copies: the switch is on and the speculator keeps no draft distribution
    (``draft_sample_method`` "greedy", the default). A probabilistic drafter's cached q would no longer match a
    copied id, so those batches stay all-greedy only."""
    if str(os.environ.get("RECALL_ACCEL_NONGREEDY", "1")).strip().lower() in ("0", "false", "no", "off"):
        return False
    if spec is None:
        spec = _state.get("spec")
    if spec is None:
        spec = getattr(runner, "speculator", None)
    if spec is None:
        return False
    return getattr(spec, "draft_logits", None) is None


def participates(runner, input_batch, draft_tokens, dummy_run: bool, is_profile: bool, spec=None) -> bool:
    """Identical on every TP rank: depends only on state all ranks share."""
    import torch

    if dummy_run or is_profile or runner is None or input_batch is None:
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    n = int(getattr(input_batch, "num_reqs", 0) or 0)
    if not (0 < n <= MAX_BS):
        return False
    if draft_tokens is None or draft_tokens.dim() != 2 or int(draft_tokens.shape[0]) != n:
        return False
    if getattr(input_batch, "has_structured_output_reqs", False):
        return False
    idx_np = np.asarray(input_batch.idx_mapping_np[:n])
    if idx_np.size != n or np.any(idx_np < 0):
        return False
    return _all_greedy(runner, idx_np) or _nongreedy_ok(spec, runner)


def _host_slot(n: int, dev):
    import torch

    buf = _state["pinned"]
    if buf is None:
        pin = bool(torch.cuda.is_available() and dev.type == "cuda")
        buf = _state["pinned"] = torch.zeros((RING, MAX_BS), dtype=torch.int32, pin_memory=pin)
    slot = _state["slot"]
    _state["slot"] = (slot + 1) % RING
    return buf[slot, :n]


def _plan_slot(n: int, w: int, dev):
    import torch

    buf = _state.get("plan_pinned")
    if buf is None or buf.shape[2] != w:
        pin = bool(torch.cuda.is_available() and dev.type == "cuda")
        buf = _state["plan_pinned"] = torch.zeros((RING, MAX_BS, w), dtype=torch.int64, pin_memory=pin)
    slot = _state.get("plan_slot", 0)
    _state["plan_slot"] = (slot + 1) % RING
    return buf[slot, :n]


def _reader(all_np, idx: int):
    def read(a, b):
        return np.array(all_np[idx, int(a):int(b)], dtype=np.int64)

    return read


def drain(engine, runner, gamma: int, block: bool = False) -> int:
    """rank0: consume completed snapshots (oldest first). Returns how many were consumed."""
    pend = _state["pending"]
    done = 0
    rs = runner.req_states
    all_np = None
    while pend:
        ev, rids, idx_np, entries, host = pend[0]
        if ev is not None:
            if block:
                ev.synchronize()
            elif not ev.query():
                break
        pend.popleft()
        done += 1
        if all_np is None:
            all_np = _uva_np(rs.all_token_ids)
        lens = host.numpy() if hasattr(host, "numpy") else np.asarray(host)
        plen_np = rs.prompt_len.np
        owner = rs.index_to_req_id
        for i, rid in enumerate(rids):
            idx = int(idx_np[i])
            if owner.get(idx) != rid:  # the slot was freed / reused meanwhile
                forget_prev(engine, (rid,))
                continue
            T = int(lens[i])
            if T <= 0:
                continue
            on_snapshot(engine, rid, T, int(plen_np[idx]), _reader(all_np, idx),
                        entries[i] if entries else None, gamma)
    return done


def _on_propose(spec, runner, input_batch, draft_tokens, last_sampled, gamma: int) -> None:
    import torch

    n = int(input_batch.num_reqs)
    rids = list(input_batch.req_ids[:n])
    idx_np = np.array(input_batch.idx_mapping_np[:n], dtype=np.int64)
    idx = input_batch.idx_mapping[:n].to(torch.int64)
    dev = draft_tokens.device
    Wn = width(gamma)
    group = _state["group"]
    if group is None:
        try:
            from vllm.distributed.parallel_state import get_tp_group

            group = _state["group"] = get_tp_group()
        except Exception:  # noqa: BLE001  tests / single process
            group = _state["group"] = _SingleGroup()
    rank0 = int(getattr(group, "rank_in_group", 0)) == 0
    world = int(getattr(group, "world_size", 1))
    plan = torch.zeros((n, 2 + Wn), dtype=torch.int64, device=dev)
    entries = None
    eng = None
    if rank0:
        eng = _get_engine(gamma)
        if eng is not None:
            try:
                drain(eng, runner, gamma)
                if len(_state["pending"]) >= RING - 1:
                    drain(eng, runner, gamma, block=True)
                arr, entries = build_plan(eng, rids, gamma, paused=_check_pause(eng))
                if arr[:, 0].any():
                    hp = _plan_slot(n, 2 + Wn, dev)
                    hp.copy_(torch.from_numpy(arr))
                    plan.copy_(hp, non_blocking=True)
                    _counts["plan_rows"] += int(arr[:, 0].sum())
            except Exception as e:  # noqa: BLE001  never skip the broadcast below
                _log_error("plan error", e)
                entries = [(0, 0, None, 0)] * n
                plan.zero_()
        _counts["steps"] += 1
        _counts["rows"] += n
    if world > 1:
        group.broadcast(plan, src=0)
    rs = runner.req_states
    true_len = rs.total_len.gpu[idx]
    anchors = last_sampled[idx, 0] if last_sampled.dim() == 2 else last_sampled[idx]
    conf = getattr(spec, "draft_token_confidence_probs", None)
    if conf is not None and conf.dim() == 2 and conf.shape[1] == gamma and conf.shape[0] >= n:
        conf = conf[:n]
    else:
        conf = None
    device_merge(plan, true_len, anchors, draft_tokens, gamma, confidence=conf)
    if rank0 and eng is not None:
        host = _host_slot(n, dev)
        host.copy_(true_len.to(torch.int32), non_blocking=True)
        ev = _event_factory(dev)
        if ev is not None:
            ev.record()
        _state["pending"].append((ev, rids, idx_np, entries, host))
        eng.tick()
        _maybe_log(eng)


class _SingleGroup:
    rank_in_group = 0
    world_size = 1

    def broadcast(self, t, src=0):
        return t


def _default_event_factory(dev):
    if dev.type != "cuda":
        return None  # CPU tensors: the copy above already happened
    import torch

    return torch.cuda.Event()


_event_factory = _default_event_factory  # tests swap this to simulate a lagging CPU view


def _make_propose(inner):
    try:
        names = list(inspect.signature(inner).parameters)
    except (TypeError, ValueError):
        names = list(PROPOSE_PARAMS)
    pos = {nm: i for i, nm in enumerate(names)}

    def arg(a, kw, name, default=None):
        if name in kw:
            return kw[name]
        i = pos.get(name)
        if i is not None and i < len(a):
            return a[i]
        return default

    def propose(*a, **kw):
        if _state.get("in_propose"):  # a subclass' wrapped propose calling super().propose
            return inner(*a, **kw)
        _state["in_propose"] = True
        try:
            out = inner(*a, **kw)
        finally:
            _state["in_propose"] = False
        try:
            runner = _state["runner"]
            ib = arg(a, kw, "input_batch")
            dummy = bool(arg(a, kw, "dummy_run", False)) or bool(arg(a, kw, "is_profile", False))
            _state["spec"] = a[0] if a else None  # the speculator whose draft_logits decide sampled batches
            if participates(runner, ib, out, dummy, False):
                _on_propose(a[0], runner, ib, out, arg(a, kw, "last_sampled"), int(out.shape[1]))
            elif not dummy and ib is not None and _state["engine"] is not None:
                forget_prev(_state["engine"], list(ib.req_ids[:int(ib.num_reqs)]))
        except Exception as e:  # noqa: BLE001  never break serving
            _log_error("propose hook error", e)
        return out

    propose.__wrapped__ = inner
    propose.__recall_wrapped__ = True
    try:
        propose.__signature__ = inspect.signature(inner)
    except (TypeError, ValueError):
        pass
    propose.__name__ = getattr(inner, "__name__", "propose")
    propose.__qualname__ = getattr(inner, "__qualname__", "propose")
    propose.__doc__ = getattr(inner, "__doc__", None)
    return propose


# vLLM V2 speculators whose propose() returns the [num_reqs, num_speculative_steps] draft tensor
# (vllm/v1/worker/gpu/spec_decode/__init__.py init_speculator). DSpark inherits DFlashSpeculator.propose.
SPECULATORS = {
    # EagleSpeculator / MTPSpeculator / Gemma4Speculator inherit AutoRegressiveSpeculator.propose (0.28.1rc1 MRO)
    "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator": ("AutoRegressiveSpeculator",),
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator": ("DFlashSpeculator",),
    "vllm.v1.worker.gpu.spec_decode.dspark.speculator": ("DSparkSpeculator",),
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator": ("DFlash2Speculator",),
    "vllm.v1.worker.gpu.spec_decode.eagle.speculator": ("EagleSpeculator",),
    "vllm.v1.worker.gpu.spec_decode.mtp.speculator": ("MTPSpeculator",),
    "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator": ("MultiModuleMTPSpeculator",),
    "vllm.v1.worker.gpu.spec_decode.gemma4.speculator": ("Gemma4Speculator",),
}


def _patch_speculator_class(cls, modname: str) -> bool:
    """Wrap ``cls.propose`` when the class defines its own (inherited ones are already wrapped)."""
    own = cls.__dict__.get("propose")
    if own is None or getattr(own, "__recall_wrapped__", False):
        return False
    cls.propose = _make_propose(own)
    print("%s patched %s.%s.propose" % (TAG, modname, cls.__name__), flush=True)
    return True


def _patch_speculator_module(module) -> None:
    for name in SPECULATORS.get(module.__name__, ("DFlashSpeculator",)):
        cls = getattr(module, name, None)
        if cls is not None:
            _patch_speculator_class(cls, module.__name__)


def _patch_runner_module(module) -> None:
    cls = module.GPUModelRunner
    if getattr(cls.__init__, "__recall_wrapped__", False):
        return
    orig_init = cls.__init__
    orig_finish = cls.finish_requests

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        if getattr(self, "speculator", None) is not None:
            _state["runner"] = self
            print("%s runner registered (speculator %s)" % (TAG, type(self.speculator).__name__), flush=True)

    def finish_requests(self, scheduler_output):
        eng = _state["engine"]
        if eng is not None and self is _state["runner"]:
            try:
                fin = set(getattr(scheduler_output, "finished_req_ids", None) or ())
                pre = set(getattr(scheduler_output, "preempted_req_ids", None) or ())
                if fin or pre:
                    gamma = int(getattr(eng, "gamma", 5))
                    drain(eng, self, gamma)
                    rs = self.req_states
                    all_np = None
                    for rid in sorted(fin | pre):
                        if rid in _hist or rid in _prev or eng.has_stream(rid):
                            idx = rs.req_id_to_index.get(rid)
                            read = None
                            if idx is not None:
                                if all_np is None:
                                    all_np = _uva_np(rs.all_token_ids)
                                read = _reader(all_np, int(idx))
                            finish(eng, rid, finished=rid in fin and rid not in pre, read=read)
            except Exception as e:  # noqa: BLE001
                _log_error("finish hook error", e)
        return orig_finish(self, scheduler_output)

    __init__.__recall_wrapped__ = True
    __init__.__wrapped__ = orig_init
    finish_requests.__wrapped__ = orig_finish
    cls.__init__ = __init__
    cls.finish_requests = finish_requests
    print("%s patched %s.GPUModelRunner" % (TAG, module.__name__), flush=True)


TARGETS = dict({name: _patch_speculator_module for name in SPECULATORS},
               **{"vllm.v1.worker.gpu.model_runner": _patch_runner_module})


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
            print("%s patch of %s failed: %r" % (TAG, module.__name__, e), flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    """Wraps the loader of whatever spec the other finders return."""

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
    print("%s hooks installed, %d module(s) pending import (pid %d)" % (TAG, len(targets), os.getpid()),
          flush=True)


def reset_state() -> None:
    """Tests: forget the engine, the runner and all per-request state."""
    _state.update(engine=None, engine_failed=False, group=None, runner=None, paused=False, pause_t=0.0,
                  log_t=0.0, slot=0, pinned=None, err_printed=0, plan_pinned=None, plan_slot=0)
    _state["pending"] = collections.deque()
    _hist.clear()
    _prev.clear()
    _counts.clear()


# ---- recall-accel doctor: what this adapter needs (registry.check_symbols)
REQUIRED_SYMBOLS = (
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", ("__init__", "finish_requests", "execute_model")),
    ("vllm.v1.worker.gpu.states", "RequestState", ("remove_request", "add_request")),
    ("vllm.v1.worker.gpu.input_batch", "InputBatch", ()),
    ("vllm.v1.worker.gpu.spec_decode.speculator", "BaseSpeculator", ("propose",)),
    ("vllm.distributed.parallel_state", None, ("get_tp_group",)),
)
OPTIONAL_SYMBOLS = tuple((m, c, ("propose",)) for m, names in SPECULATORS.items() for c in names)
SIGNATURES = (("vllm.v1.worker.gpu.spec_decode.speculator", "BaseSpeculator.propose",
               ("self", "input_batch", "last_sampled", "dummy_run", "is_profile")),)
SOURCE_MARKERS = (("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", (
    "self.speculator.propose(", "self.req_states.last_sampled_tokens", "sampling_states.temperature")),)
