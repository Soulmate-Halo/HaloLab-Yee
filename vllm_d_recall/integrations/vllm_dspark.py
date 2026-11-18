"""Recall copy drafts inside vLLM's V2 GPU model runner (``vllm/v1/worker/gpu``), DSpark / DFlash speculators.

Where it sits (vLLM 0.28.1rc1, V2 runner):
  * ``GPUModelRunner.execute_model`` samples, calls ``postprocess_sampled`` (which writes the accepted ids into
    ``req_states.all_token_ids`` - a UVA buffer in pinned host memory - and the true lengths into
    ``req_states.total_len.gpu``), then ``self.speculator.propose(...)``, and stores the returned
    ``[num_reqs, num_speculative_steps]`` drafts in ``req_states.draft_tokens``; the next step verifies them.
  * ``DSparkSpeculator`` subclasses ``DFlashSpeculator`` and does not override ``propose``, so wrapping
    ``DFlashSpeculator.propose`` covers both methods.

Per participating step (all requests greedy, no structured output, not a dummy / profile / graph capture run):
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
from . import shadow
from . import vllm_hk
from . import vllm_kad

LAG_STEPS = 3  # CPU snapshots may trail the step being launched by up to this many steps
MAX_BS = 512
RING = 16  # pinned snapshot slots in flight
ENV_NONGREEDY = "VLLM_D_RECALL_NONGREEDY"  # r648 S4-4.1a: "1" lets temperature > 0 rows take copies (default off)
_TRUE = ("1", "true", "yes", "on")
TAG = "[vllm-d-recall]"

PROPOSE_PARAMS = (
    "self", "input_batch", "attn_metadata", "slot_mappings", "last_hidden_states", "aux_hidden_states",
    "num_sampled", "num_rejected", "last_sampled", "next_prefill_tokens", "temperature", "seeds", "dp_sync",
    "dummy_run", "skip_attn_for_dummy_run", "mm_inputs", "is_profile",
)

_state = {
    "engine": None, "engine_failed": False, "group": None, "runner": None, "paused": False, "pause_t": 0.0,
    "log_t": 0.0, "installed": False, "pending": collections.deque(), "slot": 0, "pinned": None,
    "err_printed": 0, "hk_report": None,
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
          "record_ids=%d streams=%d pending=%d paused=%s errors=%d hk_reports=%d hk_cand_rows=%d"
          % (TAG, _counts["steps"], _counts["rows"], _counts["plan_rows"], _counts["used"], _counts["fallback"],
             _counts["own"], st["landed_tokens"], st["offers"], st["ids"], st["active_streams"],
             len(_state["pending"]), _state["paused"], _counts["errors"], _counts["hk_reports"],
             _counts["hk_cand_rows"]), flush=True)


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


def participates(runner, input_batch, draft_tokens, dummy_run: bool, is_profile: bool) -> bool:
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
    if _all_greedy(runner, idx_np):
        return True
    # r648 S4-4.1a: with VLLM_D_RECALL_NONGREEDY=1 rows with temperature > 0 take copies too; _on_propose then
    # turns the overwritten positions' cached draft distribution into a point mass (see _fix_pointmass).
    return _nongreedy_on() and getattr(runner, "sampler", None) is not None


def _nongreedy_on() -> bool:
    return os.environ.get(ENV_NONGREEDY, "0").strip().lower() in _TRUE


def _fix_pointmass(spec, runner, idx, idx_np, draft_tokens, ok) -> None:
    """Make copied positions of temperature > 0 rows verifiable: probabilistic drafting caches the draft
    logits (pre-temperature, rows addressed by req_state_idx, read back by the rejection sampler), but a copied
    id was not drawn from them. Each copied position becomes a point mass on the copied id (0 there, -inf
    elsewhere), which is exactly the proposal that produced it, so standard and block verification stay
    lossless (.diagnostics/r648/s5: biased without this, exact with it). Greedy drafting caches no logits and
    already treats the buffer as a point mass, so there is nothing to do then. Host decisions use only state
    every TP rank shares (temperatures), device work needs no sync."""
    import torch

    dl = getattr(spec, "draft_logits", None)
    if dl is None or not torch.is_tensor(dl) or dl.dim() != 3:
        return
    sampler = getattr(runner, "sampler", None)
    if sampler is None:
        return
    temps = np.asarray(sampler.sampling_states.temperature.np)[idx_np]
    hot = np.nonzero(temps > 0.0)[0]
    if hot.size == 0:
        return
    K = min(int(dl.shape[1]), int(draft_tokens.shape[1]))
    b = torch.from_numpy(hot.astype(np.int64)).to(device=dl.device)
    rows = idx.to(device=dl.device, dtype=torch.int64).index_select(0, b)
    cur = dl.index_select(0, rows)
    tok = draft_tokens.index_select(0, b.to(draft_tokens.device))[:, :K].to(device=dl.device, dtype=torch.int64)
    pm = torch.full_like(cur[:, :K], float("-inf"))
    pm.scatter_(2, tok.unsqueeze(-1), 0.0)
    okh = ok.index_select(0, b.to(ok.device)).to(dl.device).view(-1, 1, 1)
    cur[:, :K] = torch.where(okh, pm, cur[:, :K])
    dl.index_copy_(0, rows, cur)
    _counts["pointmass_candidates"] += int(hot.size)


def _merge(spec, runner, idx, idx_np, plan, true_len, anchors, draft_tokens, gamma: int, conf):
    """device_merge plus, with VLLM_D_RECALL_NONGREEDY=1, the point-mass fix. If the fix fails the step's
    copies are dropped (drafts restored), so a sampled row never carries a copied id without its point mass."""
    backup = draft_tokens.clone() if _nongreedy_on() else None
    ok = device_merge(plan, true_len, anchors, draft_tokens, gamma, confidence=conf)
    if backup is not None:
        try:
            _fix_pointmass(spec, runner, idx, idx_np, draft_tokens, ok)
        except Exception as e:  # noqa: BLE001
            _log_error("pointmass error", e)
            draft_tokens.copy_(backup)
    return ok


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


def _write_hk_report(rids, entries) -> None:
    """Tell the scheduler which requests the record can serve this step (HK / KAD).

    ``entries`` are the plan rows ``(flag, L, W, real)``: a row with a flag has a candidate, ``real``
    is how many tokens were really copied.
    """
    if not (vllm_hk.enabled() or vllm_kad.enabled()):
        return
    try:
        cand = {}
        if entries:
            for i, rid in enumerate(rids):
                entry = entries[i] if i < len(entries) else None
                if entry and int(entry[0]):
                    cand[rid] = int(entry[3])
        _state["hk_report"] = {"v": 1, "rids": list(rids), "cand": cand}
        _counts["hk_reports"] += 1
        _counts["hk_cand_rows"] += len(cand)
    except Exception as e:  # noqa: BLE001  never break serving
        _log_error("hk report", e)


# ------------------------------------------------------------------------------------------- shadow
# The worker-side shadow instrument: it records what the recall did to the drafts of a step.  It is
# off by default (VLLM_D_RECALL_SHADOW) and then it costs nothing at all: no tensor, no copy, no
# record.  When it is on it never synchronises the device: the probabilities are computed on the
# GPU, copied into pinned host memory without blocking and read on the CPU only once the event of
# the copy has completed (the same pattern as the snapshot queue above).
SHADOW_ROLE = "worker"
SHADOW_RING = 16  # the pinned slots in flight
SHADOW_QUEUE_MAX = 8  # the queue length that makes the next step wait for the oldest entry
SHADOW_KMAX = 64  # the max number of draft positions of one row the buffers hold
SHADOW_CHUNK_ELEMS = 1 << 21  # the max number of elements of one probability chunk (8 MB)

_shadow_state = {
    "ok": True,
    "logged": False,
    "pending": collections.deque(),
    "free": collections.deque(range(SHADOW_RING)),
    "wstep": 0,
    "qm": None,
    "qx": None,
    "okv": None,
    "tlv": None,
    "kmax": 0,
}


def _shadow_fail(e: BaseException) -> None:
    """A broken shadow instrument must not stop the worker: log once and switch it off."""
    _shadow_state["ok"] = False
    if not _shadow_state["logged"]:
        _shadow_state["logged"] = True
        print("%s worker shadow instrumentation failed, the shadow recording is off: %r" % (TAG, e),
              flush=True)


def _shadow_on() -> bool:
    """The worker shadow is on (the switch is off by default: then nothing is computed at all)."""
    if not _shadow_state["ok"]:
        return False
    try:
        return bool(shadow.enabled())
    except Exception:  # noqa: BLE001  a broken switch means no shadow
        return False


def _shadow_reset() -> None:
    """Tests: forget the queue, the counters and the pinned buffers."""
    _shadow_state.update(ok=True, logged=False, pending=collections.deque(),
                         free=collections.deque(range(SHADOW_RING)), wstep=0, qm=None, qx=None,
                         okv=None, tlv=None, kmax=0)


def _shadow_probs(spec, runner, idx, idx_np, draft_tokens, gamma: int):
    """The draft probabilities of this step, computed on the device (never synchronised).

    ``qmax[b, j]`` is the largest probability of the draft distribution of the position j of the row
    b and ``qx[b, j]`` the probability of the drafted token itself; both are the softmax of the
    cached draft logits divided by the temperature of the row (a temperature of 0 counts as 1.0).
    Returns (qmax, qx, k); k is 0 when there is no cached draft distribution.
    """
    import torch

    dl = getattr(spec, "draft_logits", None)
    if dl is None or not torch.is_tensor(dl) or dl.dim() != 3:
        return None, None, 0
    n = int(draft_tokens.shape[0])
    k = min(int(gamma), int(dl.shape[1]))
    if n <= 0 or k <= 0:
        return None, None, 0
    dev = dl.device
    v = int(dl.shape[2])
    rows = idx.to(device=dev, dtype=torch.int64)
    temps = None
    sampler = getattr(runner, "sampler", None)
    if sampler is not None:
        try:
            temps = np.asarray(sampler.sampling_states.temperature.np)[idx_np]
        except Exception:  # noqa: BLE001  no temperature: the raw logits are used
            temps = None
    if temps is None:
        temps = np.ones(n, dtype=np.float32)
    t = torch.from_numpy(np.ascontiguousarray(temps, dtype=np.float32).reshape(-1)).to(device=dev)
    t = torch.where(t > 0.0, t, torch.ones_like(t))
    tok = torch.clamp(draft_tokens[:, :k].to(device=dev, dtype=torch.int64), 0, max(0, v - 1))
    qmax = torch.empty((n, k), dtype=torch.float32, device=dev)
    qx = torch.empty((n, k), dtype=torch.float32, device=dev)
    rb = max(1, SHADOW_CHUNK_ELEMS // max(1, k * v))  # the rows of one chunk
    for r0 in range(0, n, rb):
        r1 = min(n, r0 + rb)
        cur = dl[:, :k, :].index_select(0, rows[r0:r1]).to(torch.float32)
        scaled = cur / t[r0:r1].view(r1 - r0, 1, 1)
        lse = torch.logsumexp(scaled, dim=-1)
        mx = scaled.max(dim=-1).values
        tl = scaled.gather(2, tok[r0:r1].unsqueeze(-1)).squeeze(-1)
        qmax[r0:r1] = (mx - lse).exp()
        qx[r0:r1] = (tl - lse).exp()
    return qmax, qx, k


def _shadow_collect(spec, runner, idx, idx_np, draft_tokens, gamma: int, ok, true_len, rids) -> None:
    """Queue the shadow record of one step (never raises, never synchronises the device).

    The probabilities, the recall mask and the true lengths are copied into pinned host memory
    without blocking; the event of the copy goes into the queue with them and the CPU reads them
    only after that event has completed.
    """
    if not _shadow_on():
        return
    slot = None
    try:
        import torch

        n = int(draft_tokens.shape[0])
        if n <= 0 or n > MAX_BS:
            return
        dev = draft_tokens.device
        qmax, qx, k = _shadow_probs(spec, runner, idx, idx_np, draft_tokens, gamma)
        if k > SHADOW_KMAX:  # the buffers cannot hold it: the probabilities stay empty
            qmax, qx, k = None, None, 0
        pin = bool(torch.cuda.is_available() and dev.type == "cuda")
        rows = MAX_BS
        if _shadow_state["okv"] is None:
            _shadow_state["okv"] = torch.zeros(SHADOW_RING * rows, dtype=torch.int32, pin_memory=pin)
            _shadow_state["tlv"] = torch.zeros(SHADOW_RING * rows, dtype=torch.int32, pin_memory=pin)
        if k > 0 and (_shadow_state["qm"] is None or _shadow_state["kmax"] != k):
            # the width of the buffers is the number of draft positions; a change reallocates them
            # (the entries in flight keep their own views of the old buffers)
            _shadow_state["qm"] = torch.zeros(SHADOW_RING * rows * k, dtype=torch.float32, pin_memory=pin)
            _shadow_state["qx"] = torch.zeros(SHADOW_RING * rows * k, dtype=torch.float32, pin_memory=pin)
            _shadow_state["kmax"] = k
        free = _shadow_state["free"]
        if not free:
            return
        slot = free.popleft()
        qm_slot = None
        qx_slot = None
        if k > 0:
            base = slot * rows * k
            qm_slot = _shadow_state["qm"][base:base + n * k].view(n, k)
            qx_slot = _shadow_state["qx"][base:base + n * k].view(n, k)
            qm_slot.copy_(qmax, non_blocking=True)
            qx_slot.copy_(qx, non_blocking=True)
        ok_slot = _shadow_state["okv"][slot * rows:slot * rows + n]
        tl_slot = _shadow_state["tlv"][slot * rows:slot * rows + n]
        oki = ok.to(torch.int32) if ok is not None else torch.zeros(n, dtype=torch.int32, device=dev)
        ok_slot.copy_(oki, non_blocking=True)
        tl_slot.copy_(true_len.to(torch.int32), non_blocking=True)
        ev = _event_factory(dev)
        if ev is not None:
            ev.record()
        wstep = _shadow_state["wstep"]
        _shadow_state["wstep"] = wstep + 1
        _shadow_state["pending"].append((ev, wstep, list(rids), int(gamma), int(k), slot,
                                         qm_slot, qx_slot, ok_slot, tl_slot))
        slot = None
    except Exception as e:  # noqa: BLE001  the shadow must never stop the worker
        if slot is not None:
            _shadow_state["free"].append(slot)
        _shadow_fail(e)


def _shadow_drain(block: bool = False) -> int:
    """rank0: read the finished entries of the shadow queue on the CPU (never raises).

    An entry whose event has not completed stays in the queue; with ``block`` the oldest entry is
    waited for.  Every row of an entry becomes one ``wreq`` record and the entry itself one
    ``wstep`` record.
    """
    if not _shadow_on():
        return 0
    try:
        rec = shadow.get_recorder(SHADOW_ROLE, 0)
        pend = _shadow_state["pending"]
        done = 0
        while pend:
            ev, wstep, rids, gamma, k, slot, qm_slot, qx_slot, ok_slot, tl_slot = pend[0]
            if ev is not None:
                if block:
                    ev.synchronize()
                elif not ev.query():
                    break
            pend.popleft()
            qm = qm_slot.numpy() if qm_slot is not None else None
            qx = qx_slot.numpy() if qx_slot is not None else None
            okv = ok_slot.numpy()
            tlv = tl_slot.numpy()
            for i, rid in enumerate(rids):
                rec.record(shadow.KIND_WREQ, {
                    "wstep": int(wstep),
                    "rid": str(rid),
                    "pos": int(tlv[i]),
                    "w": int(gamma),
                    "rec": int((1 << int(gamma)) - 1) if int(okv[i]) else 0,
                    "qmax": [round(float(x), 4) for x in qm[i]] if qm is not None else [],
                    "qx": [round(float(x), 4) for x in qx[i]] if qx is not None else [],
                    "conf": [],
                })
            rec.record(shadow.KIND_WSTEP, {
                "wstep": int(wstep),
                "n": len(rids),
                "bt": -1,
                "fwd_ms": -1.0,
                "drf_ms": -1.0,
            })
            _shadow_state["free"].append(slot)
            done += 1
        return done
    except Exception as e:  # noqa: BLE001  the shadow must never stop the worker
        _shadow_fail(e)
        return 0


def _shadow_flush() -> int:
    """Wait for the oldest entry, read the queue and flush the recorder to disk (tests)."""
    done = _shadow_drain(block=True)
    try:
        shadow.get_recorder(SHADOW_ROLE, 0).flush()
    except Exception:  # noqa: BLE001  the flush must never raise
        pass
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
    if rank0 and _shadow_on():
        # the queue is read at the start of the step: a long queue waits for its oldest entry
        try:
            _shadow_drain(block=len(_shadow_state["pending"]) > SHADOW_QUEUE_MAX)
        except Exception as e:  # noqa: BLE001  the shadow must never stop the worker
            _shadow_fail(e)
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
                _write_hk_report(rids, entries)
            except Exception as e:  # noqa: BLE001  never skip the broadcast below
                _log_error("plan error", e)
                entries = [(0, 0, None, 0)] * n
                plan.zero_()
                _write_hk_report(rids, entries)
        else:
            _write_hk_report(rids, None)
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
    ok = _merge(spec, runner, idx, idx_np, plan, true_len, anchors, draft_tokens, gamma, conf)
    if rank0:
        try:
            _shadow_collect(spec, runner, idx, idx_np, draft_tokens, gamma, ok, true_len, rids)
        except Exception as e:  # noqa: BLE001  the shadow must never stop the worker
            _shadow_fail(e)
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
        out = inner(*a, **kw)
        try:
            runner = _state["runner"]
            ib = arg(a, kw, "input_batch")
            dummy = bool(arg(a, kw, "dummy_run", False)) or bool(arg(a, kw, "is_profile", False))
            if participates(runner, ib, out, dummy, False):
                _on_propose(a[0], runner, ib, out, arg(a, kw, "last_sampled"), int(out.shape[1]))
            elif not dummy and ib is not None and _state["engine"] is not None:
                rids = list(ib.req_ids[:int(ib.num_reqs)])
                forget_prev(_state["engine"], rids)
                _write_hk_report(rids, None)
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


def _patch_speculator_module(module) -> None:
    cls = module.DFlashSpeculator
    if getattr(cls.propose, "__recall_wrapped__", False):
        return
    orig = cls.propose
    if vllm_kad.enabled():
        # The KAD wrapper sits under the recall wrapper: it only changes how many drafts the draft
        # head writes, the recall wrapper still does the copying.
        try:
            from . import vllm_kad_worker

            if not getattr(orig, "__recall_kad_wrapped__", False):
                orig = vllm_kad_worker.make_varlen_propose(orig)
        except Exception as e:  # noqa: BLE001  the native propose stays in place
            print("%s KAD propose wrapper failed, the native propose stays: %r" % (TAG, e), flush=True)
    cls.propose = _make_propose(orig)
    print("%s patched %s.DFlashSpeculator.propose (covers DSpark)" % (TAG, module.__name__), flush=True)


def _patch_ar_module(module) -> None:
    """Wrap ``AutoRegressiveSpeculator.propose`` (eagle, eagle3, mtp, gemma4 mtp): KAD steps path."""
    if not vllm_kad.enabled():
        return
    cls = getattr(module, "AutoRegressiveSpeculator", None)
    if cls is None:
        return
    if "propose" not in cls.__dict__:
        return  # the class inherits its propose: nothing to wrap here
    orig = cls.__dict__["propose"]
    if getattr(orig, "__recall_kad_wrapped__", False):
        return
    try:
        from . import vllm_kad_worker

        cls.propose = vllm_kad_worker.make_steps_propose(orig)
    except Exception as e:  # noqa: BLE001  the native propose stays in place
        print("%s KAD steps wrapper failed, the native propose stays: %r" % (TAG, e), flush=True)
        return
    print("%s patched %s.AutoRegressiveSpeculator.propose (kad steps)" % (TAG, module.__name__),
          flush=True)


def _patch_draft_model_module(module) -> None:
    """Wrap ``PlainDraftModelSpeculator.propose``: KAD steps path."""
    if not vllm_kad.enabled():
        return
    cls = getattr(module, "PlainDraftModelSpeculator", None)
    if cls is None:
        return
    if "propose" not in cls.__dict__:
        return
    orig = cls.__dict__["propose"]
    if getattr(orig, "__recall_kad_wrapped__", False):
        return
    try:
        from . import vllm_kad_worker

        cls.propose = vllm_kad_worker.make_steps_propose(orig)
    except Exception as e:  # noqa: BLE001  the native propose stays in place
        print("%s KAD steps wrapper failed, the native propose stays: %r" % (TAG, e), flush=True)
        return
    print("%s patched %s.PlainDraftModelSpeculator.propose (kad steps)" % (TAG, module.__name__),
          flush=True)


def _patch_runner_module(module) -> None:
    cls = module.GPUModelRunner
    if getattr(cls.__init__, "__recall_wrapped__", False):
        return
    orig_init = cls.__init__
    orig_finish = cls.finish_requests
    orig_sample = getattr(cls, "sample_tokens", None)
    orig_execute = getattr(cls, "execute_model", None)

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        if getattr(self, "speculator", None) is not None:
            _state["runner"] = self
            print("%s runner registered (speculator %s)" % (TAG, type(self.speculator).__name__), flush=True)
            try:
                if vllm_kad.enabled():
                    spec = self.speculator
                    head = type(spec).__name__
                    method = getattr(spec, "method", "")
                    k = int(getattr(spec, "num_speculative_steps", 0) or 0)
                    try:
                        from . import vllm_kad_worker

                        path = vllm_kad_worker.kad_path_of(spec)
                    except Exception:  # noqa: BLE001
                        path = "native"
                    print("[vllm-d-recall] kad(worker) head=%s method=%s k=%d path=%s"
                          % (head, method, k, path), flush=True)
                    if k != 12:
                        print("[vllm-d-recall] kad(worker) WARNING launch num_speculative_tokens=%d,"
                              " KAD needs 12 (per-request width is capped at min(12, k))" % (k,), flush=True)
            except Exception as e:  # noqa: BLE001  the log must not break the launch
                print("%s kad(worker) head log failed: %r" % (TAG, e), flush=True)

    def sample_tokens(self, *a, **kw):
        if self is _state["runner"]:
            _state["hk_report"] = None
        try:
            out = orig_sample(self, *a, **kw)
        except Exception as e:  # noqa: BLE001  never break serving
            _log_error("sample_tokens hook error", e)
            raise
        try:
            report = _state["hk_report"]
            if (vllm_hk.enabled() or vllm_kad.enabled()) and self is _state["runner"] \
                    and report and out is not None:
                target = getattr(out, "model_runner_output", out)
                setattr(target, "_xy_hk", report)
                _state["hk_report"] = None
        except Exception as e:  # noqa: BLE001  never break serving
            _log_error("hk attach error", e)
        return out

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

    if orig_execute is not None and vllm_kad.enabled() \
            and not getattr(orig_execute, "__recall_kad_wrapped__", False):

        def execute_model(self, *a, **kw):
            try:
                from . import vllm_kad_worker

                so = kw.get("scheduler_output")
                if so is None and len(a) > 0:
                    so = a[0]
                vllm_kad_worker.on_execute_model(so)
            except Exception as e:  # noqa: BLE001  never break serving
                _log_error("execute_model hook error", e)
            return orig_execute(self, *a, **kw)

        execute_model.__wrapped__ = orig_execute
        execute_model.__recall_kad_wrapped__ = True
        try:
            import inspect

            execute_model.__signature__ = inspect.signature(orig_execute)
        except (TypeError, ValueError):  # pragma: no cover
            pass
        cls.execute_model = execute_model

    __init__.__recall_wrapped__ = True
    __init__.__wrapped__ = orig_init
    finish_requests.__wrapped__ = orig_finish
    cls.__init__ = __init__
    cls.finish_requests = finish_requests
    if orig_sample is not None and not getattr(orig_sample, "__recall_hk_wrapped__", False):
        sample_tokens.__wrapped__ = orig_sample
        sample_tokens.__recall_hk_wrapped__ = True
        try:
            import inspect

            sample_tokens.__signature__ = inspect.signature(orig_sample)
        except (TypeError, ValueError):  # pragma: no cover
            pass
        cls.sample_tokens = sample_tokens
    print("%s patched %s.GPUModelRunner" % (TAG, module.__name__), flush=True)


TARGETS = {
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator": _patch_speculator_module,
    "vllm.v1.worker.gpu.model_runner": _patch_runner_module,
}

SCHEDULER_TARGET = "vllm.v1.core.sched.scheduler"
AR_TARGET = "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator"
DRAFT_MODEL_TARGET = "vllm.v1.worker.gpu.spec_decode.draft_model.speculator"


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
    if vllm_kad.enabled():
        targets[SCHEDULER_TARGET] = vllm_kad._patch_scheduler_module
        targets[AR_TARGET] = _patch_ar_module
        targets[DRAFT_MODEL_TARGET] = _patch_draft_model_module
        if vllm_hk.enabled():
            print("%s KAD and HK are both on: only the KAD scheduler patch is installed" % TAG,
                  flush=True)
    elif vllm_hk.enabled():
        targets[SCHEDULER_TARGET] = vllm_hk._patch_scheduler_module
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
                  log_t=0.0, slot=0, pinned=None, err_printed=0, plan_pinned=None, plan_slot=0,
                  hk_report=None)
    vllm_hk.reset()
    vllm_kad.reset()
    _shadow_reset()
    _state["pending"] = collections.deque()
    _hist.clear()
    _prev.clear()
    _counts.clear()
