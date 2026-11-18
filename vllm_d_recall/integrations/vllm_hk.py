"""Hot-key aware speculative draft count (HK) for the vLLM V2 engine.

The scheduler decides how many draft tokens a request sends for verification from the length of
``request.spec_token_ids``: ``scheduler.py`` computes ``num_scheduled_spec_tokens`` from the length and
puts that many ids into ``scheduled_spec_decode_tokens``.  Recall knows which requests the record can
serve; this module shortens the draft list of the requests it cannot serve and keeps it long for the
ones it can.  The bookkeeping stays consistent because the scheduler, the async scheduler and the
rejection path all use the number of tokens actually scheduled.

EngineCore side only: no torch, no vllm import.  The worker side (see ``vllm_dspark``) writes a small
report on the ``ModelRunnerOutput`` under the attribute ``_xy_hk``; the scheduler reads it in
``update_from_output`` and marks the reported requests hot for ``VLLM_D_RECALL_HK_TTL`` steps.
"""
from __future__ import annotations

import json
import os
import time

from ..core import ENV_NEW, ENV_OLD, _env

_TRUE = ("1", "true", "yes", "on")
TAG = "[vllm-d-recall]"
HK_TAG = "[vllm-d-recall] hk(sched)"

ENV_ON = "VLLM_D_RECALL"
ENV_HK = "VLLM_D_RECALL_HK"
ENV_BASE = "VLLM_D_RECALL_HK_BASE"
ENV_MODE = "VLLM_D_RECALL_HK_MODE"
ENV_TTL = "VLLM_D_RECALL_HK_TTL"
ENV_LOG_S = "VLLM_D_RECALL_HK_LOG_S"
ENV_STATS = "VLLM_D_RECALL_HK_STATS_FILE"
ENV_DIR = "VLLM_D_RECALL_DIR"

MODES = ("batch", "request")
CLEAN_EVERY = 512  # steps between two sweeps of the hot table

_state = {
    "cfg": None,
    "hot": {},  # request id -> step the hot mark expires at
    "steps": 0,
    "kmax_steps": 0,
    "base_steps": 0,
    "reports": 0,
    "cand_reports": 0,
    "reqs_kmax": 0,
    "reqs_base": 0,
    "err_printed": 0,
    "log_t": 0.0,
}


# ------------------------------------------------------------------------------------------- config
def _plain(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is None:
        return default
    v = str(v).strip()
    return v if v else default


def _int_env(name: str, default: int) -> int:
    try:
        return int(_plain(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(_plain(name, str(default)))
    except ValueError:
        return default


def _read_config() -> dict:
    on = _env(ENV_NEW["on"], ENV_OLD["on"], "0").strip().lower() in _TRUE
    hk = _plain(ENV_HK, "0").strip().lower() in _TRUE
    mode = _plain(ENV_MODE, "batch").strip().lower()
    if mode not in MODES:
        mode = "batch"
    stats_file = _plain(ENV_STATS, "")
    if not stats_file:
        directory = _env(ENV_NEW["dir"], ENV_OLD["dir"], "")
        stats_file = os.path.join(directory, "hk_stats.json") if directory else ""
    return {
        "on": on,
        "hk": hk,
        "enabled": bool(on and hk),
        "base": max(0, _int_env(ENV_BASE, 5)),
        "mode": mode,
        "ttl": max(0, _int_env(ENV_TTL, 8)),
        "log_s": max(0.0, _float_env(ENV_LOG_S, 30.0)),
        "stats_file": stats_file,
    }


def config() -> dict:
    """The process-wide configuration; the environment is read once per process."""
    cfg = _state.get("cfg")
    if cfg is None:
        cfg = _state["cfg"] = _read_config()
    return cfg


def enabled() -> bool:
    return bool(config()["enabled"])


def reset() -> None:
    """Tests: forget the configuration, the hot table and the counters."""
    _state["cfg"] = None
    _state["hot"] = {}
    _state["steps"] = 0
    _state["kmax_steps"] = 0
    _state["base_steps"] = 0
    _state["reports"] = 0
    _state["cand_reports"] = 0
    _state["reqs_kmax"] = 0
    _state["reqs_base"] = 0
    _state["err_printed"] = 0
    _state["log_t"] = 0.0


# ------------------------------------------------------------------------------------------- pure logic
def decide(eligible_rids, hot_rids, mode: str, base: int, kmax: int) -> dict:
    """The number of draft tokens every eligible request sends this step.

    ``batch``   - one hot request lifts the whole batch to ``kmax`` (the attention metadata of one
                  decode batch must have a uniform length).
    ``request`` - every request is decided on its own.
    ``kmax <= base`` - nothing to gain, every request keeps ``base``.
    """
    base = int(base)
    kmax = int(kmax)
    if kmax <= base:
        return {rid: base for rid in eligible_rids}
    hot = set(hot_rids)
    if mode == "request":
        return {rid: (kmax if rid in hot else base) for rid in eligible_rids}
    use_kmax = any(rid in hot for rid in eligible_rids)
    n = kmax if use_kmax else base
    return {rid: n for rid in eligible_rids}


# ------------------------------------------------------------------------------------------- scheduler hooks
def ingest(sched, model_runner_output) -> int:
    """Read the report of the worker and mark the requests the record can serve as hot."""
    if not enabled():
        return 0
    report = getattr(model_runner_output, "_xy_hk", None)
    if not isinstance(report, dict) or report.get("v") != 1:
        return 0
    rids = report.get("rids") or []
    cand = report.get("cand") or {}
    if not isinstance(cand, dict):
        cand = {}
    ttl = config()["ttl"]
    step = int(getattr(sched, "current_step", 0) or 0)
    hot = _state["hot"]
    hits = 0
    for rid in rids:
        if rid in cand:
            hot[rid] = step + ttl
            hits += 1
        else:
            hot.pop(rid, None)
    _state["reports"] += 1
    _state["cand_reports"] += hits
    return hits


def _eligible(sched, step: int):
    """The running requests that may send drafts this step."""
    out = []
    for r in getattr(sched, "running", ()) or ():
        rid = getattr(r, "request_id", None)
        if rid is None:
            continue
        if step < int(getattr(r, "next_decode_eligible_step", 0) or 0):
            continue
        if getattr(r, "is_prefill_chunk", False):
            continue
        if getattr(r, "use_structured_output", False):
            continue
        if not getattr(r, "spec_token_ids", None):
            continue
        out.append((r, rid))
    return out


def pre_schedule(sched) -> dict:
    """Shorten the draft lists of the requests the record cannot serve."""
    if not enabled():
        return {}
    cfg = config()
    kmax = int(getattr(sched, "num_spec_tokens", 0) or 0)
    base = int(cfg["base"])
    if kmax <= base:
        return {}
    step = int(getattr(sched, "current_step", 0) or 0) + 1
    hot = _state["hot"]
    eligible = _eligible(sched, step)
    hot_rids = [rid for _, rid in eligible if hot.get(rid, -1) >= step]
    plan = decide([rid for _, rid in eligible], hot_rids, cfg["mode"], base, kmax)
    reqs_kmax = 0
    reqs_base = 0
    for r, rid in eligible:
        n = plan[rid]
        if n >= kmax:
            reqs_kmax += 1
        else:
            reqs_base += 1
        spec = r.spec_token_ids
        if len(spec) > n:
            # A new list: the placeholders are shared by every request of the batch.
            r.spec_token_ids = spec[:n]
    _state["steps"] += 1
    if reqs_kmax:
        _state["kmax_steps"] += 1
    if reqs_base:
        _state["base_steps"] += 1
    _state["reqs_kmax"] = reqs_kmax
    _state["reqs_base"] = reqs_base
    if _state["steps"] % CLEAN_EVERY == 0:
        _clean(sched)
    return plan


def _clean(sched) -> None:
    """Drop the hot marks of the requests the scheduler does not know any more."""
    requests = getattr(sched, "requests", None) or {}
    hot = _state["hot"]
    for rid in [rid for rid in hot if rid not in requests]:
        hot.pop(rid, None)


# ------------------------------------------------------------------------------------------- logging
def _log_error(what: str, e: BaseException) -> None:
    if _state["err_printed"] < 5:
        _state["err_printed"] += 1
        print("%s hk %s failed, this step keeps the model drafts: %r" % (TAG, what, e), flush=True)


def snapshot() -> dict:
    return {
        "steps": _state["steps"],
        "kmax_steps": _state["kmax_steps"],
        "base_steps": _state["base_steps"],
        "reqs_kmax": _state["reqs_kmax"],
        "reqs_base": _state["reqs_base"],
        "reports": _state["reports"],
        "cand_reports": _state["cand_reports"],
        "hot_now": len(_state["hot"]),
    }


def _write_stats(snap: dict) -> None:
    path = config().get("stats_file") or ""
    if not path:
        return
    directory = os.path.dirname(path)
    if not directory or not os.path.isdir(directory):
        return
    payload = dict(snap)
    payload["ts"] = time.time()
    payload["mode"] = config()["mode"]
    payload["base"] = config()["base"]
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def _maybe_log() -> None:
    cfg = config()
    log_s = cfg["log_s"]
    if log_s <= 0:
        return
    now = time.monotonic()
    if now - _state["log_t"] < log_s:
        return
    _state["log_t"] = now
    snap = snapshot()
    print("%s steps=%d kmax_steps=%d base_steps=%d reqs_kmax=%d reqs_base=%d reports=%d "
          "cand_reports=%d hot_now=%d"
          % (HK_TAG, snap["steps"], snap["kmax_steps"], snap["base_steps"], snap["reqs_kmax"],
             snap["reqs_base"], snap["reports"], snap["cand_reports"], snap["hot_now"]), flush=True)
    _write_stats(snap)


# ------------------------------------------------------------------------------------------- patching
def _mark(wrapper, orig) -> None:
    wrapper.__recall_hk_wrapped__ = True
    wrapper.__recall_wrapped__ = True
    wrapper.__wrapped__ = orig
    wrapper.__name__ = getattr(orig, "__name__", wrapper.__name__)
    wrapper.__qualname__ = getattr(orig, "__qualname__", wrapper.__qualname__)
    wrapper.__doc__ = getattr(orig, "__doc__", None)
    try:
        import inspect

        wrapper.__signature__ = inspect.signature(orig)
    except (TypeError, ValueError):  # pragma: no cover
        pass


def _patch_scheduler_module(module) -> None:
    """Wrap ``Scheduler.schedule`` and ``Scheduler.update_from_output`` (the base class is enough)."""
    cls = getattr(module, "Scheduler", None)
    if cls is None:
        return
    orig_schedule = getattr(cls, "schedule", None)
    if orig_schedule is not None and not getattr(orig_schedule, "__recall_hk_wrapped__", False):

        def schedule(self, *a, **kw):
            try:
                pre_schedule(self)
            except Exception as e:  # noqa: BLE001  never break scheduling
                _log_error("pre_schedule", e)
            try:
                _maybe_log()
            except Exception as e:  # noqa: BLE001
                _log_error("log", e)
            return orig_schedule(self, *a, **kw)

        _mark(schedule, orig_schedule)
        cls.schedule = schedule

    orig_update = getattr(cls, "update_from_output", None)
    if orig_update is not None and not getattr(orig_update, "__recall_hk_wrapped__", False):

        def update_from_output(self, *a, **kw):
            try:
                mro = kw.get("model_runner_output")
                if mro is None and len(a) > 1:
                    mro = a[1]
                ingest(self, mro)
            except Exception as e:  # noqa: BLE001  never break serving
                _log_error("ingest", e)
            try:
                _maybe_log()
            except Exception as e:  # noqa: BLE001
                _log_error("log", e)
            return orig_update(self, *a, **kw)

        _mark(update_from_output, orig_update)
        cls.update_from_output = update_from_output

    print("%s patched %s.Scheduler (hk)" % (TAG, getattr(module, "__name__", "scheduler")), flush=True)
