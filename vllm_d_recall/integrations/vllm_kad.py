"""Per-request adaptive draft width (KAD) for the vLLM V2 engine.

The scheduler decides how many draft tokens a request sends for verification from the length of
``request.spec_token_ids`` (see ``vllm_hk`` for the 1.1.0 version of that hook).  KAD keeps the
mechanism but makes the width a per-request decision.  It works with any vLLM draft head, not
only the DSpark one:

  * ``v_i`` - the number of draft tokens request ``i`` sends for verification this step.  It is
    enforced by shortening ``request.spec_token_ids`` before the scheduler runs; the scheduler, the
    async scheduler and the rejection path all account for the number of tokens actually scheduled.
  * ``w_i`` - the number of draft tokens the draft head produced for request ``i`` at the end of
    the step.  It is written into the ``SchedulerOutput`` under the attribute ``_xy_kw`` and read
    by the worker in ``GPUModelRunner.execute_model``.

The invariant is ``v_i(t+1) <= w_i(t)``: the width of the next step never exceeds the width the
worker really produced.  The worker falls back to the native propose (which produces ``K_max``
drafts) whenever it cannot honour the request, so the scheduler may always assume ``K_max`` for a
request it does not know.  A request with structured output is never shortened (``pre_schedule``
skips it), so its width is ``w = K_max`` and the invariant holds for it as well.

The strategy is a JSON file that is reloaded while the server runs (``off`` / ``fixed`` /
``ladder`` / ``threshold``).  The ladder keeps one state per request and moves the width up when
the request keeps its drafts and down when it loses them.  The threshold strategy decides the
width of the whole batch at once from the conditional acceptance rates of the requests.  A
process that has not read a valid policy file yet uses the ladder default (``base`` 5, ``kmin``
5, ``kmax`` 12).

EngineCore side only: no torch, no vllm import.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np

from ..core import ENV_NEW, ENV_OLD, _env
from . import shadow
from .vllm_hk import _eligible

_TRUE = ("1", "true", "yes", "on")
TAG = "[vllm-d-recall]"
KAD_TAG = "[vllm-d-recall] kad(sched)"

# The attribute of the SchedulerOutput that carries the widths to the workers.
ATTR_SO = "_xy_kw"

# The attribute of the SchedulerOutput that carries the step number of the shadow instrument.
ATTR_SHADOW_STEP = "_xy_shadow_step"
# The role of the shadow recorder inside the scheduler process.
SHADOW_ROLE = "sched"

ENV_KAD = "VLLM_D_RECALL_KAD"
ENV_POLICY = "VLLM_D_RECALL_KAD_POLICY"
ENV_STATS = "VLLM_D_RECALL_KAD_STATS_FILE"
ENV_LOG_S = "VLLM_D_RECALL_KAD_LOG_S"
ENV_RELOAD_S = "VLLM_D_RECALL_KAD_RELOAD_S"

CLEAN_EVERY = 512  # steps between two sweeps of the per-request tables
MAX_ERRORS = 5

MODES = ("off", "fixed", "ladder", "threshold")

_state = {
    "cfg": None,
    "policy": None,
    "policy_sha256": "",
    "policy_stamp": None,
    "last_stat_t": 0.0,
    "w_prev": {},  # request id -> the width of the last step
    "ladder": {},  # request id -> {"cur": int, "streak": int, "hot_until": int}
    "thr": {},  # request id -> {"g": [12 floats], "bucket": str}
    "thr_fail": False,
    "steps": 0,
    "mixed_steps": 0,
    "w_hist": {},  # width -> number of requests
    "v_steps": 0,
    "sum_v": 0,
    "sum_a": 0,
    "acc_by_v": {},  # width -> [steps, accepted]
    "ingest_steps": 0,
    "kmax_run": 0,
    "err_printed": 0,
    "errors": 0,
    "log_t": 0.0,
    "shadow_ok": True,
    "shadow_logged": False,
    "shadow_t": None,
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
    kad = _plain(ENV_KAD, "0").strip().lower() in _TRUE
    directory = _env(ENV_NEW["dir"], ENV_OLD["dir"], "")
    policy_file = _plain(ENV_POLICY, "")
    if not policy_file:
        policy_file = os.path.join(directory, "kad_policy.json") if directory else ""
    stats_file = _plain(ENV_STATS, "")
    if not stats_file:
        stats_file = os.path.join(directory, "kad_stats.json") if directory else ""
    return {
        "on": on,
        "kad": kad,
        "enabled": bool(on and kad),
        "policy_file": policy_file,
        "stats_file": stats_file,
        "log_s": max(0.0, _float_env(ENV_LOG_S, 30.0)),
        "reload_s": max(0.0, _float_env(ENV_RELOAD_S, 1.0)),
    }


def config() -> dict:
    """The process-wide configuration; the environment is read once per process."""
    cfg = _state.get("cfg")
    if cfg is None:
        cfg = _state["cfg"] = _read_config()
    return cfg


def enabled() -> bool:
    """KAD is on when the recall switch and the KAD switch are both on."""
    return bool(config()["enabled"])


def reset() -> None:
    """Tests: forget the configuration, the policy, the per-request tables and the counters."""
    _state["cfg"] = None
    _state["policy"] = None
    _state["policy_sha256"] = ""
    _state["policy_stamp"] = None
    _state["last_stat_t"] = 0.0
    _state["w_prev"] = {}
    _state["ladder"] = {}
    _state["thr"] = {}
    _state["thr_fail"] = False
    _state["steps"] = 0
    _state["mixed_steps"] = 0
    _state["w_hist"] = {}
    _state["v_steps"] = 0
    _state["sum_v"] = 0
    _state["sum_a"] = 0
    _state["acc_by_v"] = {}
    _state["ingest_steps"] = 0
    _state["kmax_run"] = 0
    _state["err_printed"] = 0
    _state["errors"] = 0
    _state["log_t"] = 0.0
    _state["shadow_ok"] = True
    _state["shadow_logged"] = False
    _state["shadow_t"] = None


# ------------------------------------------------------------------------------------------- policy
def _policy_int(obj: dict, name: str, default: int, minimum: int) -> int:
    if name not in obj:
        return int(default)
    value = obj[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("the policy field %r must be an integer" % (name,))
    return max(int(minimum), int(value))


def parse_policy(obj) -> dict:
    """The normalized policy; raises ValueError when the object is not a valid policy."""
    if not isinstance(obj, dict):
        raise ValueError("the policy must be a JSON object")
    mode = obj.get("mode", "fixed")
    if not isinstance(mode, str):
        raise ValueError("the policy field 'mode' must be a string")
    mode = mode.strip().lower()
    if mode not in MODES:
        raise ValueError("unknown policy mode %r" % (mode,))
    down = obj.get("down", "accepted")
    if not isinstance(down, str):
        raise ValueError("the policy field 'down' must be a string")
    down = down.strip().lower()
    if down not in ("accepted", "step"):
        raise ValueError("unknown policy 'down' %r" % (down,))
    pol = {
        "mode": mode,
        "fixed": _policy_int(obj, "fixed", 5, 1),
        "base": _policy_int(obj, "base", 5, 1),
        "kmin": _policy_int(obj, "kmin", 5, 1),
        "kmax": _policy_int(obj, "kmax", 12, 1),
        "up_after": _policy_int(obj, "up_after", 1, 1),
        "up_step": _policy_int(obj, "up_step", 2, 0),
        "down": down,
        "down_slack": _policy_int(obj, "down_slack", 1, 0),
        "down_step": _policy_int(obj, "down_step", 2, 0),
        "hot_k": _policy_int(obj, "hot_k", 12, 1),
        "hot_ttl": _policy_int(obj, "hot_ttl", 8, 0),
    }
    if pol["kmin"] > pol["kmax"]:
        pol["kmin"] = pol["kmax"]
    pol["base"] = max(pol["kmin"], min(pol["base"], pol["kmax"]))
    if mode == "threshold":
        pol.update(_thr_policy_fields(obj))
    return pol


def _default_policy() -> dict:
    """The policy of a process that has no valid policy file yet: the Yee 3.0.0 ladder default."""
    return parse_policy({"mode": "ladder", "base": 5, "kmin": 5, "kmax": 12, "up_after": 1,
                          "up_step": 2, "down": "accepted", "down_slack": 1, "down_step": 2,
                          "hot_k": 12, "hot_ttl": 8})


def _log_error(what: str, e: BaseException) -> None:
    _state["errors"] += 1
    if _state["err_printed"] < MAX_ERRORS:
        _state["err_printed"] += 1
        print("%s kad %s failed, the step keeps the model drafts: %r" % (TAG, what, e), flush=True)


def _maybe_reload(path: str) -> None:
    """Reload the policy file when it changed; a broken file keeps the last valid policy."""
    cfg = config()
    now = time.monotonic()
    if now - _state["last_stat_t"] < cfg["reload_s"]:
        return
    _state["last_stat_t"] = now
    try:
        info = os.stat(path)
    except OSError:
        return  # no file yet: keep the last valid policy
    stamp = (int(info.st_mtime_ns), int(info.st_size))
    if stamp == _state["policy_stamp"]:
        return
    _state["policy_stamp"] = stamp
    try:
        with open(path, "rb") as f:
            data = f.read()
        pol = parse_policy(json.loads(data.decode("utf-8")))
    except Exception as e:  # noqa: BLE001  a broken policy must not stop the server
        _log_error("policy load", e)
        return
    sha = hashlib.sha256(data).hexdigest()
    _state["policy"] = pol
    _state["policy_sha256"] = sha
    print("%s policy loaded sha=%s %s" % (KAD_TAG, sha[:12], json.dumps(pol, sort_keys=True)), flush=True)
    _write_stats(snapshot())


def _current_policy() -> dict:
    """The policy in force; the file is reloaded at most every ``RELOAD_S`` seconds."""
    if not enabled():
        pol = _state["policy"]
        return pol if pol is not None else _default_policy()
    path = config()["policy_file"]
    if path:
        try:
            _maybe_reload(path)
        except Exception as e:  # noqa: BLE001  never break scheduling
            _log_error("policy reload", e)
    pol = _state["policy"]
    if pol is None:
        pol = _state["policy"] = _default_policy()
    return pol


# ------------------------------------------------------------------------------------------- pure logic
def ladder_update(st: dict, v: int, a: int, pol: dict) -> None:
    """Move the width of one request up or down; ``st`` is changed in place."""
    v = int(v)
    a = int(a)
    if v <= 0:
        return
    cur = int(st.get("cur", pol.get("base", 5)))
    kmin = int(pol.get("kmin", 5))
    kmax = int(pol.get("kmax", 12))
    if a >= v:
        st["streak"] = int(st.get("streak", 0)) + 1
        if st["streak"] >= int(pol.get("up_after", 1)):
            cur = min(cur + int(pol.get("up_step", 2)), kmax)
            st["streak"] = 0
    else:
        st["streak"] = 0
        if pol.get("down", "accepted") == "accepted":
            cur = max(kmin, min(cur, a + int(pol.get("down_slack", 1))))
        else:
            cur = max(kmin, cur - int(pol.get("down_step", 2)))
    st["cur"] = max(kmin, min(cur, kmax))


def decide_w(st, pol: dict, step: int, kmax_run: int) -> int:
    """The width of one request for the next step."""
    kmax_run = int(kmax_run)
    mode = pol.get("mode", "fixed")
    if mode == "off":
        return max(1, kmax_run)
    if mode == "fixed":
        return max(1, min(int(pol.get("fixed", 5)), kmax_run))
    base = int(pol.get("base", 5))
    cur = int(st.get("cur", base)) if st else base
    if st and int(st.get("hot_until", -1)) >= int(step):
        cur = max(cur, int(pol.get("hot_k", 12)))
    hi = min(int(pol.get("kmax", 12)), kmax_run)
    return max(1, min(cur, max(1, hi)))


# ----------------------------------------------------------------------------------- threshold mode
THR_KMAX = 12  # the number of draft positions the conditional acceptance rates cover
THR_BUCKETS = ("T0", "T(0,0.5]", "T(0.5,0.9]", "T>0.9")
# The default survival rates S_j of every temperature bucket; the conditional acceptance rates
# g_j are derived from them (g_1 = S_1, g_j = S_j / S_{j-1}).
THR_DEFAULT_SURVIVAL = {
    "T0": (0.605, 0.332, 0.177, 0.101, 0.069),
    "T(0,0.5]": (0.577, 0.302, 0.144, 0.074, 0.046),
    "T(0.5,0.9]": (0.577, 0.302, 0.144, 0.074, 0.046),
    "T>0.9": (0.577 * 0.9, 0.302 * 0.9, 0.144 * 0.9, 0.074 * 0.9, 0.046 * 0.9),
}
THR_DEFAULT = {
    "c_ms": 0.49,
    "base_ms": 40.0,
    "p_ms": 0.30,
    "wmin": 1,
    "wmax": 12,
    "ema": 0.2,
    "budget": 0,
}
# The numeric bounds of the policy fields; a value outside of them falls back to the default.
THR_BOUNDS = {
    "c_ms": (0.0, 1e6),
    "base_ms": (0.0, 1e6),
    "p_ms": (0.0, 1e6),
    "wmin": (1, THR_KMAX),
    "wmax": (1, THR_KMAX),
    "ema": (0.0, 1.0),
    "budget": (0, 1e9),
}


def _thr_survival_to_g(surv) -> list:
    """The conditional acceptance rates g_j of the survival rates S_j (12 values).

    ``g_1 = S_1`` and ``g_j = S_j / S_{j-1}``; the positions without a survival rate are
    extrapolated geometrically from the ratio of the last two known survival rates.
    """
    s = [float(x) for x in surv]
    while len(s) < THR_KMAX:
        if len(s) >= 2 and s[-2] > 0.0:
            r = min(s[-1] / s[-2], 1.0)
        elif s:
            r = min(s[-1], 1.0)
        else:
            r = 0.0
        s.append(s[-1] * r if s else 0.0)
    g = []
    for j in range(THR_KMAX):
        if j == 0:
            g.append(s[0])
        else:
            g.append(s[j] / s[j - 1] if s[j - 1] > 0.0 else 0.0)
    return g


def thr_default_priors() -> dict:
    """The default conditional acceptance rates of every temperature bucket."""
    return {b: _thr_survival_to_g(THR_DEFAULT_SURVIVAL[b]) for b in THR_BUCKETS}


def _thr_num(obj: dict, name: str) -> float:
    """A numeric policy field; the default when it is missing or not a number."""
    value = obj.get(name, None)
    if value is None or isinstance(value, bool):
        raise ValueError("the policy field %r must be a number" % (name,))
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("the policy field %r must be a number" % (name,))


def _thr_policy_fields(obj: dict) -> dict:
    """The threshold fields of a policy; an invalid value falls back to the default."""
    out = {}
    for name, default in THR_DEFAULT.items():
        if name not in obj:
            out[name] = default
            continue
        try:
            value = _thr_num(obj, name)
        except ValueError:
            print("%s kad threshold policy field %r is invalid, the default %r is used"
                  % (TAG, name, default), flush=True)
            out[name] = default
            continue
        lo, hi = THR_BOUNDS[name]
        if not (lo <= value <= hi):
            print("%s kad threshold policy field %r is out of range, the default %r is used"
                  % (TAG, name, default), flush=True)
            out[name] = default
            continue
        if name in ("wmin", "wmax", "budget"):
            out[name] = int(value)
        else:
            out[name] = value
    out["wmin"] = max(1, int(out["wmin"]))
    out["wmax"] = max(out["wmin"], min(int(out["wmax"]), THR_KMAX))
    out["budget"] = max(0, int(out["budget"]))
    priors = thr_default_priors()
    raw = obj.get("priors", None)
    if raw is not None:
        if not isinstance(raw, dict):
            print("%s kad threshold policy 'priors' must be an object, the defaults are used"
                  % (TAG,), flush=True)
        else:
            for bucket, values in raw.items():
                if bucket not in priors:
                    print("%s kad threshold policy has an unknown prior bucket %r, it is ignored"
                          % (TAG, bucket), flush=True)
                    continue
                if not isinstance(values, (list, tuple)) or len(values) != THR_KMAX:
                    print("%s kad threshold prior %r must be a list of %d numbers, the default is used"
                          % (TAG, bucket, THR_KMAX), flush=True)
                    continue
                bad = False
                g = []
                for x in values:
                    if isinstance(x, bool) or not isinstance(x, (int, float)):
                        bad = True
                        break
                    g.append(float(x))
                if bad or any(x < 0.0 or x > 1.0 for x in g):
                    print("%s kad threshold prior %r is invalid, the default is used"
                          % (TAG, bucket), flush=True)
                    continue
                priors[bucket] = g
    out["priors"] = priors
    return out


def thr_bucket(temp) -> str:
    """The temperature bucket of a request; the unknown temperatures go to ``T0``."""
    try:
        t = float(temp)
    except (TypeError, ValueError):
        return "T0"
    if t != t or t < 0.0:
        return "T0"
    if t == 0.0:
        return "T0"
    if t <= 0.5:
        return "T(0,0.5]"
    if t <= 0.9:
        return "T(0.5,0.9]"
    return "T>0.9"


def thr_survival(g) -> list:
    """The survival rates S_j = g_1 * ... * g_j of the conditional acceptance rates."""
    s = []
    acc = 1.0
    for x in g:
        acc *= float(x)
        s.append(acc)
    return s


def thr_update_g(g: list, v: int, a: int, ema: float) -> list:
    """The EMA update of the conditional acceptance rates after the verification of a step.

    The positions ``1..a`` are observed as accepted, the position ``a + 1`` (when ``a < v``)
    is observed as rejected and the positions beyond are not observed at all.
    """
    v = int(v)
    a = int(a)
    ema = float(ema)
    out = []
    for j in range(THR_KMAX):
        x = float(g[j])
        if j < a and j < v:
            obs = 1.0
        elif j == a and a < v:
            obs = 0.0
        else:
            out.append(x)
            continue
        x = (1.0 - ema) * x + ema * obs
        out.append(min(0.999, max(0.001, x)))
    return out


def thr_tau(S_list, w, base: float, c: float, p: float) -> float:
    """The threshold ``tau = c * R`` of the batch; ``R`` is the throughput of the step."""
    tok = 0.0
    total = 0
    for s, wi in zip(S_list, w):
        tok += 1.0 + float(sum(s[:wi]))
        total += int(wi) + 1
    dur = float(base) + float(c) * total + float(p) * len(S_list)
    if dur <= 0.0:
        return float("inf")
    return float(c) * tok / dur


def thr_solve_batch(surv, base: float, c: float, p: float, wmin: int, wmax: int, iters: int = 60):
    """The widths of the batch and the threshold ``tau`` they were solved with."""
    wmin = max(1, int(wmin))
    wmax = max(wmin, int(wmax))
    surv = [np.asarray(s, dtype=float) for s in surv]
    w = [wmin] * len(surv)
    tau = 0.0
    for _ in range(int(iters)):
        tau = thr_tau(surv, w, base, c, p)
        nw = []
        for s in surv:
            nw.append(int(min(wmax, max(wmin, int((s > tau).sum())))))
        if nw == w:
            break
        w = nw
    return w, tau


def thr_solve(S_list, base: float, c: float, p: float, wmin: int, wmax: int, iters: int = 60):
    """The width of every request of the batch (Dinkelbach iteration).

    The position ``j`` of the request ``i`` is sent for verification when ``S_ij > tau`` with
    ``tau = c * R`` and ``R`` the throughput of the whole batch (the expected tokens of the
    step divided by its duration).
    """
    return thr_solve_batch(S_list, base, c, p, wmin, wmax, iters)[0]


def thr_apply_budget(w, S_list, budget: int, wmin: int) -> list:
    """Drop the positions with the smallest survival rates until the batch fits the budget."""
    budget = int(budget)
    if budget <= 0:
        return list(w)
    wmin = max(1, int(wmin))
    out = [int(x) for x in w]
    total = sum(x + 1 for x in out)
    while total > budget:
        # the position with the smallest survival rate: the last position of the request whose
        # survival rate is the smallest
        best = None
        for i, wi in enumerate(out):
            if wi <= wmin:
                continue
            s = S_list[i]
            if len(s) == 0:
                continue
            val = float(s[min(wi, len(s)) - 1])
            if best is None or val < best[0]:
                best = (val, i)
        if best is None:
            break
        out[best[1]] -= 1
        total -= 1
    return out


def _thr_state(rid, req, pol: dict) -> dict:
    """The threshold state of a request; it is created with the prior of its temperature."""
    states = _state["thr"]
    st = states.get(rid)
    if st is None:
        sp = getattr(req, "sampling_params", None) if req is not None else None
        temp = _shadow_num(sp, "temperature", -1.0) if sp is not None else -1.0
        bucket = thr_bucket(temp)
        priors = pol.get("priors") or thr_default_priors()
        g = priors.get(bucket) or priors.get("T0")
        st = states[rid] = {"g": [min(0.999, max(0.001, float(x))) for x in g], "bucket": bucket}
    return st


def _thr_fail(e: BaseException) -> None:
    """A broken threshold decision must not stop the server: log once and use the ladder."""
    if _state["thr_fail"]:
        return
    _state["thr_fail"] = True
    print("%s kad threshold failed, the ladder decides the widths: %r" % (TAG, e), flush=True)

def _thr_widths(pol: dict, rids, requests, kmax_run: int) -> dict:
    """The width of every request of the batch; the whole batch is decided at once.

    The requests that carry drafts are the batch of the decision: their conditional acceptance
    rates give the survival rates S_ij and their widths are solved together (the threshold tau
    depends on the throughput of the whole batch).  A request without drafts is not verified
    this step: it gets the width of the same threshold applied to its own survival rates.  A
    structured output request is never shortened: it keeps the full width.
    """
    wmin = int(pol.get("wmin", 1))
    wmax = max(wmin, min(int(pol.get("wmax", THR_KMAX)), THR_KMAX))
    kmax_run = max(1, int(kmax_run))
    base = float(pol.get("base_ms", 40.0))
    c = float(pol.get("c_ms", 0.49))
    p = float(pol.get("p_ms", 0.30))
    hi = min(wmax, kmax_run)
    batch = []      # the requests that carry drafts: the batch of the decision
    others = []     # the requests without drafts
    structured = []
    for rid in rids:
        req = requests.get(rid) if hasattr(requests, "get") else None
        if req is not None and getattr(req, "use_structured_output", False):
            structured.append(rid)
            continue
        spec = getattr(req, "spec_token_ids", None) if req is not None else None
        if spec:
            batch.append(rid)
        else:
            others.append(rid)
    surv = []
    for rid in batch:
        st = _thr_state(rid, requests.get(rid) if hasattr(requests, "get") else None, pol)
        surv.append(thr_survival(st["g"]))
    if not surv:
        # no request is verified this step: the requests without drafts decide the batch
        surv = []
        for rid in others:
            st = _thr_state(rid, requests.get(rid) if hasattr(requests, "get") else None, pol)
            surv.append(thr_survival(st["g"]))
        batch = list(others)
        others = []
    if surv:
        widths, tau = thr_solve_batch(surv, base, c, p, wmin, hi)
    else:
        widths, tau = [], 0.0
    out = {}
    for rid, wi in zip(batch, widths):
        out[rid] = max(1, min(int(wi), hi))
    for rid in others:
        st = _thr_state(rid, requests.get(rid) if hasattr(requests, "get") else None, pol)
        s = thr_survival(st["g"])
        n = int(sum(1 for x in s if x > tau))
        out[rid] = max(wmin, min(n, hi))
    for rid in structured:
        out[rid] = kmax_run
    budget = int(pol.get("budget", 0))
    if budget > 0:
        # the budget caps the sum of (w + 1) of the batch: the positions with the smallest
        # survival rates are dropped first, no request goes below wmin
        fixed = sum(out[rid] + 1 for rid in structured)
        adjustable = [rid for rid in out if rid not in structured]
        if adjustable:
            S = [thr_survival(_thr_state(rid, requests.get(rid) if hasattr(requests, "get") else None,
                                         pol)["g"]) for rid in adjustable]
            capped = thr_apply_budget([out[rid] for rid in adjustable], S, budget - fixed, wmin)
            for rid, wi in zip(adjustable, capped):
                out[rid] = max(1, min(int(wi), hi))
    return out


def _ladder_widths(pol: dict, rids, requests, kmax_run: int, step: int) -> dict:
    """The width of every request of the batch with the ladder (the fallback of the threshold)."""
    states = _state["ladder"]
    out = {}
    for rid in rids:
        st = states.get(rid)
        if st is None:
            st = states[rid] = {"cur": int(pol.get("base", 5)), "streak": 0, "hot_until": -1}
        out[rid] = decide_w(st, pol, step, kmax_run)
        req = requests.get(rid) if hasattr(requests, "get") else None
        if req is not None and getattr(req, "use_structured_output", False):
            out[rid] = int(kmax_run)
    return out


# ------------------------------------------------------------------------------------------- shadow
def _shadow_field(obj, name: str, default):
    """A field of a sampling-parameter-like object; the default when it is missing or broken."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(name, None)
    else:
        value = getattr(obj, name, None)
    if value is None or isinstance(value, bool):
        return default
    return value


def _shadow_num(obj, name: str, default: float) -> float:
    """A numeric field of a sampling-parameter-like object; the default when it is not a number."""
    value = _shadow_field(obj, name, None)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _shadow_fail(e: BaseException) -> None:
    """A broken shadow instrument must not stop the server: log once and switch it off."""
    _state["shadow_ok"] = False
    if not _state["shadow_logged"]:
        _state["shadow_logged"] = True
        print("%s shadow instrumentation failed, the shadow recording is off: %r" % (TAG, e), flush=True)


def _shadow_step_dt() -> float:
    """The wall-clock milliseconds since the last post_schedule; the first call returns 0."""
    now = time.time()
    prev = _state["shadow_t"]
    _state["shadow_t"] = now
    if prev is None:
        return 0.0
    return max(0.0, (now - float(prev)) * 1000.0)


def _shadow_sent_w(so, rid, spec) -> int:
    """The number of drafts the request really sends for verification after the shortening."""
    if spec:
        return len(spec)
    sched_spec = getattr(so, "scheduled_spec_decode_tokens", None) or {}
    tokens = sched_spec.get(rid) if hasattr(sched_spec, "get") else None
    if tokens:
        return len(tokens)
    return 0


def _shadow_post(sched, so, step: int) -> None:
    """Record the shadow instrumentation of one scheduling step (never raises).

    A scheduled request is a prefill one when its computed tokens are below its prompt length;
    the other scheduled requests are decode ones.  Only the decode requests that carry drafts
    are recorded (``n_dec``) and they define the batch size (``bt`` = sum of ``w + 1``).
    """
    if not _state["shadow_ok"]:
        return
    try:
        setattr(so, ATTR_SHADOW_STEP, int(step))
        rec = shadow.get_recorder(SHADOW_ROLE, 0)
        scheduled = getattr(so, "num_scheduled_tokens", None) or {}
        requests = getattr(sched, "requests", None) or {}
        n_dec = 0
        n_pre = 0
        tok_pre = 0
        bt = 0
        for rid in scheduled:
            req = requests.get(rid) if hasattr(requests, "get") else None
            if req is None:
                continue
            try:
                n_computed = int(getattr(req, "num_computed_tokens", 0) or 0)
                n_prompt = int(getattr(req, "num_prompt_tokens", 0) or 0)
            except (TypeError, ValueError):
                continue
            if n_computed < n_prompt:
                n_pre += 1
                tok_pre += int(scheduled.get(rid, 0) or 0)
            spec = getattr(req, "spec_token_ids", None)
            if not spec:
                continue
            w = _shadow_sent_w(so, rid, spec)
            if w <= 0:
                continue
            n_dec += 1
            bt += w + 1
            sp = getattr(req, "sampling_params", None)
            top_k = _shadow_field(sp, "top_k", -1)
            try:
                top_k = int(top_k)
            except (TypeError, ValueError):
                top_k = -1
            if top_k == 0:
                top_k = -1
            rec.record(shadow.KIND_REQ, {
                "step": int(step),
                "rid": str(rid),
                "pos": n_computed,
                "temp": _shadow_num(sp, "temperature", -1),
                "top_p": _shadow_num(sp, "top_p", -1),
                "top_k": top_k,
                "st": bool(getattr(req, "use_structured_output", False)),
                "w": w,
            })
        dt_ms = _shadow_step_dt()
        rec.record(shadow.KIND_STEP, {
            "step": int(step),
            "n_dec": int(n_dec),
            "n_pre": int(n_pre),
            "tok_pre": int(tok_pre),
            "bt": int(bt),
            "dt_ms": float(dt_ms),
        })
    except Exception as e:  # noqa: BLE001  the shadow must never stop the scheduler
        _shadow_fail(e)


def _shadow_acc(rec, step: int, rid, pos: int, v: int, a: int) -> None:
    """Record the verification result of one request (never raises)."""
    if rec is None or not _state["shadow_ok"]:
        return
    try:
        rec.record(shadow.KIND_ACC, {
            "step": int(step),
            "rid": str(rid),
            "pos": int(pos),
            "v": int(v),
            "a": int(a),
        })
    except Exception as e:  # noqa: BLE001  the shadow must never stop the scheduler
        _shadow_fail(e)


# ------------------------------------------------------------------------------------------- hooks
def pre_schedule(sched) -> None:
    """Shorten the draft lists to the width the worker produced in the previous step."""
    if not enabled():
        return
    pol = _current_policy()
    kmax = int(getattr(sched, "num_spec_tokens", 0) or 0)
    _state["kmax_run"] = kmax
    if kmax <= 0 or pol["mode"] == "off":
        return
    step = int(getattr(sched, "current_step", 0) or 0) + 1
    w_prev = _state["w_prev"]
    for r, rid in _eligible(sched, step):
        spec = r.spec_token_ids
        n = min(len(spec), int(w_prev.get(rid, kmax)))
        if len(spec) > n:
            # A new list: the placeholders are shared by every request of the batch.
            r.spec_token_ids = spec[:n]


def post_schedule(sched, so) -> None:
    """Publish the width of every scheduled request to the workers."""
    if not enabled():
        return
    pol = _current_policy()
    kmax = int(getattr(sched, "num_spec_tokens", 0) or 0)
    if kmax <= 0:
        return
    _state["kmax_run"] = kmax
    step = int(getattr(sched, "current_step", 0) or 0)
    scheduled = getattr(so, "num_scheduled_tokens", None) or {}
    states = _state["ladder"]
    requests = getattr(sched, "requests", None) or {}
    w = {}
    if pol["mode"] == "threshold":
        # the whole batch is decided at once by the threshold strategy
        if not _state["thr_fail"]:
            try:
                w = _thr_widths(pol, list(scheduled.keys()), requests, kmax)
            except Exception as e:  # noqa: BLE001  the ladder takes over
                _thr_fail(e)
        if not w:
            w = _ladder_widths(pol, list(scheduled.keys()), requests, kmax, step)
    else:
        for rid in scheduled:
            st = None
            if pol["mode"] == "ladder":
                st = states.get(rid)
                if st is None:
                    st = states[rid] = {"cur": int(pol["base"]), "streak": 0, "hot_until": -1}
            w[rid] = decide_w(st, pol, step, kmax)
            req = requests.get(rid) if hasattr(requests, "get") else None
            if req is not None and getattr(req, "use_structured_output", False):
                # A structured output request is never shortened (pre_schedule skips it): the
                # worker must draft the full width, otherwise the next step would verify stale
                # drafts.
                w[rid] = kmax
    setattr(so, ATTR_SO, {"v": 1, "w": w})
    w_prev = _state["w_prev"]
    hist = _state["w_hist"]
    for rid, width in w.items():
        w_prev[rid] = int(width)
        key = str(int(width))
        hist[key] = hist.get(key, 0) + 1
    _state["steps"] += 1
    if len(set(w.values())) > 1:
        _state["mixed_steps"] += 1
    if shadow.enabled():
        _shadow_post(sched, so, step)


def ingest(sched, scheduler_output, model_runner_output) -> None:
    """Read the verification result of the step and move the widths of the requests."""
    if not enabled():
        return
    pol = _current_policy()
    step = int(getattr(sched, "current_step", 0) or 0)
    spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    idx_map = getattr(model_runner_output, "req_id_to_index", None) or {}
    sampled = getattr(model_runner_output, "sampled_token_ids", None) or []
    states = _state["ladder"]
    requests = getattr(sched, "requests", None) or {}
    shadow_rec = None
    shadow_step = -1
    if shadow.enabled():
        try:
            shadow_rec = shadow.get_recorder(SHADOW_ROLE, 0)
            shadow_step = getattr(scheduler_output, ATTR_SHADOW_STEP, -1)
            shadow_step = int(shadow_step)
        except Exception as e:  # noqa: BLE001  the shadow must never stop the scheduler
            _shadow_fail(e)
            shadow_rec = None
            shadow_step = -1
    for rid, tokens in spec.items():
        v = len(tokens or ())
        if v <= 0:
            continue
        idx = idx_map.get(rid) if hasattr(idx_map, "get") else None
        if idx is None:
            continue
        try:
            gen = sampled[idx]
        except (IndexError, KeyError, TypeError):
            continue
        n = len(gen) if gen is not None else 0
        if n == 0:
            continue
        a = max(0, min(n - 1, v))
        if shadow_rec is not None:
            req = requests.get(rid) if hasattr(requests, "get") else None
            pos = -1
            if req is not None:
                try:
                    pos = int(getattr(req, "num_computed_tokens", -1))
                except (TypeError, ValueError):
                    pos = -1
            _shadow_acc(shadow_rec, shadow_step, rid, pos, v, a)
        if pol["mode"] == "ladder":
            st = states.get(rid)
            if st is None:
                st = states[rid] = {"cur": int(pol["base"]), "streak": 0, "hot_until": -1}
            ladder_update(st, v, a, pol)
        elif pol["mode"] == "threshold":
            # the conditional acceptance rates move towards the observation of this step
            try:
                st = _thr_state(rid, requests.get(rid) if hasattr(requests, "get") else None, pol)
                st["g"] = thr_update_g(st["g"], v, a, float(pol.get("ema", 0.2)))
            except Exception as e:  # noqa: BLE001  the threshold must not stop the scheduler
                _thr_fail(e)
        _state["v_steps"] += 1
        _state["sum_v"] += v
        _state["sum_a"] += a
        acc = _state["acc_by_v"].get(str(v))
        if acc is None:
            acc = _state["acc_by_v"][str(v)] = [0, 0]
        acc[0] += 1
        acc[1] += a
    report = getattr(model_runner_output, "_xy_hk", None)
    if pol["mode"] == "ladder" and isinstance(report, dict) and report.get("v") == 1:
        cand = report.get("cand") or {}
        if isinstance(cand, dict):
            ttl = int(pol["hot_ttl"])
            for rid in cand:
                st = states.get(rid)
                if st is None:
                    st = states[rid] = {"cur": int(pol["base"]), "streak": 0, "hot_until": -1}
                st["hot_until"] = step + ttl
    _state["ingest_steps"] += 1
    if _state["ingest_steps"] % CLEAN_EVERY == 0:
        _clean(sched)


def _clean(sched) -> None:
    """Drop the state of the requests the scheduler does not know any more."""
    requests = getattr(sched, "requests", None) or {}
    if not requests:
        return
    w_prev = _state["w_prev"]
    for rid in [rid for rid in w_prev if rid not in requests]:
        w_prev.pop(rid, None)
    states = _state["ladder"]
    for rid in [rid for rid in states if rid not in requests]:
        states.pop(rid, None)
    thr = _state["thr"]
    for rid in [rid for rid in thr if rid not in requests]:
        thr.pop(rid, None)


# ------------------------------------------------------------------------------------------- logging
def snapshot() -> dict:
    pol = _state["policy"] or _default_policy()
    return {
        "mode": pol.get("mode", ""),
        "policy": dict(pol),
        "policy_sha256": _state["policy_sha256"],
        "kmax_run": _state["kmax_run"],
        "steps": _state["steps"],
        "mixed_steps": _state["mixed_steps"],
        "w_hist": {str(k): int(v) for k, v in _state["w_hist"].items()},
        "v_steps": _state["v_steps"],
        "sum_v": _state["sum_v"],
        "sum_a": _state["sum_a"],
        "acc_by_v": {str(k): [int(v[0]), int(v[1])] for k, v in _state["acc_by_v"].items()},
        "tracked": len(_state["w_prev"]),
        "errors": _state["errors"],
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
    payload["pid"] = os.getpid()
    tmp = "%s.tmp.%d" % (path, os.getpid())
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError as e:  # noqa: BLE001  the statistics must not stop the server
        _log_error("stats write", e)
        try:
            os.remove(tmp)
        except OSError:
            pass


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
    steps = snap["v_steps"]
    mean_v = (snap["sum_v"] / steps) if steps else 0.0
    mean_a = (snap["sum_a"] / steps) if steps else 0.0
    hist = ",".join("%s:%d" % (k, snap["w_hist"][k]) for k in sorted(snap["w_hist"], key=lambda x: int(x)))
    print("%s steps=%d mixed=%d w_hist=%s mean_v=%.3f mean_a=%.3f tracked=%d sha=%s"
          % (KAD_TAG, snap["steps"], snap["mixed_steps"], hist, mean_v, mean_a, snap["tracked"],
             snap["policy_sha256"][:12]), flush=True)
    _write_stats(snap)


# ------------------------------------------------------------------------------------------- patching
def _mark(wrapper, orig) -> None:
    wrapper.__recall_kad_wrapped__ = True
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
    if orig_schedule is not None and not getattr(orig_schedule, "__recall_kad_wrapped__", False):

        def schedule(self, *a, **kw):
            try:
                pre_schedule(self)
            except Exception as e:  # noqa: BLE001  never break scheduling
                _log_error("pre_schedule", e)
            so = orig_schedule(self, *a, **kw)
            try:
                post_schedule(self, so)
            except Exception as e:  # noqa: BLE001  without _xy_kw the worker uses K_max
                _log_error("post_schedule", e)
            try:
                _maybe_log()
            except Exception as e:  # noqa: BLE001
                _log_error("log", e)
            return so

        _mark(schedule, orig_schedule)
        cls.schedule = schedule

    orig_update = getattr(cls, "update_from_output", None)
    if orig_update is not None and not getattr(orig_update, "__recall_kad_wrapped__", False):

        def update_from_output(self, *a, **kw):
            try:
                so = kw.get("scheduler_output")
                mro = kw.get("model_runner_output")
                if so is None and len(a) > 0:
                    so = a[0]
                if mro is None and len(a) > 1:
                    mro = a[1]
                ingest(self, so, mro)
            except Exception as e:  # noqa: BLE001  never break serving
                _log_error("ingest", e)
            try:
                _maybe_log()
            except Exception as e:  # noqa: BLE001
                _log_error("log", e)
            return orig_update(self, *a, **kw)

        _mark(update_from_output, orig_update)
        cls.update_from_output = update_from_output

    print("%s patched %s.Scheduler (kad)" % (TAG, getattr(module, "__name__", "scheduler")), flush=True)
