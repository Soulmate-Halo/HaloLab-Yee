"""Yee KAD core: per-request adaptive draft width for every engine (Yee 3.2.0).

KAD gives every request its own draft width k (the number of draft tokens proposed and verified
in one step), inside the launch width K of the engine (12). With the default ladder policy a
request starts at ``base`` (5), climbs by ``up_step`` (2) after every fully accepted
verification up to ``kmax`` (12), and falls to max(kmin, min(cur, a + 1)) when only ``a`` drafts
were accepted.

The policy logic (``parse_policy``, ``ladder_update``, ``decide_k``) is the one of the vLLM
production adapter ``vllm_d_recall.integrations.vllm_kad`` (``decide_w`` there);
``tests/test_kad_core.py`` checks that both give the same results. The policy file format is the
same, so one ``kad_policy.json`` works on every engine.

One ``Kad`` object per engine process. Per engine step an adapter calls::

    kad.begin_step()                  # step counter, policy reload, periodic log
    k = kad.width(rid)                # or kad.widths(rids) / kad.batch_width(rids)
    ...  draft and verify  ...
    kad.observe(rid, v, a)            # v drafts verified, a of them accepted (bonus token excluded)
    kad.note_batch(widths, k_used)    # statistics
    kad.release(rid)                  # the request finished

Multi-rank engines pass ``leader`` (rank 0) and ``sync`` (a broadcast from rank 0 over the CPU
group of the engine); a new policy then takes effect on every rank at the same step.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

__all__ = ["TAG", "MODES", "KMAX", "DEFAULT_POLICY", "parse_policy", "ladder_update", "decide_k",
           "env_enabled", "Kad"]

TAG = "[yee-kad]"
MODES = ("off", "fixed", "ladder")
KMAX = 12  # the launch width KAD is designed for
MAX_ERRORS = 5
_TRUE = ("1", "true", "yes", "on")

ENV_KAD = "YEE_KAD"
ENV_DIR = "YEE_KAD_DIR"
ENV_POLICY = "YEE_KAD_POLICY"
ENV_STATS = "YEE_KAD_STATS_FILE"
ENV_LOG_S = "YEE_KAD_LOG_S"
ENV_RELOAD_S = "YEE_KAD_RELOAD_S"
ENV_RELOAD_EVERY = "YEE_KAD_RELOAD_EVERY"

DEFAULT_POLICY = {"mode": "ladder", "base": 5, "kmin": 5, "kmax": 12, "up_after": 1, "up_step": 2,
                  "down": "accepted", "down_slack": 1, "down_step": 2, "hot_k": 12, "hot_ttl": 8}


# ------------------------------------------------------------------------------------------- environment
def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _float_env(name: str, default: float) -> float:
    try:
        return float(_env(name, ""))
    except ValueError:
        return float(default)


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name, ""))
    except ValueError:
        return int(default)


def env_enabled() -> bool:
    """True when ``YEE_KAD`` is 1 / true / yes / on."""
    return _env(ENV_KAD).lower() in _TRUE


def _kad_dir() -> str:
    return os.path.expanduser(_env(ENV_DIR, os.path.join("~", ".cache", "yee", "kad")))


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name)) or "engine"


# ------------------------------------------------------------------------------------------- policy (same as vllm_kad)
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
    return pol


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


def decide_k(st, pol: dict, step: int, kmax_run: int) -> int:
    """The width of one request for the next step (``decide_w`` of the vLLM adapter)."""
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


def _new_state(pol: dict) -> dict:
    return {"cur": int(pol.get("base", 5)), "streak": 0, "hot_until": -1}


# ------------------------------------------------------------------------------------------- controller
class Kad:
    """Per-request adaptive draft width for one engine process.

    ``base`` names the engine in logs and in the default statistics file name. ``kmax_run`` is
    the launch width of the engine (``set_kmax_run`` when it is only known later). Arguments left
    to None read the ``YEE_KAD*`` environment variables.
    """

    def __init__(self, base: str, kmax_run: int = KMAX, *, enabled=None, policy_file=None,
                 stats_file=None, log_s=None, reload_s=None, leader: bool = True, sync=None,
                 reload_every=None):
        self.base = str(base)
        self.tag = "%s %s" % (TAG, self.base)
        self.enabled = env_enabled() if enabled is None else bool(enabled)
        kdir = _kad_dir()
        self.policy_file = (policy_file if policy_file is not None
                            else _env(ENV_POLICY, os.path.join(kdir, "kad_policy.json")))
        self.stats_file = (stats_file if stats_file is not None
                           else _env(ENV_STATS, os.path.join(kdir, "kad_stats_%s.json" % _safe(self.base))))
        self.log_s = float(log_s) if log_s is not None else _float_env(ENV_LOG_S, 30.0)
        self.reload_s = float(reload_s) if reload_s is not None else _float_env(ENV_RELOAD_S, 1.0)
        every = int(reload_every) if reload_every is not None else _int_env(ENV_RELOAD_EVERY, 64)
        self.reload_every = max(1, every)
        self.leader = bool(leader)
        self.sync = sync
        self.step = 0
        self.kmax_run = KMAX
        self._lock = threading.RLock()
        self._st = {}
        self._policy = None
        self._policy_sha = ""
        self._stamp = None
        self._last_stat_t = None
        self._errors = 0
        self._err_printed = 0
        self._warned = set()
        self._log_t = time.monotonic()
        self._steps_noted = 0
        self._mixed = 0
        self._w_hist = {}
        self._b_hist = {}
        self._v_steps = 0
        self._sum_v = 0
        self._sum_a = 0
        self._acc_by_v = {}
        self.set_kmax_run(kmax_run)
        if self.enabled and self.sync is None:
            try:
                self._reload_time_based()
            except Exception as e:  # noqa: BLE001  a broken policy must not stop the engine
                self.error("policy load", e)

    # --------------------------------------------------------------------------------------- setup
    def set_kmax_run(self, k) -> None:
        """Set the launch width of the engine; a width other than 12 prints one warning."""
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = KMAX
        k = max(1, k)
        with self._lock:
            self.kmax_run = k
            if self.enabled and k != KMAX and k not in self._warned:
                self._warned.add(k)
                if self.leader:
                    print("%s WARNING launch K=%d, KAD needs %d (per-request width is capped at min(%d, K))"
                          % (self.tag, k, KMAX, KMAX), flush=True)

    # --------------------------------------------------------------------------------------- policy
    def policy(self) -> dict:
        """The policy in force (the default ladder until a valid policy file was read)."""
        pol = self._policy
        if pol is None:
            pol = self._policy = parse_policy(DEFAULT_POLICY)
        return pol

    def _apply(self, data: bytes) -> bool:
        try:
            pol = parse_policy(json.loads(data.decode("utf-8")))
        except Exception as e:  # noqa: BLE001  keep the last valid policy
            self.error("policy load", e)
            return False
        sha = hashlib.sha256(data).hexdigest()
        self._policy = pol
        self._policy_sha = sha
        if self.leader:
            print("%s policy loaded sha=%s %s" % (self.tag, sha[:12], json.dumps(pol, sort_keys=True)),
                  flush=True)
            self._write_stats(self.snapshot())
        return True

    def _read_if_changed(self):
        """The bytes of the policy file when its (mtime_ns, size) changed, else None."""
        path = self.policy_file
        if not path:
            return None
        try:
            info = os.stat(path)
        except OSError:
            return None  # no file yet: keep the last valid policy
        stamp = (int(info.st_mtime_ns), int(info.st_size))
        if stamp == self._stamp:
            return None
        self._stamp = stamp
        with open(path, "rb") as f:
            return f.read()

    def _reload_time_based(self) -> None:
        now = time.monotonic()
        if self._last_stat_t is not None and now - self._last_stat_t < self.reload_s:
            return
        self._last_stat_t = now
        data = self._read_if_changed()
        if data is not None:
            self._apply(data)

    def _reload_synced(self) -> None:
        if not (self.step == 1 or self.step % self.reload_every == 0):
            return
        data = None
        if self.leader:
            try:
                data = self._read_if_changed()
            except Exception as e:  # noqa: BLE001  the broadcast below must still happen
                self.error("policy read", e)
                data = None
        data = self.sync(data)  # every rank calls it at the same step
        if data is not None:
            self._apply(bytes(data))

    # --------------------------------------------------------------------------------------- per step
    def begin_step(self) -> int:
        """Advance the step counter; reload the policy and log when it is time. Returns the step."""
        with self._lock:
            self.step += 1
            if not self.enabled:
                return self.step
            try:
                if self.sync is None:
                    self._reload_time_based()
                else:
                    self._reload_synced()
            except Exception as e:  # noqa: BLE001  never break the engine step
                self.error("policy reload", e)
            try:
                self.maybe_log()
            except Exception as e:  # noqa: BLE001
                self.error("log", e)
            return self.step

    def width(self, rid, structured: bool = False) -> int:
        """The draft width of one request for this step."""
        kmax = self.kmax_run
        if not self.enabled or structured:
            return kmax
        try:
            with self._lock:
                return decide_k(self._st.get(rid), self.policy(), self.step, kmax)
        except Exception as e:  # noqa: BLE001  fall back to the launch width
            self.error("width", e)
            return kmax

    def widths(self, rids, structured=None) -> dict:
        """{rid: width} for the requests of a batch."""
        rids = list(rids)
        flags = _flags(rids, structured)
        return {rid: self.width(rid, flag) for rid, flag in zip(rids, flags)}

    def batch_width(self, rids, structured=None) -> int:
        """max(width) over a batch, for engines whose verification needs one width per batch."""
        ws = self.widths(rids, structured)
        return max(ws.values()) if ws else self.kmax_run

    def note_batch(self, widths, k_used=None) -> None:
        """Record the widths decided for one step (and the batch width actually run)."""
        if not self.enabled:
            return
        try:
            vals = [int(w) for w in (widths.values() if isinstance(widths, dict) else widths)]
            with self._lock:
                if vals:
                    self._steps_noted += 1
                    for w in vals:
                        key = str(w)
                        self._w_hist[key] = self._w_hist.get(key, 0) + 1
                    if len(set(vals)) > 1:
                        self._mixed += 1
                if k_used is not None:
                    key = str(int(k_used))
                    self._b_hist[key] = self._b_hist.get(key, 0) + 1
        except Exception as e:  # noqa: BLE001
            self.error("stats", e)

    def observe(self, rid, v, a) -> None:
        """One verification of one request: ``v`` drafts verified, ``a`` accepted (no bonus token)."""
        if not self.enabled:
            return
        try:
            v = int(v)
            if v <= 0:
                return
            a = max(0, min(int(a), v))
            with self._lock:
                pol = self.policy()
                if pol["mode"] == "ladder":
                    st = self._st.get(rid)
                    if st is None:
                        st = self._st[rid] = _new_state(pol)
                    ladder_update(st, v, a, pol)
                self._v_steps += 1
                self._sum_v += v
                self._sum_a += a
                acc = self._acc_by_v.get(str(v))
                if acc is None:
                    acc = self._acc_by_v[str(v)] = [0, 0]
                acc[0] += 1
                acc[1] += a
        except Exception as e:  # noqa: BLE001
            self.error("observe", e)

    def mark_hot(self, rid) -> None:
        """A recall hit for the request: width max(cur, hot_k) for the next ``hot_ttl`` steps."""
        if not self.enabled:
            return
        try:
            with self._lock:
                pol = self.policy()
                if pol["mode"] != "ladder":
                    return
                st = self._st.get(rid)
                if st is None:
                    st = self._st[rid] = _new_state(pol)
                st["hot_until"] = self.step + int(pol["hot_ttl"])
        except Exception as e:  # noqa: BLE001
            self.error("hot", e)

    def release(self, rid) -> None:
        """The request finished: forget its state."""
        with self._lock:
            self._st.pop(rid, None)

    def retain(self, live_rids) -> None:
        """Keep only the state of the requests in ``live_rids``."""
        live = set(live_rids)
        with self._lock:
            for rid in [r for r in self._st if r not in live]:
                self._st.pop(rid, None)

    def state(self, rid):
        """A copy of the ladder state of one request (None when it has none)."""
        with self._lock:
            st = self._st.get(rid)
            return dict(st) if st is not None else None

    # --------------------------------------------------------------------------------------- errors and stats
    def error(self, what: str, exc: BaseException) -> None:
        """Count one internal error; the first ``MAX_ERRORS`` are printed."""
        self._errors += 1
        if self._err_printed < MAX_ERRORS:
            self._err_printed += 1
            print("%s %s failed, the step keeps the native width: %r" % (self.tag, what, exc), flush=True)

    def snapshot(self) -> dict:
        with self._lock:
            pol = self.policy()
            return {
                "base": self.base,
                "enabled": self.enabled,
                "mode": pol.get("mode", ""),
                "policy": dict(pol),
                "policy_sha256": self._policy_sha,
                "kmax_run": self.kmax_run,
                "steps": self.step,
                "noted_steps": self._steps_noted,
                "mixed_steps": self._mixed,
                "w_hist": dict(self._w_hist),
                "b_hist": dict(self._b_hist),
                "v_steps": self._v_steps,
                "sum_v": self._sum_v,
                "sum_a": self._sum_a,
                "acc_by_v": {k: [int(x[0]), int(x[1])] for k, x in self._acc_by_v.items()},
                "tracked": len(self._st),
                "errors": self._errors,
            }

    def _write_stats(self, snap: dict) -> None:
        path = self.stats_file
        if not path or not self.leader:
            return
        payload = dict(snap)
        payload["ts"] = time.time()
        payload["pid"] = os.getpid()
        tmp = "%s.tmp.%d" % (path, os.getpid())
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except OSError as e:  # noqa: BLE001  the statistics must not stop the engine
            self.error("stats write", e)
            try:
                os.remove(tmp)
            except OSError:
                pass

    def maybe_log(self, force: bool = False) -> None:
        """One log line and one statistics file write every ``log_s`` seconds (leader only)."""
        if not self.enabled or not self.leader:
            return
        now = time.monotonic()
        if not force and (self.log_s <= 0 or now - self._log_t < self.log_s):
            return
        self._log_t = now
        snap = self.snapshot()
        n = snap["v_steps"]
        mean_v = (snap["sum_v"] / n) if n else 0.0
        mean_a = (snap["sum_a"] / n) if n else 0.0
        print("%s steps=%d mixed=%d w_hist=%s b_hist=%s mean_v=%.3f mean_a=%.3f tracked=%d errors=%d sha=%s"
              % (self.tag, snap["steps"], snap["mixed_steps"], _fmt_hist(snap["w_hist"]),
                 _fmt_hist(snap["b_hist"]), mean_v, mean_a, snap["tracked"], snap["errors"],
                 snap["policy_sha256"][:12]), flush=True)
        self._write_stats(snap)


def _flags(rids, structured):
    if structured is None:
        return [False] * len(rids)
    if isinstance(structured, (set, frozenset, dict)):
        return [rid in structured for rid in rids]
    flags = [bool(f) for f in structured]
    if len(flags) != len(rids):
        raise ValueError("structured must be None, a set of request ids or one flag per request")
    return flags


def _fmt_hist(hist: dict) -> str:
    return ",".join("%s:%d" % (k, hist[k]) for k in sorted(hist, key=lambda x: int(x)))
