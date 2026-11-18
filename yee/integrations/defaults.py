"""Yee 4.2 startup defaults: every accelerator is on unless the environment says otherwise.

``apply()`` runs first in the sitecustomize template and in the ``.pth`` autoload, before any engine module is
imported. It only fills variables that are unset (or empty) under every alias the readers accept, so an explicit
value always wins, and it is idempotent. Setting the master switch off (``YEE`` / ``RECALL_ACCEL`` /
``SGLANG_D_RECALL`` / ``VLLM_D_RECALL`` / ``DSV41_RECALL`` = 0/false/no/off) keeps the whole of Yee off: nothing
is filled in.

Inside the SGLang Mia DSV41 image (``/opt/dsv41/adapter`` exists, or the Mia sitecustomize path is set) the block
size is read from the launch command (``--speculative-dspark-block-size N``) and the verify-cap variables Mia's
adapter reads at import time are filled from it; launch arguments themselves (block size, EP, MoE backend) are
never changed. Server worker processes inherit the environment of the launcher process, which is where the
command line is visible. Standard library only: importable before ``yee`` itself.
"""
from __future__ import annotations

import os
import sys

VERSION = "4.3.0"
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")
MASTER = ("YEE", "RECALL_ACCEL", "SGLANG_D_RECALL", "VLLM_D_RECALL", "DSV41_RECALL")
MIA_ADAPTER = "/opt/dsv41/adapter"
MIA_SITE_VARS = ("RECALL_ACCEL_MIA_SITECUSTOMIZE", "SGLANG_D_RECALL_MIA_SITECUSTOMIZE",
                 "VLLM_D_RECALL_MIA_SITECUSTOMIZE", "DSV41_RECALL_MIA_SITECUSTOMIZE", "YEE_MIA_SITECUSTOMIZE")
BLOCK_FLAG = "--speculative-dspark-block-size"
APPLIED = "YEE_DEFAULTS_APPLIED"  # names filled by apply() (inherited by child processes), shown as "default"


def _aliases(suffix):
    return ("RECALL_ACCEL_" + suffix, "YEE_" + suffix, "SGLANG_D_RECALL_" + suffix, "VLLM_D_RECALL_" + suffix,
            "DSV41_RECALL_" + suffix)


# (variable to fill, value, names that count as "already set"); filled only while the master switch is not off
GENERIC = (
    ("RECALL_ACCEL", "1", MASTER),
    ("VLLM_D_RECALL", "1", ("VLLM_D_RECALL",)),
    ("RECALL_ACCEL_NONGREEDY", "1", ("RECALL_ACCEL_NONGREEDY", "YEE_NONGREEDY")),
    ("VLLM_D_RECALL_NONGREEDY", "1", ("VLLM_D_RECALL_NONGREEDY",)),
    ("YEE_KAD", "1", ("YEE_KAD",)),
    ("VLLM_D_RECALL_KAD", "1", ("VLLM_D_RECALL_KAD",)),
    ("YEE_EXPERT_BUDGET", "1", ("YEE_EXPERT_BUDGET",)),
    ("YEE_EXPERT_BUDGET_ARMED", "1", ("YEE_EXPERT_BUDGET_ARMED",)),
    ("YEE_EXPERT_BUDGET_P", "0.9", ("YEE_EXPERT_BUDGET_P",)),
    ("RECALL_ACCEL_GIB", "0.25", _aliases("GIB")),
    ("VLLM_D_RECALL_GIB", "0.25", _aliases("GIB")),
    ("RECALL_ACCEL_INDEX_DENSITY", "1", _aliases("INDEX_DENSITY") + ("RECALL_INDEX_DENSITY",)),
    ("VLLM_D_RECALL_INDEX_DENSITY", "1", _aliases("INDEX_DENSITY") + ("RECALL_INDEX_DENSITY",)),
)

_DONE = {"printed": False}


def _val(env, name):
    v = env.get(name)
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def master_off(environ=None):
    """True when any master-switch alias is explicitly off."""
    env = os.environ if environ is None else environ
    for n in MASTER:
        v = _val(env, n)
        if v is not None and v.lower() in _FALSE:
            return True
    return False


def in_mia(environ=None):
    env = os.environ if environ is None else environ
    for n in MIA_SITE_VARS:
        p = _val(env, n)
        if p and os.path.isfile(p):
            return True
    return os.path.isdir(MIA_ADAPTER)


def _cmdline():
    try:
        with open("/proc/self/cmdline", "rb") as f:
            return [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
    except OSError:
        return []


def block_size_from_argv(argv):
    for i, a in enumerate(argv or ()):
        v = None
        if a == BLOCK_FLAG and i + 1 < len(argv):
            v = argv[i + 1]
        elif a.startswith(BLOCK_FLAG + "="):
            v = a.split("=", 1)[1]
        if v is not None:
            try:
                n = int(str(v).strip())
            except ValueError:
                return None
            return n if n > 0 else None
    return None


def _set(env, out, name, value, aliases):
    if any(_val(env, a) is not None for a in aliases):
        return
    env[name] = value
    out[name] = value


def apply(environ=None, argv=None):
    """Fill the 4.2 defaults; returns {name: value} of what was set. Never raises."""
    env = os.environ if environ is None else environ
    out = {}
    try:
        if master_off(env):
            return out
        for name, value, aliases in GENERIC:
            _set(env, out, name, value, aliases)
        if in_mia(env):
            if _val(env, "DSPARK_BLOCK_SIZE") is None:
                n = block_size_from_argv(list(sys.argv) if argv is None else list(argv))
                if n is None and argv is None:
                    n = block_size_from_argv(_cmdline())
                if n is not None:
                    _set(env, out, "DSPARK_BLOCK_SIZE", str(n), ("DSPARK_BLOCK_SIZE",))
            try:
                block = int(_val(env, "DSPARK_BLOCK_SIZE") or "")
            except ValueError:
                block = None
            if block and block > 0:
                _set(env, out, "DSV41_VERIFY_CAP", "conf:0.1", ("DSV41_VERIFY_CAP",))
                _set(env, out, "DSV41_VERIFY_CAP_STRIDE", str(block + 1), ("DSV41_VERIFY_CAP_STRIDE",))
            _set(env, out, "SGLANG_DSPARK_FOLDED_SAMPLING", "2", ("SGLANG_DSPARK_FOLDED_SAMPLING",))
    except Exception as e:  # noqa: BLE001  never take the process down
        try:
            print("[yee] defaults failed: %r" % (e,), flush=True)
        except Exception:  # noqa: BLE001
            pass
        return out
    if out:
        done = set(filter(None, (env.get(APPLIED) or "").split(",")))
        env[APPLIED] = ",".join(sorted(done | set(out)))
    if out and environ is None and not _DONE["printed"]:
        _DONE["printed"] = True
        try:
            print("[yee] %s defaults: %s" % (VERSION, " ".join("%s=%s" % kv for kv in out.items())), flush=True)
        except Exception:  # noqa: BLE001
            pass
    return out


# names shown by ``python -m yee config``: (name, aliases, default shown when unset)
SHOWN = (
    ("YEE / RECALL_ACCEL", MASTER, "1"),
    ("YEE_KAD", ("YEE_KAD",), "1"),
    ("YEE_EXPERT_BUDGET", ("YEE_EXPERT_BUDGET",), "1"),
    ("YEE_EXPERT_BUDGET_ARMED", ("YEE_EXPERT_BUDGET_ARMED",), "1"),
    ("YEE_EXPERT_BUDGET_P", ("YEE_EXPERT_BUDGET_P",), "0.9"),
    ("RECALL_ACCEL_NONGREEDY", ("RECALL_ACCEL_NONGREEDY", "YEE_NONGREEDY"), "1"),
    ("RECALL_ACCEL_GIB", _aliases("GIB"), "0.25"),
    ("RECALL_ACCEL_INDEX_DENSITY", _aliases("INDEX_DENSITY") + ("RECALL_INDEX_DENSITY",), "1"),
    ("RECALL_ACCEL_QHASH", _aliases("QHASH"), "1"),
    ("RECALL_ACCEL_SHARED", _aliases("SHARED"), "1"),
    ("YEE_RECALL_BUDGET", ("YEE_RECALL_BUDGET",), "1"),
    ("VLLM_D_RECALL", ("VLLM_D_RECALL",), "1"),
    ("VLLM_D_RECALL_NONGREEDY", ("VLLM_D_RECALL_NONGREEDY",), "1"),
    ("VLLM_D_RECALL_KAD", ("VLLM_D_RECALL_KAD",), "1"),
)
SHOWN_MIA = (
    ("DSPARK_BLOCK_SIZE", ("DSPARK_BLOCK_SIZE",), "(from --speculative-dspark-block-size)"),
    ("DSV41_VERIFY_CAP", ("DSV41_VERIFY_CAP",), "conf:0.1 (when the block size is known)"),
    ("DSV41_VERIFY_CAP_STRIDE", ("DSV41_VERIFY_CAP_STRIDE",), "block size + 1"),
    ("SGLANG_DSPARK_FOLDED_SAMPLING", ("SGLANG_DSPARK_FOLDED_SAMPLING",), "2"),
)


def effective(environ=None):
    """[(name, value, source)] with source env / default / off, for ``python -m yee config``."""
    env = os.environ if environ is None else environ
    off = master_off(env)
    applied = set(filter(None, (env.get(APPLIED) or "").split(",")))
    rows = []
    shown = SHOWN + (SHOWN_MIA if in_mia(env) else ())
    for name, aliases, default in shown:
        hit = None
        for a in aliases:
            v = _val(env, a)
            if v is not None:
                hit = (a, v)
                break
        if hit is not None:
            label = name if hit[0] in name.split(" / ") else "%s (%s)" % (name, hit[0])
            rows.append((label, hit[1], "default" if hit[0] in applied else "env"))
        elif off:
            rows.append((name, "0", "off"))
        else:
            rows.append((name, default, "default"))
    return rows


__all__ = ["VERSION", "apply", "effective", "master_off", "in_mia", "block_size_from_argv", "GENERIC"]
