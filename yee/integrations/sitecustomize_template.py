"""Yee shim for SGLang, first on PYTHONPATH (``install.sh`` puts it there as ``sitecustomize.py``).

Order: (1) Yee 4.2 startup defaults (every accelerator on unless the environment says otherwise; nothing when
``YEE=0``), (2) Mia's own adapter sitecustomize unchanged, so the adapter modules already see the defaults,
(3) the Recall hooks, KAD and the hot-expert verify budget. Any Yee failure is printed and serving goes on without it.

Manual install: copy this file to ``sitecustomize.py`` in a directory that holds the unpacked wheel and put that
directory first on the server's PYTHONPATH, e.g.

    cp sitecustomize_template.py /opt/yee/sitecustomize.py
    export PYTHONPATH=/opt/yee:$PYTHONPATH

Environment:
  YEE / RECALL_ACCEL (SGLANG_D_RECALL, DSV41_RECALL)  master switch; 4.2 default on, 0 turns all of Yee off
  YEE_KAD, YEE_EXPERT_BUDGET                          4.2 default on; 0 turns one off
  SGLANG_D_RECALL_MIA_SITECUSTOMIZE  path of Mia's own sitecustomize.py (default /opt/dsv41/adapter/sitecustomize.py)
  ``python -m yee config`` lists every switch with its value and source.
"""
import importlib.util
import os
import sys

_TRUE = ("1", "true", "on", "yes")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_defaults():
    """yee/integrations/defaults.py next to this file, without importing the yee package first."""
    path = os.path.join(_HERE, "yee", "integrations", "defaults.py")
    try:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("_yee_defaults", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        from yee.integrations import defaults as mod  # installed as a regular package

        return mod
    except Exception as e:  # noqa: BLE001
        print("[yee] defaults not loaded: %r" % (e,), flush=True)
        return None


def _fallback_defaults():
    """Minimal copy of the 4.2 defaults for when the yee package cannot be imported."""
    env = os.environ
    names = ("YEE", "RECALL_ACCEL", "SGLANG_D_RECALL", "VLLM_D_RECALL", "DSV41_RECALL")
    if any((env.get(n) or "").strip().lower() in ("0", "false", "no", "off") for n in names):
        return
    if not any((env.get(n) or "").strip() for n in names):
        env["RECALL_ACCEL"] = "1"
    for k, v in (("YEE_KAD", "1"), ("YEE_EXPERT_BUDGET", "1"), ("YEE_EXPERT_BUDGET_ARMED", "1"),
                 ("YEE_EXPERT_BUDGET_P", "0.9")):
        if not (env.get(k) or "").strip():
            env[k] = v
    if os.path.isdir("/opt/dsv41/adapter"):
        block = (env.get("DSPARK_BLOCK_SIZE") or "").strip()
        argv = list(sys.argv)
        for i, a in enumerate(argv):
            if not block and a == "--speculative-dspark-block-size" and i + 1 < len(argv):
                block = argv[i + 1]
            elif not block and a.startswith("--speculative-dspark-block-size="):
                block = a.split("=", 1)[1]
        if block.isdigit() and int(block) > 0:
            env.setdefault("DSPARK_BLOCK_SIZE", block)
            if not (env.get("DSV41_VERIFY_CAP") or "").strip():
                env["DSV41_VERIFY_CAP"] = "conf:0.1"
            if not (env.get("DSV41_VERIFY_CAP_STRIDE") or "").strip():
                env["DSV41_VERIFY_CAP_STRIDE"] = str(int(block) + 1)
        if not (env.get("SGLANG_DSPARK_FOLDED_SAMPLING") or "").strip():
            env["SGLANG_DSPARK_FOLDED_SAMPLING"] = "2"


def _on(*names):
    for n in names:
        v = (os.environ.get(n) or "").strip()
        if v:
            return v.lower() in _TRUE
    return False


_D = _load_defaults()
if _D is not None:
    _D.apply()
else:
    try:
        _fallback_defaults()
    except Exception as _e:  # noqa: BLE001
        print("[yee] fallback defaults failed: %r" % (_e,), flush=True)

_MIA = ""
for _n in ("RECALL_ACCEL_MIA_SITECUSTOMIZE", "SGLANG_D_RECALL_MIA_SITECUSTOMIZE", "DSV41_RECALL_MIA_SITECUSTOMIZE"):
    _MIA = (os.environ.get(_n) or "").strip()
    if _MIA:
        break
_MIA = _MIA or "/opt/dsv41/adapter/sitecustomize.py"
if os.path.exists(_MIA) and os.path.abspath(_MIA) != os.path.abspath(__file__):
    _spec = importlib.util.spec_from_file_location("_mia_sitecustomize", _MIA)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["_mia_sitecustomize"] = _mod
    _spec.loader.exec_module(_mod)

if _on("RECALL_ACCEL", "YEE", "SGLANG_D_RECALL", "DSV41_RECALL"):
    try:
        from yee.integrations import registry

        registry.install_auto()
    except Exception as _e:  # noqa: BLE001
        print("[recall-accel] install failed: %r" % (_e,), flush=True)

# KAD and the expert budget have their own switches (filled by the defaults unless YEE=0), as in 4.1
if _on("YEE_KAD"):
    try:
        from yee.integrations import kad_auto

        kad_auto.install(True)
    except Exception as _e:  # noqa: BLE001
        print("[yee-kad] install failed: %r" % (_e,), flush=True)

if _on("YEE_EXPERT_BUDGET"):
    try:
        from yee.integrations import expert_budget

        expert_budget.install()
    except Exception as _e:  # noqa: BLE001
        print("[yee-eb] install failed: %r" % (_e,), flush=True)

# 4.3: what Yee does for this launch command and which parameters to change (YEE_NOTICE=0: quiet)
try:
    from yee.integrations import plan as _plan

    _plan.print_startup()
except Exception:  # noqa: BLE001
    pass
