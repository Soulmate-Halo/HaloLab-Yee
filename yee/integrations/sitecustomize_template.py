"""Recall shim for SGLang, first on PYTHONPATH: run Mia's adapter sitecustomize unchanged, then
(SGLANG_D_RECALL=1) install the Recall hooks. Any Recall failure is printed and serving goes on without it.

Copy this file to ``sitecustomize.py`` on the PYTHONPATH that comes first for the server process, e.g.

    cp sitecustomize_template.py /opt/recall-accel/sitecustomize.py
    export PYTHONPATH=/opt/recall-accel:$PYTHONPATH

Environment:
  SGLANG_D_RECALL                 1/true/yes/on turns the hooks on (default 0)
  SGLANG_D_RECALL_MIA_SITECUSTOMIZE  path of Mia's own sitecustomize.py (default /opt/dsv41/adapter/sitecustomize.py)
  DSV41_RECALL / DSV41_RECALL_MIA_SITECUSTOMIZE  the old names, still read as a fallback
"""
import importlib.util
import os
import sys

try:  # the package's own reader when it is importable; the fallback below is the same logic
    from yee.core import ENV_NEW, ENV_OLD, _env
except Exception:  # noqa: BLE001  the template must work as a standalone sitecustomize.py
    ENV_NEW = {"on": "SGLANG_D_RECALL", "mia_sitecustomize": "SGLANG_D_RECALL_MIA_SITECUSTOMIZE"}
    ENV_OLD = {"on": "DSV41_RECALL", "mia_sitecustomize": "DSV41_RECALL_MIA_SITECUSTOMIZE"}

    def _env(new, old, default=""):
        """Read one environment variable: the new name wins, the old name is the fallback."""
        for name in (new, old):
            if not name:
                continue
            v = os.environ.get(name)
            if v is None:
                continue
            v = str(v).strip()
            if v:
                return v
        return default


_MIA = _env(ENV_NEW["mia_sitecustomize"], ENV_OLD["mia_sitecustomize"],
            "/opt/dsv41/adapter/sitecustomize.py")
if os.path.exists(_MIA):
    _spec = importlib.util.spec_from_file_location("_mia_sitecustomize", _MIA)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["_mia_sitecustomize"] = _mod
    _spec.loader.exec_module(_mod)

if _env(ENV_NEW["on"], ENV_OLD["on"], "0").strip().lower() in ("1", "true", "on", "yes"):
    try:
        from yee.integrations import registry

        registry.install_auto()
    except Exception as _e:  # noqa: BLE001
        print("[recall-accel] install failed: %r" % (_e,), flush=True)
