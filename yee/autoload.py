"""Autoload entry point: ``import yee.autoload`` (the .pth written by ``recall-accel install-hook``).

Yee 4.2 first fills its startup defaults (``yee.integrations.defaults``: every accelerator on unless set; nothing
when YEE=0), then installs the hooks when RECALL_ACCEL=1 (SGLANG_D_RECALL / VLLM_D_RECALL / DSV41_RECALL as
fallbacks); YEE_EXPERT_BUDGET=1 also installs the hot-expert verify budget. Server worker processes inherit the
environment, so every worker gets them. ``RECALL_ACCEL_ADAPTER``
(auto | dspark | dflash | eagle | ngram | vllm_spec | off) narrows the choice. YEE_KAD=1 (independent of
RECALL_ACCEL) arms the KAD adapters of SGLang, TensorRT-LLM, LMDeploy, mlx-lm and transformers
(``yee.integrations.kad_auto``: each installs when its engine module is imported). Never raises.
"""
from __future__ import annotations

import os

from .core import ENV_NEW, ENV_OLD, _env

_TRUE = ("1", "true", "yes", "on")


def enabled() -> bool:
    return _env(ENV_NEW["on"], ENV_OLD["on"], "0").strip().lower() in _TRUE


def kad_enabled() -> bool:
    """YEE_KAD is 1 / true / yes / on: per-request adaptive draft count (``yee doc kad``)."""
    return os.environ.get("YEE_KAD", "").strip().lower() in _TRUE


def main() -> int:
    try:  # 4.2: every accelerator on unless the environment says otherwise (nothing when YEE=0)
        from .integrations import defaults

        defaults.apply()
    except Exception as e:  # noqa: BLE001  never take the process down
        print("[yee] defaults failed: %r" % (e,), flush=True)
    if enabled():
        try:
            from .integrations import registry

            registry.install_auto()
        except Exception as e:  # noqa: BLE001  never take the process down
            print("[recall-accel] install failed: %r" % (e,), flush=True)
    if kad_enabled():
        try:
            from .integrations import kad_auto

            kad_auto.install(True)
        except Exception as e:  # noqa: BLE001  never take the process down
            print("[yee-kad] autoload failed: %r" % (e,), flush=True)
    if os.environ.get("YEE_EXPERT_BUDGET", "").strip().lower() in _TRUE:
        try:
            from .integrations import expert_budget

            expert_budget.install()
        except Exception as e:  # noqa: BLE001  never take the process down
            print("[yee-eb] autoload failed: %r" % (e,), flush=True)
    return 0


main()

__all__ = ["enabled", "kad_enabled", "main"]
