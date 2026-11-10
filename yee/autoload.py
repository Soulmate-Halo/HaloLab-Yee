"""Autoload entry point: ``import yee.autoload`` (the .pth written by ``recall-accel install-hook``).

Installs the hooks at interpreter start when RECALL_ACCEL=1 (SGLANG_D_RECALL / VLLM_D_RECALL / DSV41_RECALL as
fallbacks). Server worker processes inherit the environment, so every worker gets them. ``RECALL_ACCEL_ADAPTER``
(auto | dspark | dflash | eagle | ngram | vllm_spec | off) narrows the choice. Never raises.
"""
from __future__ import annotations

from .core import ENV_NEW, ENV_OLD, _env

_TRUE = ("1", "true", "yes", "on")


def enabled() -> bool:
    return _env(ENV_NEW["on"], ENV_OLD["on"], "0").strip().lower() in _TRUE


def main() -> int:
    if not enabled():
        return 0
    try:
        from .integrations import registry

        registry.install_auto()
    except Exception as e:  # noqa: BLE001  never take the process down
        print("[recall-accel] install failed: %r" % (e,), flush=True)
    return 0


main()

__all__ = ["enabled", "main"]
