"""Autoload entry point for sglang-d-recall.

Put ``import sglang_d_recall.autoload`` on the PYTHONPATH (a .pth file does that) and the Recall hooks
are installed as soon as the interpreter starts, but only when Recall is switched on
(SGLANG_D_RECALL=1, DSV41_RECALL=1 as fallback). Any failure is printed and the process keeps running:
serving must never break because of the accelerator.
"""
from __future__ import annotations

import os

from .core import ENV_NEW, ENV_OLD, _env

_TRUE = ("1", "true", "yes", "on")


def enabled() -> bool:
    """True when SGLANG_D_RECALL (or the old DSV41_RECALL) switches the hooks on."""
    return _env(ENV_NEW["on"], ENV_OLD["on"], "0").strip().lower() in _TRUE


def main() -> int:
    """Install the SGLang hooks when Recall is on. Never raises."""
    if not enabled():
        return 0
    try:
        from .integrations import sglang_dspark

        sglang_dspark.install()
    except Exception as e:  # noqa: BLE001  never take the process down
        print("[sglang-d-recall] install failed: %r" % (e,), flush=True)
    return 0


main()

__all__ = ["enabled", "main"]
