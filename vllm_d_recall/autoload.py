"""Autoload entry point for vllm-d-recall.

``import vllm_d_recall.autoload`` (the .pth written by ``vllm-d-recall install-hook`` does that) installs the
vLLM hooks at interpreter start, but only when VLLM_D_RECALL=1 (SGLANG_D_RECALL / DSV41_RECALL as fallback).
vLLM worker processes inherit the environment, so the hooks land in every worker. Any failure is printed and
the process keeps running: serving must never break because of the accelerator.
"""
from __future__ import annotations

from .core import ENV_NEW, ENV_OLD, _env

_TRUE = ("1", "true", "yes", "on")


def enabled() -> bool:
    return _env(ENV_NEW["on"], ENV_OLD["on"], "0").strip().lower() in _TRUE


def main() -> int:
    """Install the vLLM hooks when Recall is on (Yee 4.2: on by default unless VLLM_D_RECALL=0 / YEE=0). Never raises."""
    try:
        from yee.integrations import defaults

        defaults.apply()
    except Exception as e:  # noqa: BLE001  never take the process down
        print("[vllm-d-recall] defaults failed: %r" % (e,), flush=True)
    if not enabled():
        return 0
    try:
        from .integrations import vllm_dspark

        vllm_dspark.install()
    except Exception as e:  # noqa: BLE001  never take the process down
        print("[vllm-d-recall] install failed: %r" % (e,), flush=True)
    return 0


main()

__all__ = ["enabled", "main"]
