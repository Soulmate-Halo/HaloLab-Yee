"""KAD for LMDeploy (PyTorch engine): native, the engine has one fixed draft count.

LMDeploy 0.18.0 builds every speculative shape from the constant ``num_speculative_tokens``
(``pytorch/config.py`` SpecDecodeConfig): ``strategies/ar_spec/engine.py`` decodes ``num_spec_tokens + 1`` tokens
per request, ``strategies/ar_spec/model_inputs.py`` and ``strategies/ar_spec/cudagraph.py`` size inputs and CUDA
graphs by it, ``spec_decode/spec_agent.py`` always runs ``num_spec_tokens - 1`` draft forwards, and
``step_inputs.py`` reshapes to ``(1, -1, num_spec_tokens + 1)``. A per-request or per-batch count would need
changes in all of these, so KAD cannot save work here and LMDeploy is native: run the 12-token launch
(``--speculative-num-draft-tokens 12`` / ``SpeculativeConfig(num_speculative_tokens=12)``).
``install()`` only reports this; it never patches LMDeploy.
"""
from __future__ import annotations

from .. import kad as _core

BASE = "lmdeploy"
VERIFIED = ("0.18.0",)
REASON = ("native: LMDeploy runs a fixed num_speculative_tokens for every batch (input shapes, CUDA graphs and "
          "the draft loop are built from it), so KAD cannot shorten drafting or verification")


def install(enabled=None):
    """Report that LMDeploy is native for KAD. Returns (False, reason)."""
    on = _core.env_enabled() if enabled is None else bool(enabled)
    if on:
        print("%s %s %s; launch with 12 speculative tokens" % (_core.TAG, BASE, REASON), flush=True)
    return False, REASON


def status() -> dict:
    return {"base": BASE, "installed": False, "mode": "native", "reason": REASON, "verified": list(VERIFIED)}


__all__ = ["BASE", "VERIFIED", "REASON", "install", "status"]
