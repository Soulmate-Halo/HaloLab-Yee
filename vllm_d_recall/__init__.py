"""vllm-d-recall: a Recall speculative-draft accelerator for vLLM's DSpark / DFlash decode path.

Remembers the token ids the server read and wrote and copies drafts from them into vLLM's V2 GPU model
runner (``vllm/v1/worker/gpu``) DSpark / DFlash speculator. The record logic (``vllm_d_recall.core``) is the
same as sglang-d-recall 1.2.0; see README.md.
Shipped in Yee 4.0.0 as its vLLM adapter (KAD: per-request adaptive draft width up to K_max = 12 for
every vLLM draft head).
"""
from __future__ import annotations

from .core import (  # noqa: F401
    BACKENDS,
    CHUNK_IDS,
    DEFAULT_DEVICE,
    DEFAULT_INDEX_DENSITY,
    DEDUP_RUN,
    DEDUP_RUN_HEAT,
    ENV_NEW,
    ENV_OLD,
    ENV_OLDER,
    EVICT_MODES,
    HEAT_SEG,
    HOT_FLAG,
    MAX_BACK,
    MIN_TAIL,
    OVERLAP,
    SENTINEL,
    WINDOW,
    Proposal,
    RecallBackendError,
    RecallEngine,
    RecallIdentityError,
    RecallRecord,
    engine_from_env,
    get_engine,
    tier_of,
    tokenizer_identity,
)

__version__ = "4.3.0"

__all__ = [
    "RecallRecord", "RecallEngine", "Proposal", "tokenizer_identity", "engine_from_env", "get_engine",
    "__version__", "RecallIdentityError", "RecallBackendError", "BACKENDS", "CHUNK_IDS", "DEFAULT_DEVICE",
    "DEFAULT_INDEX_DENSITY", "DEDUP_RUN", "DEDUP_RUN_HEAT", "ENV_NEW", "ENV_OLD", "ENV_OLDER", "EVICT_MODES",
    "HEAT_SEG", "HOT_FLAG", "MAX_BACK", "MIN_TAIL", "OVERLAP", "SENTINEL", "WINDOW", "tier_of",
]
