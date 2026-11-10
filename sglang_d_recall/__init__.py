"""sglang-d-recall: an SGLang decode(D) side Recall speculative-draft accelerator.

Remembers the token ids the server read and wrote and copies drafts from them into SGLang's DSpark
spec-v2 decode path. Three interchangeable storage backends (disk / memory / cuda) sit behind the same
record logic; see ``sglang_d_recall.core`` for the semantics and README.md for the pick-a-backend guide.
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

__version__ = "1.2.0"

__all__ = [
    "RecallRecord",
    "RecallEngine",
    "Proposal",
    "tokenizer_identity",
    "engine_from_env",
    "get_engine",
    "__version__",
    "RecallIdentityError",
    "RecallBackendError",
    "BACKENDS",
    "CHUNK_IDS",
    "DEFAULT_DEVICE",
    "DEFAULT_INDEX_DENSITY",
    "DEDUP_RUN",
    "DEDUP_RUN_HEAT",
    "ENV_NEW",
    "ENV_OLD",
    "EVICT_MODES",
    "HEAT_SEG",
    "HOT_FLAG",
    "MAX_BACK",
    "MIN_TAIL",
    "OVERLAP",
    "SENTINEL",
    "WINDOW",
    "tier_of",
]
