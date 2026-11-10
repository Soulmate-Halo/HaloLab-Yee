"""Yee (formerly recall-accel): one Recall speculative-draft accelerator for SGLang, vLLM, TensorFold, cuteafd
and mlx-serve.

Remembers the token ids a server read and wrote and copies drafts from them into whatever speculative head the
server runs (SGLang DSPARK / DFLASH / EAGLE / EAGLE3 / NEXTN-MTP / STANDALONE / UNO / NGRAM, vLLM V2 DSpark /
DFlash / EAGLE / MTP); TensorFold has Recall built in and gets launch flags, record conversion and the
recall-after-priced patch; cuteafd (DeepSeek V4.1 native serve) and mlx-serve (M5 DSpark) get source patches
(``yee patch``). The target model verifies every copied id, so output is unchanged. ``yee auto`` picks the
adapter, ``yee doc`` shows the built-in manual, ``yee bases`` the base x head table.

Environment: every ``YEE_*`` variable is also visible as ``RECALL_ACCEL_*`` (copied below when that name is
unset), and ``YEE`` as ``RECALL_ACCEL``; the old names keep working.
"""
from __future__ import annotations

import os as _os


def _alias_env(environ=None) -> None:
    env = _os.environ if environ is None else environ
    for k, v in list(env.items()):
        if k == "YEE":
            env.setdefault("RECALL_ACCEL", v)
        elif k.startswith("YEE_"):
            env.setdefault("RECALL_ACCEL_" + k[4:], v)


_alias_env()

from .core import (  # noqa: F401,E402
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

__version__ = "2.2.0"

__all__ = [
    "RecallRecord", "RecallEngine", "Proposal", "tokenizer_identity", "engine_from_env", "get_engine",
    "__version__", "RecallIdentityError", "RecallBackendError", "BACKENDS", "CHUNK_IDS", "DEFAULT_DEVICE",
    "DEFAULT_INDEX_DENSITY", "DEDUP_RUN", "DEDUP_RUN_HEAT", "ENV_NEW", "ENV_OLD", "ENV_OLDER", "EVICT_MODES",
    "HEAT_SEG", "HOT_FLAG", "MAX_BACK", "MIN_TAIL", "OVERLAP", "SENTINEL", "WINDOW", "tier_of",
]
