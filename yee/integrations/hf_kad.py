"""KAD for Hugging Face transformers assisted generation: per-request adaptive draft count, max 12.

transformers verifies one round of draft tokens with one target forward (``GenerationMixin._assisted_decoding``,
batch size 1). The candidate generator decides how many tokens a round drafts:

* ``num_assistant_tokens``: AssistedCandidateGenerator and its subclasses (early exit, different tokenizers,
  universal speculative decoding) and SinglePositionMultiTokenCandidateGenerator (Gemma assistants);
* ``num_output_tokens``: PromptLookupCandidateGenerator (``prompt_lookup_num_tokens``).

KAD replaces the built-in schedule for one generate() call (= one request): the round's count comes from the
ladder (5, 7, 9, 11, 12 while everything is accepted; max(5, accepted + 1) otherwise), so the assistant really
drafts k tokens and the target really verifies k + 1. The assistant's own early stop
(``assistant_confidence_threshold``) still applies below that cap. MTPCandidateGenerator (``use_mtp``) and
DFlashTokenCandidateGenerator (``speculation_type="dflash"``) draft a model-defined count in one forward:
native, statistics only.

Opt in per call with ``num_assistant_tokens_schedule="yee_kad"`` (in generate() or in the assistant's
generation_config), or for every assisted generation of the process with ``YEE_KAD=1``; an explicit
``YEE_KAD=0`` turns it off everywhere. ``install()`` wraps ``GenerationMixin._get_candidate_generator`` once
(no change for calls that do not opt in). ``enable(assistant_model)`` installs and sets the assistant's
generation_config to 12 tokens with the yee_kad schedule.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import itertools
import os
import sys
import threading
import weakref

from .. import kad as _core

BASE = "transformers"
SCHEDULE = "yee_kad"
TARGET = "transformers.generation.utils"
CANDIDATES = "transformers.generation.candidate_generator"
VERIFIED = ("5.22.2",)
MARK = "__yee_kad__"
NATIVE = ("MTPCandidateGenerator", "DFlashTokenCandidateGenerator")

_KAD = None
_ORIG = None
_LOCK = threading.RLock()
_IDS = itertools.count(1)
_WARNED = set()
_NOTED = set()


def kad():
    """The KAD state of this process (created on first use)."""
    global _KAD
    with _LOCK:
        if _KAD is None:
            _KAD = _core.Kad(BASE, _core.KMAX, enabled=True)
        return _KAD


def _env_choice():
    """True / False when YEE_KAD is set, None when it is not."""
    raw = os.environ.get(_core.ENV_KAD)
    if raw is None or not raw.strip():
        return None
    return _core.env_enabled()


def active(generation_config=None, candidate_generator=None) -> bool:
    """Whether KAD applies to one generate() call."""
    env = _env_choice()
    if env is not None:
        return env
    if getattr(generation_config, "num_assistant_tokens_schedule", None) == SCHEDULE:
        return True
    assistant = getattr(candidate_generator, "assistant_model", None)
    cfg = getattr(assistant, "generation_config", None)
    return getattr(cfg, "num_assistant_tokens_schedule", None) == SCHEDULE


def _note_once(key, text) -> None:
    if key not in _NOTED:
        _NOTED.add(key)
        print("%s %s %s" % (_core.TAG, BASE, text), flush=True)


def _warn_launch(k: int) -> None:
    if k != _core.KMAX and k not in _WARNED:
        _WARNED.add(k)
        print("%s %s WARNING launch K=%d, KAD needs %d (per-request width is capped at min(%d, K))"
              % (_core.TAG, BASE, k, _core.KMAX, _core.KMAX), flush=True)


def _observe(k, rid, scores, num_matches):
    """v = candidates verified this round (target logits are v + 1), a = accepted. Returns v."""
    v = 0
    try:
        v = int(len(scores[0])) - 1 if scores is not None else 0
        if v > 0:
            k.observe(rid, v, max(0, min(int(num_matches), v)))
    except Exception as e:  # noqa: BLE001
        k.error("observe", e)
    return v


def _release(rid) -> None:
    k = _KAD
    if k is not None:
        try:
            k.release(rid)
        except Exception:  # noqa: BLE001  runs from a finalizer
            pass


def attach(cg, generation_config=None) -> str:
    """Put KAD on one candidate generator. Returns "kad", "native" or "off"."""
    if cg is None or getattr(cg, MARK, None):
        return "off"
    if not active(generation_config, cg):
        return "off"
    k = kad()
    cls = type(cg)
    name = cls.__name__
    attr = None
    if hasattr(cg, "num_assistant_tokens"):
        attr = "num_assistant_tokens"
    elif hasattr(cg, "num_output_tokens"):
        attr = "num_output_tokens"
    rid = next(_IDS)
    ref = weakref.ref(cg)
    cls_upd = cls.update_candidate_strategy
    if name in NATIVE or attr is None:
        _note_once(name, "%s: native (draft count fixed by the model), statistics only" % name)

        def update_native(input_ids, scores, num_matches, *args, **kwargs):
            obj = ref()
            try:
                k.begin_step()
                v = _observe(k, rid, scores, num_matches)
                if v > 0:
                    k.note_batch([v])
            except Exception as e:  # noqa: BLE001
                k.error("stats", e)
            return cls_upd(obj, input_ids, scores, num_matches, *args, **kwargs)

        cg.update_candidate_strategy = update_native
        setattr(cg, MARK, "native")
        weakref.finalize(cg, _release, rid)
        return "native"

    launch = int(getattr(cg, attr))
    _warn_launch(launch)
    cfg = getattr(cg, "assistant_generation_config", None)  # a per-call deepcopy in transformers
    if cfg is not None and getattr(cfg, "num_assistant_tokens_schedule", None) in ("heuristic", "heuristic_transient"):
        cfg.num_assistant_tokens_schedule = "constant"  # KAD replaces the built-in schedule for this call
    cls_get = cls.get_candidates

    def get_candidates(input_ids, *args, **kwargs):
        obj = ref()
        w = launch
        try:
            k.begin_step()
            w = max(1, min(int(k.width(rid)), launch))
            k.note_batch([w])
        except Exception as e:  # noqa: BLE001  keep the launch count
            k.error("width", e)
            w = launch
        setattr(obj, attr, w)
        try:
            return cls_get(obj, input_ids, *args, **kwargs)
        finally:
            setattr(obj, attr, launch)  # transformers persists num_assistant_tokens for "heuristic"

    def update_candidate_strategy(input_ids, scores, num_matches, *args, **kwargs):
        obj = ref()
        _observe(k, rid, scores, num_matches)
        return cls_upd(obj, input_ids, scores, num_matches, *args, **kwargs)

    cg.get_candidates = get_candidates
    cg.update_candidate_strategy = update_candidate_strategy
    setattr(cg, MARK, "kad")
    weakref.finalize(cg, _release, rid)
    return "kad"


def _skip(reason: str):
    print("%s %s not installed: %s" % (_core.TAG, BASE, reason), flush=True)
    return False, reason


def check() -> list:
    """Structure checks on the installed transformers: [(name, ok, detail)]."""
    rows = []

    def row(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, repr(e)
        rows.append((name, bool(ok), detail))

    utils = importlib.import_module(TARGET)
    cands = importlib.import_module(CANDIDATES)
    mixin = utils.GenerationMixin

    def has_hook():
        fn = mixin.__dict__.get("_get_candidate_generator")
        fn = getattr(fn, "__wrapped__", fn)
        return fn is not None and "generation_config" in inspect.signature(fn).parameters, "signature"

    def loop():
        src = inspect.getsource(mixin._assisted_decoding)
        need = ("candidate_generator.get_candidates(", "candidate_generator.update_candidate_strategy(")
        miss = [s for s in need if s not in src]
        return not miss, "missing %s" % (miss,) if miss else "get_candidates / update_candidate_strategy"

    def uses(cname, needle):
        def fn():
            cls = getattr(cands, cname)
            return needle in inspect.getsource(cls), needle
        return fn

    row("GenerationMixin._get_candidate_generator(generation_config, ...)", has_hook)
    row("_assisted_decoding calls the candidate generator", loop)
    row("AssistedCandidateGenerator uses num_assistant_tokens", uses("AssistedCandidateGenerator", "self.num_assistant_tokens"))
    row("PromptLookupCandidateGenerator uses num_output_tokens", uses("PromptLookupCandidateGenerator", "self.num_output_tokens"))
    return rows


def install(enabled=None):
    """Wrap GenerationMixin._get_candidate_generator once. Returns (installed, reason)."""
    global _ORIG
    if enabled is False or (enabled is None and _env_choice() is False):
        return False, "YEE_KAD=0"
    try:
        utils = importlib.import_module(TARGET)
        mixin = utils.GenerationMixin
    except Exception as e:  # noqa: BLE001  torch / transformers missing or broken
        return _skip("%s not importable: %r" % (TARGET, e))
    with _LOCK:
        cur = mixin.__dict__.get("_get_candidate_generator")
        if cur is None:
            return _skip("GenerationMixin._get_candidate_generator missing")
        if getattr(cur, MARK, False):
            return True, "already installed"
        try:
            bad = [name for name, ok, _ in check() if not ok]
        except Exception as e:  # noqa: BLE001
            bad = ["structure check: %r" % (e,)]
        if bad:
            return _skip("structure changed: %s" % "; ".join(bad))
        orig = cur

        @functools.wraps(orig)
        def _get_candidate_generator(self, generation_config, *args, **kwargs):
            cg = orig(self, generation_config, *args, **kwargs)
            try:
                attach(cg, generation_config)
            except Exception as e:  # noqa: BLE001  never break generate()
                kad().error("attach", e)
            return cg

        setattr(_get_candidate_generator, MARK, True)
        mixin._get_candidate_generator = _get_candidate_generator
        _ORIG = (mixin, orig)
    version = getattr(importlib.import_module("transformers"), "__version__", "?")
    print("%s %s installed on transformers %s%s; opt in with num_assistant_tokens_schedule=%r or YEE_KAD=1"
          % (_core.TAG, BASE, version, "" if version in VERIFIED else " (not in verified %s)" % (VERIFIED,),
             SCHEDULE), flush=True)
    return True, "patched GenerationMixin._get_candidate_generator"


def enable(assistant_model=None, num_tokens: int = _core.KMAX):
    """install() and, for an assistant model, set num_assistant_tokens=12 with the yee_kad schedule."""
    ok, why = install(enabled=True)
    if assistant_model is not None:
        cfg = assistant_model.generation_config
        cfg.num_assistant_tokens = int(num_tokens)
        cfg.num_assistant_tokens_schedule = SCHEDULE
    return ok, why


def uninstall() -> bool:
    global _ORIG
    with _LOCK:
        if _ORIG is None:
            return False
        mixin, orig = _ORIG
        mixin._get_candidate_generator = orig
        _ORIG = None
        return True


def installed() -> bool:
    module = sys.modules.get(TARGET)
    if module is None:
        return False
    return bool(getattr(module.GenerationMixin.__dict__.get("_get_candidate_generator"), MARK, False))


def status() -> dict:
    out = {"base": BASE, "installed": installed(), "verified": list(VERIFIED), "schedule": SCHEDULE}
    if _KAD is not None:
        out["kad"] = _KAD.snapshot()
    return out


__all__ = ["BASE", "SCHEDULE", "VERIFIED", "NATIVE", "active", "attach", "check", "install", "enable",
           "uninstall", "installed", "status", "kad"]
