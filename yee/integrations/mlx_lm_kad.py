"""KAD for mlx-lm speculative decoding: per-request adaptive draft count, max 12.

mlx-lm drafts ``num_draft_tokens`` tokens every round in ``mlx_lm.generate.speculative_generate_step`` and
verifies them with one target forward of ``num_draft + 1`` tokens. ``mlx_lm.generate --draft-model`` and
``mlx_lm.server --draft-model`` both reach that function through ``stream_generate`` (the server never batches
requests that use a draft model), and one call of it is one request.

``install()`` rebuilds that function from its own source with five line edits and swaps it in:

1. the definition is renamed (the module attribute is replaced only after the build succeeded);
2. a request id is taken right before ``with mx.stream(stream):``;
3. the per-round draft count ``min(max_tokens - ntoks, num_draft_tokens)`` gets the KAD width as a third term;
4. right before the bonus token (``if ntoks < max_tokens:``) the round is reported: v drafted, a accepted;
5. the request state is released in the ``finally:`` that rewinds the caches.

Every anchor must match exactly once, otherwise nothing changes and the reason is printed. The draft model
really runs k steps and the target really verifies k + 1 tokens, so low-acceptance requests draft less and
copy-like requests climb to 12. Greedy output is unchanged (verification is untouched).

Switch: ``YEE_KAD=1`` (``install(enabled=True)`` forces it on). Launch K is the call's ``num_draft_tokens``
(``--num-draft-tokens 12``; mlx-lm's default is 3); another value prints one warning and caps the width.
CLI wrapper (KAD on unless ``YEE_KAD=0``): ``python -m yee.integrations.mlx_lm_kad server|generate <mlx_lm args>``.
"""
from __future__ import annotations

import __future__
import importlib
import inspect
import itertools
import os
import sys
import textwrap
import threading

from .. import kad as _core

BASE = "mlx-lm"
TARGET = "mlx_lm.generate"
FUNC = "speculative_generate_step"
VERIFIED = ("0.32.2",)
MARK = "__yee_kad__"
NEEDED = ("prompt", "model", "draft_model", "num_draft_tokens", "max_tokens")
HELPERS = ("_yee_kad_begin", "_yee_kad_width", "_yee_kad_observe", "_yee_kad_end")

_FUTURE = 0
for _name in __future__.all_feature_names:
    _FUTURE |= getattr(__future__, _name).compiler_flag

_KAD = None
_ORIG = {}
_LOCK = threading.RLock()
_IDS = itertools.count(1)
_WARNED = set()


class AnchorError(RuntimeError):
    """The upstream source no longer has one of the lines the edit relies on."""


def kad():
    """The KAD state of this process (created on first use, always enabled once installed)."""
    global _KAD
    with _LOCK:
        if _KAD is None:
            _KAD = _core.Kad(BASE, _core.KMAX, enabled=True)
        return _KAD


# ------------------------------------------------------------------------------------------- source edit
def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _unique(lines, pred, what) -> int:
    hits = [i for i, line in enumerate(lines) if pred(i, line)]
    if len(hits) != 1:
        raise AnchorError("%s: %d matches" % (what, len(hits)))
    return hits[0]


def rewrite(src: str) -> str:
    """The KAD version of the ``speculative_generate_step`` source (AnchorError when an anchor is not unique)."""
    head = "def %s(" % FUNC
    if src.count(head) != 1:
        raise AnchorError("%s: %d matches" % (head, src.count(head)))
    src = src.replace(head, "def _yee_kad_%s(" % FUNC)
    old = "num_draft = min(max_tokens - ntoks, num_draft_tokens)"
    if src.count(old) != 1:
        raise AnchorError("%s: %d matches" % (old, src.count(old)))
    src = src.replace(old, "num_draft = min(max_tokens - ntoks, num_draft_tokens, "
                           "_yee_kad_width(_yk_rid, num_draft_tokens))")
    lines = src.split("\n")
    i = _unique(lines, lambda _, s: s.strip() == "with mx.stream(stream):", "with mx.stream(stream):")
    lines.insert(i, _indent(lines[i]) + "_yk_rid = _yee_kad_begin(num_draft_tokens)")
    i = _unique(lines, lambda _, s: s.strip() == "if ntoks < max_tokens:", "if ntoks < max_tokens:")
    lines.insert(i, _indent(lines[i]) + "_yee_kad_observe(_yk_rid, num_draft, n)")
    i = _unique(lines, lambda j, s: s.strip() == "finally:" and j + 1 < len(lines)
                and lines[j + 1].strip() == "_rewind_cache(num_draft, n)", "finally: _rewind_cache(num_draft, n)")
    lines.insert(i + 1, _indent(lines[i + 1]) + "_yee_kad_end(_yk_rid)")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- runtime hooks
def _err(what, e) -> None:
    k = _KAD
    if k is not None:
        k.error(what, e)


def _begin(launch_k) -> int:
    """A new request: its id. Warns once per launch K other than 12."""
    try:
        k = int(launch_k)
        if k != _core.KMAX and k not in _WARNED:
            _WARNED.add(k)
            print("%s %s WARNING launch K=%d, KAD needs %d (per-request width is capped at min(%d, K))"
                  % (_core.TAG, BASE, k, _core.KMAX, _core.KMAX), flush=True)
    except Exception as e:  # noqa: BLE001  never break generation
        _err("begin", e)
    return next(_IDS)


def _width(rid, launch_k) -> int:
    """The draft count of this round: min(KAD width, launch K); the launch K on any error."""
    try:
        launch = int(launch_k)
    except (TypeError, ValueError):
        return launch_k
    try:
        k = kad()
        k.begin_step()
        w = max(0, min(int(k.width(rid)), launch))
        k.note_batch([w])
        return w
    except Exception as e:  # noqa: BLE001
        _err("width", e)
        return launch


def _observe(rid, v, a) -> None:
    try:
        if int(v) > 0:
            kad().observe(rid, int(v), int(a))
    except Exception as e:  # noqa: BLE001
        _err("observe", e)


def _end(rid) -> None:
    try:
        kad().release(rid)
    except Exception as e:  # noqa: BLE001
        _err("release", e)


# ------------------------------------------------------------------------------------------- install
def _skip(reason: str):
    print("%s %s not installed: %s" % (_core.TAG, BASE, reason), flush=True)
    return False, reason


def install_into(module, func: str = FUNC):
    """Swap the KAD version of ``module.<func>`` in. Returns (installed, reason)."""
    with _LOCK:
        cur = getattr(module, func, None)
        if cur is None or not callable(cur):
            return _skip("%s.%s missing" % (module.__name__, func))
        if getattr(cur, MARK, False):
            return True, "already installed"
        try:
            params = inspect.signature(cur).parameters
            missing = [p for p in NEEDED if p not in params]
            if missing:
                return _skip("%s.%s signature lacks %s" % (module.__name__, func, ", ".join(missing)))
            src = textwrap.dedent(inspect.getsource(cur))
            new_src = rewrite(src)
            first = cur.__code__.co_firstlineno
            filename = getattr(module, "__file__", None) or "<%s>" % module.__name__
            code = compile("\n" * (first - 1) + new_src, filename, "exec",
                           flags=cur.__code__.co_flags & _FUTURE, dont_inherit=True)
        except AnchorError as e:
            return _skip("source anchors changed (%s)" % (e,))
        except (OSError, TypeError, SyntaxError, ValueError) as e:
            return _skip("source of %s.%s unusable: %r" % (module.__name__, func, e))
        g = module.__dict__
        g.update({"_yee_kad_begin": _begin, "_yee_kad_width": _width,
                  "_yee_kad_observe": _observe, "_yee_kad_end": _end})
        exec(code, g)  # noqa: S102  the upstream function's own source with the five edits above
        fn = g.pop("_yee_kad_" + func)
        fn.__name__ = cur.__name__
        fn.__qualname__ = cur.__qualname__
        fn.__wrapped__ = cur
        setattr(fn, MARK, True)
        _ORIG[(module.__name__, func)] = (module, cur)
        setattr(module, func, fn)
        kad()
        return True, "patched %s.%s" % (module.__name__, func)


def install(enabled=None):
    """Install on mlx_lm.generate when YEE_KAD=1 (or enabled=True). Returns (installed, reason)."""
    on = _core.env_enabled() if enabled is None else bool(enabled)
    if not on:
        return False, "YEE_KAD is off"
    try:
        module = importlib.import_module(TARGET)
    except Exception as e:  # noqa: BLE001  mlx / mlx_lm missing or broken
        return _skip("%s not importable: %r" % (TARGET, e))
    version = getattr(importlib.import_module("mlx_lm"), "__version__", "?")
    ok, why = install_into(module)
    if ok and why.startswith("patched"):
        k = kad()
        print("%s %s installed on mlx_lm %s%s (%s); policy %s" % (
            _core.TAG, BASE, version, "" if version in VERIFIED else " (not in verified %s)" % (VERIFIED,),
            why, k.policy_file), flush=True)
    return ok, why


def uninstall() -> bool:
    """Put the original functions back (requests already running keep the KAD version until they end)."""
    with _LOCK:
        done = False
        for (_, func), (module, orig) in list(_ORIG.items()):
            setattr(module, func, orig)
            done = True
        _ORIG.clear()
        return done


def installed() -> bool:
    module = sys.modules.get(TARGET)
    return bool(module is not None and getattr(getattr(module, FUNC, None), MARK, False))


def status() -> dict:
    out = {"base": BASE, "installed": installed(), "verified": list(VERIFIED)}
    if _KAD is not None:
        out["kad"] = _KAD.snapshot()
    return out


def main(argv=None) -> int:
    """``python -m yee.integrations.mlx_lm_kad server|generate <mlx_lm arguments>``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("server", "generate"):
        print("usage: python -m yee.integrations.mlx_lm_kad server|generate <mlx_lm arguments>", file=sys.stderr)
        return 2
    cmd = args.pop(0)
    raw = os.environ.get(_core.ENV_KAD, "").strip()
    install(enabled=True if not raw else _core.env_enabled())
    module = importlib.import_module("mlx_lm." + cmd)
    sys.argv = ["mlx_lm." + cmd] + args
    rc = module.main()
    return int(rc) if isinstance(rc, int) else 0


__all__ = ["BASE", "VERIFIED", "AnchorError", "rewrite", "install", "install_into", "uninstall", "installed",
           "status", "kad", "main"]

if __name__ == "__main__":
    sys.exit(main())
