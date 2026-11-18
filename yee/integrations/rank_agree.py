"""Yee 4.2: agree across the TP ranks before a rank-coupled feature runs.

The DSpark Recall hook broadcasts rank 0's copy plan to every TP rank, and the hot-expert verify budget must pick
the same experts on every rank. If one host runs without Yee (or with another version) those ranks never join, the
collectives go out of step and the server hangs a couple of steps after the first request. A rank without Yee
cannot take part in any collective we add, so the check uses only the key/value store of the default process group
(no collective): every rank that has Yee writes ``yee/<version>/<feature>/<ranks>/r<rank>``, waits until all ranks
of its group have written theirs, and the first rank to see them all (or to time out) fixes the verdict with an
atomic ``compare_set``. Every Yee rank then reads the same verdict, so they all turn the feature on or all leave it
off. Ranks without Yee are unaffected: with the feature off nothing extra is sent.
"""
from __future__ import annotations

import time
from datetime import timedelta

VERSION = "4.2.0"
_CACHE = {}


def default_store():
    """The store of torch's default process group, or None (no torch / not initialised)."""
    try:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return None
        from torch.distributed import distributed_c10d as c10d

        fn = getattr(c10d, "_get_default_store", None)
        if fn is not None:
            return fn()
    except Exception:  # noqa: BLE001
        return None
    return None


def _has(store, key):
    try:
        check = getattr(store, "check", None)
        if check is not None:
            return bool(check([key]))
    except Exception:  # noqa: BLE001
        pass
    try:
        store.wait([key], timedelta(milliseconds=20))
        return True
    except Exception:  # noqa: BLE001
        return False


def _get(store, key):
    v = store.get(key)
    return v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else str(v)


def _decide(store, key, value):
    """Atomically set ``key`` when absent; returns what the store holds afterwards."""
    cas = getattr(store, "compare_set", None)
    if cas is not None:
        try:
            out = cas(key, "", value)
            return out.decode("utf-8", "replace") if isinstance(out, (bytes, bytearray)) else str(out)
        except Exception:  # noqa: BLE001
            pass
    if not _has(store, key):  # no compare_set: last writer wins between near-simultaneous deciders
        store.set(key, value)
    return _get(store, key)


def agree(feature, ranks, my_rank, timeout_s, store=None, poll_s=0.05):
    """(ok, reason). Cached per feature: one decision per process."""
    hit = _CACHE.get(feature)
    if hit is not None:
        return hit
    try:
        res = _agree(feature, ranks, my_rank, timeout_s, store, poll_s)
    except Exception as e:  # noqa: BLE001  a broken store must not take serving down; same answer on every rank
        res = (False, "rank check error: %r" % (e,))
    _CACHE[feature] = res
    return res


def _agree(feature, ranks, my_rank, timeout_s, store, poll_s):
    ranks = sorted(set(int(r) for r in ranks))
    if len(ranks) <= 1:
        return True, "single rank"
    if store is None:
        store = default_store()
    if store is None:
        return True, "no distributed store (single process)"
    prefix = "yee/%s/%s/%s/" % (VERSION, feature, "-".join(str(r) for r in ranks))
    store.set(prefix + "r%d" % int(my_rank), "1")
    verdict_key = prefix + "verdict"
    deadline = time.monotonic() + float(timeout_s)
    while True:
        if _has(store, verdict_key):
            verdict = _get(store, verdict_key)
            break
        missing = [r for r in ranks if not _has(store, prefix + "r%d" % r)]
        if not missing:
            verdict = _decide(store, verdict_key, "on")
            break
        if time.monotonic() >= deadline:
            verdict = _decide(store, verdict_key, "off:" + ",".join(str(r) for r in missing))
            break
        time.sleep(poll_s)
    if verdict == "on":
        return True, "%d ranks" % len(ranks)
    miss = verdict.split(":", 1)[1] if ":" in verdict else "?"
    return False, ("ranks %s did not report Yee %s within %gs (install the same Yee on every host)"
                   % (miss, VERSION, float(timeout_s)))


def tp_ranks():
    """(global ranks of this process's TP group, my global rank), or (None, None) outside a distributed run."""
    try:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return None, None
        me = int(dist.get_rank())
        try:
            from sglang.srt.distributed import get_tp_group

            g = get_tp_group()
            ranks = list(getattr(g, "ranks", None) or [])
            if ranks:
                return [int(r) for r in ranks], me
        except Exception:  # noqa: BLE001
            pass
        return list(range(int(dist.get_world_size()))), me
    except Exception:  # noqa: BLE001
        return None, None


def reset():
    """Tests only."""
    _CACHE.clear()


__all__ = ["agree", "default_store", "tp_ranks", "reset", "VERSION"]
