"""Recall: a persistent record of the token ids a server has read and written, and copy drafts from it.

Engine agnostic (numpy only). The semantics follow TensorFold's docs/recall.md:

* the record keeps token ids only, in chunk files of up to 1,048,576 ids (int32, little endian) under one
  directory (0700, files 0600), beside an ``identity`` file naming the tokenizer and a ``lock`` file;
* each request adds the part of its prompt that no earlier request read, and its reply when the reply ends;
  runs of 128 or more ids the record already holds are not added again, so a request sent twice adds nothing;
* past the size budget the oldest chunk file goes first; with less than ``min_free_gib`` free nothing is added;
* a copy is offered when the record's continuation of the stream's last 16 or more tokens is expected to land
  more tokens than the stream's own drafts have lately; expectations are kept apart for matches of 16-31,
  32-63 and 64+ tokens; a stream whose copy landed nothing leaves the record alone for four rounds; while
  copies keep missing only probes are offered, rarer each time a probe misses;
* the stream's own context is asked first and wins ties.

* the target model verifies every copied token, so replies are the same with Recall on, off or read-only.

Eviction (``SGLANG_D_RECALL_EVICT`` / ``RECALL_EVICT``): ``fifo`` is the behaviour above, unchanged.
``heat`` (default) keeps a decayed heat score per 256-id segment (tokens the target accepted from a copy of
it, plus a fraction of the duplicate ids an add skipped because it already held them) and copies hot segments
into a separate hot zone that outlives the FIFO ring; the hot zone drops its lowest-heat entry when full.
Hot entries are exact copies of recorded ids, so the verification guarantee is untouched.

This package adds three interchangeable storage backends behind the very same record logic:

* ``disk``   - chunk files in a directory (the original behaviour, unchanged);
* ``memory`` - numpy int32 blocks in the process, nothing is written to disk;
* ``cuda``   - torch int32 tensors on a GPU device, the 16-gram index stays on the host.

All three keep the same add/lookup/continuation/clear/stats semantics; ``tests/test_backends_equiv.py``
compares them on random sequences.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import queue
import re
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

try:  # POSIX advisory lock; Windows (tests only) runs without it
    import fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

CHUNK_IDS = 1 << 20
SENTINEL = -1
WINDOW = 16
DEDUP_RUN = 128
OVERLAP = 32
MAX_BACK = 128
MIN_TAIL = 4  # a window is indexed only when at least this many ids of its segment follow it
# heat eviction (evict mode "heat"): the record is cut into HEAT_SEG-id segments by global position; a segment
# whose decayed heat reaches the promote threshold is copied (with HOT_PRE ids before and up to HOT_POST after)
# into a hot zone that survives the FIFO eviction of the cold ring and is itself evicted lowest-heat first.
HEAT_SEG = 256
HOT_PRE = WINDOW
HOT_POST = 64
HOT_SLOT = HEAT_SEG + HOT_PRE + HOT_POST  # one hot entry never holds more ids than this
HOT_FLAG = 1 << 62  # positions >= HOT_FLAG point into the hot zone: HOT_FLAG + (uid << HOT_OFF_BITS) + offset
HOT_OFF_BITS = 9
# A skipped duplicate run of R ids still costs 2*OVERLAP re-stored ids plus a sentinel (both kept neighbours are
# widened into it), so R < 2*OVERLAP+2 = 66 stores more than it skips; 80 = 2*OVERLAP+WINDOW nets >= 15 ids.
DEDUP_RUN_HEAT = 2 * OVERLAP + WINDOW
EVICT_MODES = ("heat", "fifo")
MASK64 = (1 << 64) - 1
_P = 0x9E3779B97F4A7C15
_P15 = pow(_P, WINDOW - 1, 1 << 64)
_M1 = 0xBF58476D1CE4E5B9
_CHUNK_RE = re.compile(r"^chunk-(\d{8})\.ids$")
_TRUE = ("1", "true", "yes", "on")
BACKENDS = ("disk", "memory", "cuda")
DEFAULT_DEVICE = "cuda:0"
# index density: how many hash slots one stored id gets. The 16-gram index is a direct-mapped hash table
# (every position is inserted, a later window overwrites an older one in the same slot), so a sparse table
# loses old windows once the pool is full. 0.25 slots per id is the 1.1.0 behaviour; the value is clamped.
DEFAULT_INDEX_DENSITY = 0.25
MIN_INDEX_DENSITY = 1.0 / 64.0
MAX_INDEX_DENSITY = 4.0

# New environment variable names first, the old ones stay as fallback so an existing deployment keeps
# working. Every read goes through :func:`_env` (new name wins, old name second).
ENV_NEW = {
    "on": "SGLANG_D_RECALL",
    "dir": "SGLANG_D_RECALL_DIR",
    "gib": "SGLANG_D_RECALL_GIB",
    "backend": "SGLANG_D_RECALL_BACKEND",
    "device": "SGLANG_D_RECALL_DEVICE",
    "min_match": "SGLANG_D_RECALL_MIN_MATCH",
    "prior": "SGLANG_D_RECALL_PRIOR",
    "read_only": "SGLANG_D_RECALL_READ_ONLY",
    "min_free_gib": "SGLANG_D_RECALL_MIN_FREE_GIB",
    "tokenizer_dir": "SGLANG_D_RECALL_TOKENIZER_DIR",
    "model_name": "SGLANG_D_RECALL_MODEL_NAME",
    "mia_sitecustomize": "SGLANG_D_RECALL_MIA_SITECUSTOMIZE",
    "evict": "SGLANG_D_RECALL_EVICT",
    "hot_frac": "SGLANG_D_RECALL_HOT_FRAC",
    "dedup_run": "SGLANG_D_RECALL_DEDUP_RUN",
    "heat_halflife": "SGLANG_D_RECALL_HEAT_HALFLIFE_IDS",
    "heat_seen_weight": "SGLANG_D_RECALL_HEAT_SEEN_WEIGHT",
    "promote_tokens": "SGLANG_D_RECALL_PROMOTE_TOKENS",
    "index_density": "SGLANG_D_RECALL_INDEX_DENSITY",
}
ENV_OLD = {
    "on": "DSV41_RECALL",
    "dir": "DSV41_RECALL_DIR",
    "gib": "DSV41_RECALL_GIB",
    "backend": "RECALL_BACKEND",
    "device": "RECALL_DEVICE",
    "min_match": "DSV41_RECALL_MIN_MATCH",
    "prior": "DSV41_RECALL_PRIOR",
    "read_only": "DSV41_RECALL_READ_ONLY",
    "min_free_gib": "DSV41_RECALL_MIN_FREE_GIB",
    "tokenizer_dir": "DSV41_RECALL_TOKENIZER_DIR",
    "model_name": "DSV41_RECALL_MODEL_NAME",
    "mia_sitecustomize": "DSV41_RECALL_MIA_SITECUSTOMIZE",
    "evict": "RECALL_EVICT",
    "hot_frac": "RECALL_HOT_FRAC",
    "dedup_run": "RECALL_DEDUP_RUN",
    "heat_halflife": "RECALL_HEAT_HALFLIFE_IDS",
    "heat_seen_weight": "RECALL_HEAT_SEEN_WEIGHT",
    "promote_tokens": "RECALL_PROMOTE_TOKENS",
    "index_density": "RECALL_INDEX_DENSITY",
}


def _env(new: str, old: str, default: str = "", environ=None) -> str:
    """Read one environment variable: the new name wins, the old name is the fallback.

    An empty or blank value counts as "not set", so the fallback and the default still apply.
    """
    env = os.environ if environ is None else environ
    for name in (new, old):
        if not name:
            continue
        v = env.get(name)
        if v is None:
            continue
        v = str(v).strip()
        if v:
            return v
    return default


class RecallIdentityError(RuntimeError):
    """The record was written by a different tokenizer."""


class RecallBackendError(RuntimeError):
    """The requested storage backend cannot be used (e.g. no torch, no CUDA)."""


# ----------------------------------------------------------------------------------------------- hashing
def _hash_window_int(tokens) -> int:
    h = 0
    for t in tokens:
        h = (h * _P + int(t) + 1) & MASK64
    return h


def _roll_int(h: int, out_tok: int, in_tok: int) -> int:
    return ((h - (out_tok + 1) * _P15) * _P + in_tok + 1) & MASK64


def _mix_int(h: int) -> int:
    h ^= h >> 31
    h = (h * _M1) & MASK64
    h ^= h >> 27
    return h


def _window_hashes(a: np.ndarray) -> np.ndarray:
    """Hashes of every WINDOW-long window of ``a`` (same polynomial as _hash_window_int)."""
    m = int(a.size) - WINDOW + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    x = (a.astype(np.int64) + 1).astype(np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    p = np.uint64(_P)
    with np.errstate(over="ignore"):
        for k in range(WINDOW):
            h = h * p + x[k:k + m]
    return h


def _mix_np(h: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        h = h ^ (h >> np.uint64(31))
        h = h * np.uint64(_M1)
        h = h ^ (h >> np.uint64(27))
    return h


def tier_of(match_len: int) -> int:
    if match_len >= 64:
        return 2
    if match_len >= 32:
        return 1
    return 0


def _segments(keep: np.ndarray, overlap: int) -> List[Tuple[int, int]]:
    """Runs of True in ``keep``, each widened by ``overlap`` ids into the duplicate runs around it."""
    n = int(keep.size)
    if n == 0:
        return []
    if keep.all():
        return [(0, n)]
    k = np.concatenate(([0], keep.astype(np.int8), [0]))
    d = np.diff(k)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(max(0, int(s) - overlap), min(n, int(e) + overlap)) for s, e in zip(starts, ends)]


def _write_private(path: str, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


# ----------------------------------------------------------------------------------------------- stores
class _Store:
    """Where the token ids live. The record logic above it is identical for every backend."""

    backend = "base"

    def __init__(self, cap_ids: int, chunk_ids: int) -> None:
        self.chunk_ids = int(chunk_ids)
        self.cap_ids = max(self.chunk_ids, int(cap_ids))
        self.base_pos = 0
        self.end_pos = 0
        self._ranges: List[Tuple[int, int]] = []  # (g0, g1) ranges that still have to be indexed

    # -- interface ---------------------------------------------------------------------------------
    def read(self, g0: int, g1: int) -> Optional[np.ndarray]:
        raise NotImplementedError

    def write(self, arr: np.ndarray) -> None:
        raise NotImplementedError

    def evict(self) -> int:
        raise NotImplementedError

    def clear(self) -> List[str]:
        """Drop every token; returns the list of files that could not be removed."""
        raise NotImplementedError

    def restore(self, ids: np.ndarray, base_pos: int) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def chunk_count(self) -> int:
        raise NotImplementedError

    def disk_bytes(self) -> int:
        return 0

    def mem_bytes(self) -> int:
        return 0

    def gpu_bytes(self) -> int:
        return 0

    def index_ranges(self) -> List[Tuple[int, int]]:
        return list(self._ranges)

    # -- helpers -----------------------------------------------------------------------------------
    def _drop_range(self, seq: int) -> None:
        C = self.chunk_ids
        self._ranges = [r for r in self._ranges if not (r[0] // C == seq)]


class _DiskStore(_Store):
    """Chunk files of int32 ids (little endian) in one directory, plus an identity and a lock file."""

    backend = "disk"

    def __init__(self, directory: str, identity: Optional[dict] = None, read_only: bool = False,
                 min_free_bytes: float = 0.0, cap_ids: int = CHUNK_IDS, chunk_ids: int = CHUNK_IDS) -> None:
        super().__init__(cap_ids, chunk_ids)
        self.dir = os.path.abspath(os.path.expanduser(str(directory)))
        self.identity = dict(identity or {})
        self.identity_on_disk: dict = {}
        self.read_only = bool(read_only)
        self.writer = False
        self.min_free_bytes = float(min_free_bytes)
        self.last_error = ""
        self._lock_fh = None
        self._fh = None
        self._fh_seq = -1
        self.load_evicted = 0
        self._chunks: Dict[int, np.ndarray] = {}
        self._fill: Dict[int, int] = {}
        self._open_dir()
        self._load()

    # -- setup -------------------------------------------------------------------------------------
    def _chunk_path(self, seq: int) -> str:
        return os.path.join(self.dir, "chunk-%08d.ids" % seq)

    def _open_dir(self) -> None:
        if not os.path.isdir(self.dir):
            if self.read_only:
                return
            os.makedirs(self.dir, mode=0o700, exist_ok=True)
        try:
            os.chmod(self.dir, 0o700)
        except OSError:
            pass
        ident_path = os.path.join(self.dir, "identity")
        if os.path.exists(ident_path):
            try:
                with open(ident_path, "r", encoding="utf-8") as f:
                    old = json.load(f)
            except (OSError, ValueError):
                old = {}
            od, nd = old.get("tokenizer_digest"), self.identity.get("tokenizer_digest")
            ov, nv = old.get("vocab_size"), self.identity.get("vocab_size")
            if (od and nd and od != nd) or (ov is not None and nv is not None and int(ov) != int(nv)):
                raise RecallIdentityError(
                    "recall record %s was written by tokenizer %s (vocab %s, model %s); this server's "
                    "tokenizer is %s (vocab %s)" % (self.dir, str(od)[:16], ov, old.get("model"),
                                                    str(nd)[:16], nv))
            self.identity_on_disk = old
        elif not self.read_only:
            data = dict(self.identity)
            data.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            _write_private(ident_path, json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")
            self.identity_on_disk = data
        if self.read_only:
            return
        lock_path = os.path.join(self.dir, "lock")
        fh = open(lock_path, "a+")
        try:
            os.chmod(lock_path, 0o600)
        except OSError:
            pass
        if fcntl is None:
            self._lock_fh = fh
            self.writer = True
            return
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            self.writer = False
            return
        self._lock_fh = fh
        self.writer = True

    def _load(self) -> None:
        if not os.path.isdir(self.dir):
            return
        seqs = sorted(int(m.group(1)) for m in (_CHUNK_RE.match(n) for n in os.listdir(self.dir)) if m)
        if not seqs:
            return
        run = [seqs[-1]]
        for s in reversed(seqs[:-1]):
            if s != run[0] - 1:
                break
            run.insert(0, s)
        if self.writer:
            for s in seqs:
                if s < run[0]:
                    try:
                        os.remove(self._chunk_path(s))
                    except OSError:
                        pass
        C = self.chunk_ids
        for s in run:
            path = self._chunk_path(s)
            with open(path, "rb") as f:
                data = f.read()
            usable = len(data) // 4 * 4
            if usable != len(data) and self.writer:
                os.truncate(path, usable)
            arr = np.frombuffer(data[:usable], dtype="<i4")[:C]
            buf = np.full(C, SENTINEL, dtype=np.int32)
            buf[:arr.size] = arr
            self._chunks[s] = buf
            self._fill[s] = int(arr.size)
        last = run[-1]
        self.base_pos = run[0] * C
        self.end_pos = last * C + self._fill[last]
        if self.writer:
            self.load_evicted = self.evict()
        for s in sorted(self._chunks):
            g0 = s * C
            self._ranges.append((g0, g0 + self._fill[s] if s == last else g0 + C))

    # -- low level ---------------------------------------------------------------------------------
    def read(self, g0: int, g1: int) -> Optional[np.ndarray]:
        if g0 < self.base_pos or g1 > self.end_pos or g0 > g1:
            return None
        C = self.chunk_ids
        seq = g0 // C
        o = g0 - seq * C
        if o + (g1 - g0) <= C:
            buf = self._chunks.get(seq)
            if buf is None:
                return None
            return buf[o:o + (g1 - g0)].astype(np.int64)
        out = np.empty(g1 - g0, dtype=np.int64)
        pos, w = g0, 0
        while pos < g1:
            seq = pos // C
            o = pos - seq * C
            buf = self._chunks.get(seq)
            if buf is None:
                return None
            take = min(C - o, g1 - pos)
            out[w:w + take] = buf[o:o + take]
            pos += take
            w += take
        return out

    def _open_file(self, seq: int) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        path = self._chunk_path(seq)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0), 0o600)
        self._fh = os.fdopen(fd, "ab")
        self._fh_seq = seq

    def write(self, arr: np.ndarray) -> None:
        a32 = np.ascontiguousarray(arr, dtype="<i4")
        C = self.chunk_ids
        off = 0
        while off < a32.size:
            seq = self.end_pos // C
            o = self.end_pos - seq * C
            if seq not in self._chunks:
                self._chunks[seq] = np.full(C, SENTINEL, dtype=np.int32)
                self._fill[seq] = 0
            if self._fh is None or self._fh_seq != seq:
                self._open_file(seq)
            take = min(C - o, int(a32.size) - off)
            piece = a32[off:off + take]
            self._chunks[seq][o:o + take] = piece
            self._fh.write(piece.tobytes())
            self._fill[seq] = o + take
            self.end_pos += take  # committed only after the ids are in memory
            off += take

    def evict(self) -> int:
        C = self.chunk_ids
        evicted = 0
        while self.end_pos - self.base_pos > self.cap_ids and len(self._chunks) > 1:
            oldest = min(self._chunks)
            self.base_pos = (oldest + 1) * C  # readers stop trusting it first
            self._chunks.pop(oldest, None)
            self._fill.pop(oldest, None)
            self._drop_range(oldest)
            try:
                os.remove(self._chunk_path(oldest))
            except FileNotFoundError:
                pass
            except OSError as e:
                self.last_error = "evict %s: %s" % (self._chunk_path(oldest), e)
            evicted += 1
        return evicted

    def clear(self) -> List[str]:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
            self._fh_seq = -1
        self.base_pos = 0
        self.end_pos = 0
        self._chunks.clear()
        self._fill.clear()
        self._ranges = []
        failed = []
        for name in os.listdir(self.dir):
            if _CHUNK_RE.match(name):
                try:
                    os.remove(os.path.join(self.dir, name))
                except OSError:
                    failed.append(os.path.join(self.dir, name))
        return failed

    def restore(self, ids: np.ndarray, base_pos: int) -> None:
        raise RecallBackendError("the disk backend is restored by opening its directory, not by a snapshot")

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        if self._lock_fh is not None:
            try:
                self._lock_fh.close()
            except OSError:
                pass
            self._lock_fh = None
        self.writer = False

    def chunk_count(self) -> int:
        return len(self._chunks)

    def disk_bytes(self) -> int:
        total = 0
        try:
            for name in os.listdir(self.dir):
                if _CHUNK_RE.match(name):
                    try:
                        total += os.stat(os.path.join(self.dir, name)).st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def mem_bytes(self) -> int:
        return len(self._chunks) * self.chunk_ids * 4


class _MemoryStore(_Store):
    """numpy int32 blocks in this process; nothing is written to disk."""

    backend = "memory"

    def __init__(self, cap_ids: int = CHUNK_IDS, chunk_ids: int = CHUNK_IDS) -> None:
        super().__init__(cap_ids, chunk_ids)
        self._chunks: Dict[int, np.ndarray] = {}
        self._fill: Dict[int, int] = {}

    def read(self, g0: int, g1: int) -> Optional[np.ndarray]:
        if g0 < self.base_pos or g1 > self.end_pos or g0 > g1:
            return None
        C = self.chunk_ids
        seq = g0 // C
        o = g0 - seq * C
        if o + (g1 - g0) <= C:
            buf = self._chunks.get(seq)
            if buf is None:
                return None
            return buf[o:o + (g1 - g0)].astype(np.int64)
        out = np.empty(g1 - g0, dtype=np.int64)
        pos, w = g0, 0
        while pos < g1:
            seq = pos // C
            o = pos - seq * C
            buf = self._chunks.get(seq)
            if buf is None:
                return None
            take = min(C - o, g1 - pos)
            out[w:w + take] = buf[o:o + take]
            pos += take
            w += take
        return out

    def write(self, arr: np.ndarray) -> None:
        a32 = np.ascontiguousarray(arr, dtype="<i4")
        C = self.chunk_ids
        off = 0
        while off < a32.size:
            seq = self.end_pos // C
            o = self.end_pos - seq * C
            if seq not in self._chunks:
                self._chunks[seq] = np.full(C, SENTINEL, dtype=np.int32)
                self._fill[seq] = 0
            take = min(C - o, int(a32.size) - off)
            self._chunks[seq][o:o + take] = a32[off:off + take]
            self._fill[seq] = o + take
            self.end_pos += take
            off += take

    def evict(self) -> int:
        C = self.chunk_ids
        evicted = 0
        while self.end_pos - self.base_pos > self.cap_ids and len(self._chunks) > 1:
            oldest = min(self._chunks)
            self.base_pos = (oldest + 1) * C
            self._chunks.pop(oldest, None)
            self._fill.pop(oldest, None)
            self._drop_range(oldest)
            evicted += 1
        return evicted

    def clear(self) -> List[str]:
        self.base_pos = 0
        self.end_pos = 0
        self._chunks.clear()
        self._fill.clear()
        self._ranges = []
        return []

    def restore(self, ids: np.ndarray, base_pos: int) -> None:
        C = self.chunk_ids
        arr = np.ascontiguousarray(ids, dtype="<i4")
        self.base_pos = int(base_pos)
        self.end_pos = int(base_pos) + int(arr.size)
        seq = self.base_pos // C
        o = self.base_pos - seq * C
        pos = 0
        while pos < arr.size:
            if seq not in self._chunks:
                self._chunks[seq] = np.full(C, SENTINEL, dtype=np.int32)
                self._fill[seq] = 0
            take = min(C - o, int(arr.size) - pos)
            self._chunks[seq][o:o + take] = arr[pos:pos + take]
            self._fill[seq] = o + take
            pos += take
            o += take
            if o >= C:
                o = 0
                seq += 1

    def chunk_count(self) -> int:
        return len(self._chunks)

    def mem_bytes(self) -> int:
        return len(self._chunks) * self.chunk_ids * 4


class _CudaStore(_Store):
    """torch int32 tensors on a GPU device; the 16-gram index stays on the host."""

    backend = "cuda"

    def __init__(self, cap_ids: int = CHUNK_IDS, chunk_ids: int = CHUNK_IDS, device: str = DEFAULT_DEVICE) -> None:
        super().__init__(cap_ids, chunk_ids)
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise RecallBackendError(
                "the cuda backend needs torch (pip install 'sglang-d-recall[torch]'): %r" % (e,))
        self._torch = torch
        if not torch.cuda.is_available():
            raise RecallBackendError("the cuda backend needs a CUDA device, but torch.cuda.is_available() is False")
        self.device = torch.device(device)
        self._chunks: Dict[int, "torch.Tensor"] = {}
        self._fill: Dict[int, int] = {}

    def read(self, g0: int, g1: int) -> Optional[np.ndarray]:
        if g0 < self.base_pos or g1 > self.end_pos or g0 > g1:
            return None
        C = self.chunk_ids
        out = np.empty(g1 - g0, dtype=np.int64)
        pos, w = g0, 0
        while pos < g1:
            seq = pos // C
            o = pos - seq * C
            buf = self._chunks.get(seq)
            if buf is None:
                return None
            take = min(C - o, g1 - pos)
            out[w:w + take] = buf[o:o + take].to("cpu").numpy().astype(np.int64)
            pos += take
            w += take
        return out

    def write(self, arr: np.ndarray) -> None:
        torch = self._torch
        a32 = np.ascontiguousarray(arr, dtype="<i4")
        C = self.chunk_ids
        off = 0
        while off < a32.size:
            seq = self.end_pos // C
            o = self.end_pos - seq * C
            if seq not in self._chunks:
                self._chunks[seq] = torch.full((C,), SENTINEL, dtype=torch.int32, device=self.device)
                self._fill[seq] = 0
            take = min(C - o, int(a32.size) - off)
            piece = torch.from_numpy(np.array(a32[off:off + take], dtype=np.int32))
            self._chunks[seq][o:o + take] = piece.to(self.device, non_blocking=False)
            self._fill[seq] = o + take
            self.end_pos += take
            off += take

    def evict(self) -> int:
        C = self.chunk_ids
        evicted = 0
        while self.end_pos - self.base_pos > self.cap_ids and len(self._chunks) > 1:
            oldest = min(self._chunks)
            self.base_pos = (oldest + 1) * C
            self._chunks.pop(oldest, None)
            self._fill.pop(oldest, None)
            self._drop_range(oldest)
            evicted += 1
        return evicted

    def clear(self) -> List[str]:
        self.base_pos = 0
        self.end_pos = 0
        self._chunks.clear()
        self._fill.clear()
        self._ranges = []
        return []

    def restore(self, ids: np.ndarray, base_pos: int) -> None:
        torch = self._torch
        C = self.chunk_ids
        arr = np.ascontiguousarray(ids, dtype="<i4")
        self.base_pos = int(base_pos)
        self.end_pos = int(base_pos) + int(arr.size)
        seq = self.base_pos // C
        o = self.base_pos - seq * C
        pos = 0
        while pos < arr.size:
            if seq not in self._chunks:
                self._chunks[seq] = torch.full((C,), SENTINEL, dtype=torch.int32, device=self.device)
                self._fill[seq] = 0
            take = min(C - o, int(arr.size) - pos)
            piece = torch.from_numpy(np.array(arr[pos:pos + take], dtype=np.int32))
            self._chunks[seq][o:o + take] = piece.to(self.device, non_blocking=False)
            self._fill[seq] = o + take
            pos += take
            o += take
            if o >= C:
                o = 0
                seq += 1

    def chunk_count(self) -> int:
        return len(self._chunks)

    def mem_bytes(self) -> int:
        return 0  # the ids live in device memory, not in host memory

    def gpu_bytes(self) -> int:
        return len(self._chunks) * self.chunk_ids * 4

    def close(self) -> None:
        self._chunks.clear()
        self._fill.clear()


# ----------------------------------------------------------------------------------------------- record
class RecallRecord:
    """Append-only record of token ids plus a lossy in-memory 16-gram index (newest wins).

    ``backend`` selects where the ids live: ``disk`` (chunk files), ``memory`` (numpy blocks in this
    process) or ``cuda`` (torch tensors on a GPU device). Every backend answers add/lookup/continuation/
    clear/stats the same way.
    """

    def __init__(self, directory: Optional[str] = None, identity: Optional[dict] = None, gib: float = 1.0 / 16,
                 read_only: bool = False, min_free_gib: float = 4.0, chunk_ids: int = CHUNK_IDS,
                 backend: str = "disk", device: Optional[str] = None, evict: Optional[str] = None,
                 hot_frac: Optional[float] = None, dedup_run: Optional[int] = None,
                 heat_halflife_ids: Optional[float] = None, heat_seen_weight: Optional[float] = None,
                 promote_tokens: Optional[float] = None, index_density: Optional[float] = None, environ=None):
        backend = str(backend or "disk").strip().lower()
        if backend in ("gpu", "cuda"):
            backend = "cuda"
        if backend not in BACKENDS:
            raise ValueError("unknown recall backend %r (use one of %s)" % (backend, ", ".join(BACKENDS)))
        self.backend = backend
        self.chunk_ids = int(chunk_ids)
        if self.chunk_ids < 4 * WINDOW:
            raise ValueError("chunk_ids too small")
        self.gib = float(gib)
        self.cap_ids = max(self.chunk_ids, int(self.gib * (1 << 30)) // 4)
        env = os.environ if environ is None else environ

        def _cfg(val, key, default):
            return val if val is not None else _env(ENV_NEW[key], ENV_OLD[key], str(default), env)

        mode = str(_cfg(evict, "evict", "heat")).strip().lower()
        if mode not in EVICT_MODES:
            raise ValueError("unknown recall evict mode %r (use one of %s)" % (mode, ", ".join(EVICT_MODES)))
        frac = min(0.95, max(0.0, float(_cfg(hot_frac, "hot_frac", 0.5)))) if mode == "heat" else 0.0
        if frac <= 0.0:
            mode = "fifo"  # no hot zone: the store behaves exactly like fifo
        self.evict_mode = mode
        dflt_run = DEDUP_RUN_HEAT if mode == "heat" else DEDUP_RUN
        self.dedup_run = max(2 * WINDOW, int(_cfg(dedup_run, "dedup_run", dflt_run)))
        self.hot_cap_ids = int(self.cap_ids * frac) if mode == "heat" else 0
        self.cold_cap_ids = max(self.chunk_ids, self.cap_ids - self.hot_cap_ids)
        self.heat_halflife = max(1.0, float(_cfg(heat_halflife_ids, "heat_halflife", max(1, self.cold_cap_ids // 4))))
        self.heat_seen_weight = max(0.0, float(_cfg(heat_seen_weight, "heat_seen_weight", 0.25)))
        self.promote_tokens = max(0.0, float(_cfg(promote_tokens, "promote_tokens", 32)))
        # index density: hash slots per stored id (0.25 = the 1.1.0 table, one slot per 4 ids)
        self.index_density = min(MAX_INDEX_DENSITY, max(MIN_INDEX_DENSITY,
                                                        float(_cfg(index_density, "index_density",
                                                                   DEFAULT_INDEX_DENSITY))))
        self.min_free_bytes = float(min_free_gib) * (1 << 30)
        self.identity = dict(identity or {})
        self.identity_on_disk: dict = {}
        self.read_only = bool(read_only)
        self.writer = True
        self.add_blocked = False
        self.skipped_low_disk = 0
        self.added_ids = 0
        self.deduped_ids = 0
        self.adds = 0
        self.evicted_chunks = 0
        self.last_error = ""
        self._clear_requested = False
        want = max(2, int((self.cold_cap_ids if self.evict_mode == "heat" else self.cap_ids) * self.index_density))
        # ~1 slot per 4 ids by default: older windows survive later overwrites
        self._slots = 1 << max(16, (want - 1).bit_length())
        self._mask = self._slots - 1
        self._keys = np.zeros(self._slots, dtype=np.uint64)
        self._pos = np.full(self._slots, -1, dtype=np.int64)
        self._init_heat()
        self._wlock = threading.RLock()
        self.store = self._make_store(directory, device)
        self.dir = getattr(self.store, "dir", None)
        if self.dir is None and directory:
            # memory / cuda keep no ids on disk; the directory only holds control files
            # (paused, clear.request, stats.json) so the same operating scripts work for every backend
            self.dir = os.path.abspath(os.path.expanduser(str(directory)))
            try:
                os.makedirs(self.dir, mode=0o700, exist_ok=True)
            except OSError:
                self.dir = None
        self.writer = bool(getattr(self.store, "writer", True))
        self.identity_on_disk = dict(getattr(self.store, "identity_on_disk", {}))
        self.evicted_chunks += int(getattr(self.store, "load_evicted", 0))
        for g0, g1 in self.store.index_ranges():
            self._index_range(g0, g1)

    # -- setup -------------------------------------------------------------------------------------
    def _make_store(self, directory: Optional[str], device: Optional[str]) -> _Store:
        if self.backend == "disk":
            if directory is None:
                raise ValueError("the disk backend needs a directory")
            return _DiskStore(directory, self.identity, read_only=self.read_only,
                              min_free_bytes=self.min_free_bytes, cap_ids=self.cold_cap_ids,
                              chunk_ids=self.chunk_ids)
        if self.backend == "memory":
            return _MemoryStore(cap_ids=self.cold_cap_ids, chunk_ids=self.chunk_ids)
        return _CudaStore(cap_ids=self.cold_cap_ids, chunk_ids=self.chunk_ids, device=device or DEFAULT_DEVICE)

    @property
    def state(self) -> str:
        if self.add_blocked:
            return "blocked"
        return "writing" if self.writer else "read-only"

    @property
    def base_pos(self) -> int:
        return int(self.store.base_pos)

    @property
    def end_pos(self) -> int:
        return int(self.store.end_pos)

    def _read(self, g0: int, g1: int) -> Optional[np.ndarray]:
        return self.store.read(g0, g1)

    def _write_ids(self, arr: np.ndarray) -> None:
        self.store.write(arr)

    # -- heat / hot zone ---------------------------------------------------------------------------
    def _init_heat(self) -> None:
        """(Re)build the heat table and the hot zone; fifo mode keeps neither."""
        self.promotions = self.hot_evictions = self.hot_hits = self.cold_hits = 0
        self.hot_ids = 0
        self._clock = 0
        self._heat_on = self.evict_mode == "heat"
        self._hot_n = 0
        self._hot_slots = 0
        if not self._heat_on:
            return
        nseg = (self.cold_cap_ids + self.chunk_ids) // HEAT_SEG + 2  # live cold segments never exceed this
        self._nseg = 1 << max(4, (nseg - 1).bit_length())
        self._seg_tag = np.full(self._nseg, -1, dtype=np.int64)
        self._seg_L = np.full(self._nseg, -np.inf, dtype=np.float64)
        self._seg_hot = np.full(self._nseg, -1, dtype=np.int64)
        n = int(self.hot_cap_ids // HOT_SLOT)
        self._hot_n = n
        self._hot_buf = np.zeros((n, HOT_SLOT), dtype=np.int32)
        self._hot_len = np.zeros(n, dtype=np.int32)
        self._hot_L = np.full(n, -np.inf, dtype=np.float64)
        self._hot_gen = np.zeros(n, dtype=np.int64)
        self._hot_seg = np.full(n, -1, dtype=np.int64)
        self._hot_free: List[int] = list(range(n - 1, -1, -1))
        self._hot_heap: List[Tuple[float, int]] = []
        want = max(2, int(self.hot_cap_ids * self.index_density))  # like the cold index: 1 slot per 4 ids by default
        hs = 1 << max(12, (want - 1).bit_length())
        self._hmask = hs - 1
        self._hot_slots = hs
        # the hot index keeps 32 key bits and (slot << HOT_OFF_BITS | offset); every hit is checked against the
        # entry's ids, so a stale or colliding slot can only miss, never return a wrong position
        self._hkeys = np.zeros(hs, dtype=np.uint32)
        self._hpos = np.full(hs, -1, dtype=np.int32 if n < (1 << (31 - HOT_OFF_BITS)) else np.int64)

    def _heat_bytes(self) -> int:
        if not self._heat_on:
            return 0
        arrs = (self._seg_tag, self._seg_L, self._seg_hot, self._hot_buf, self._hot_len, self._hot_L,
                self._hot_gen, self._hot_seg, self._hkeys, self._hpos)
        return int(sum(x.nbytes for x in arrs))

    def _now(self) -> float:
        return self._clock / self.heat_halflife

    @staticmethod
    def _decayed(L: float, now: float) -> float:
        x = L - now
        return 2.0 ** x if x > -1000.0 else 0.0

    def _bump(self, L: float, c: float) -> float:
        """Log-domain heat: L = log2(score) + clock/halflife, so ordering by L is ordering by decayed score."""
        now = self._now()
        s = self._decayed(L, now) + c
        return math.log2(s) + now if s > 0.0 else -math.inf

    def _seg_index(self, seg: int) -> int:
        i = seg & (self._nseg - 1)
        if int(self._seg_tag[i]) != seg:
            self._seg_tag[i] = seg
            self._seg_L[i] = -np.inf
            self._seg_hot[i] = -1
        return i

    def _hot_uid(self, slot: int) -> int:
        return int(self._hot_gen[slot]) * self._hot_n + int(slot)

    def _hot_slot_of_uid(self, uid: int) -> Optional[int]:
        if self._hot_n == 0 or uid < 0:
            return None
        slot, gen = uid % self._hot_n, uid // self._hot_n
        if int(self._hot_gen[slot]) != gen or int(self._hot_len[slot]) == 0:
            return None
        return slot

    def _hot_decode(self, pos: int) -> Optional[Tuple[int, int]]:
        rel = int(pos) - HOT_FLAG
        if rel < 0:
            return None
        slot = self._hot_slot_of_uid(rel >> HOT_OFF_BITS)
        if slot is None:
            return None
        return slot, rel & ((1 << HOT_OFF_BITS) - 1)

    def heat_of(self, pos: int) -> float:
        """Current (decayed) heat of the segment / hot entry that ``pos`` belongs to; 0.0 when untracked."""
        if not self._heat_on:
            return 0.0
        pos = int(pos)
        now = self._now()
        if pos >= HOT_FLAG:
            r = self._hot_decode(pos)
            return 0.0 if r is None else self._decayed(float(self._hot_L[r[0]]), now)
        seg = pos // HEAT_SEG
        i = seg & (self._nseg - 1)
        if int(self._seg_tag[i]) != seg:
            return 0.0
        slot = self._hot_slot_of_uid(int(self._seg_hot[i]))
        if slot is not None:
            return self._decayed(float(self._hot_L[slot]), now)
        return self._decayed(float(self._seg_L[i]), now)

    def credit(self, pos: int, tokens: float) -> None:
        """Add ``tokens`` of heat to the source of a copy at ``pos`` (as returned by :meth:`lookup`)."""
        if not self._heat_on or tokens <= 0:
            return
        with self._wlock:
            self._credit(int(pos), float(tokens))

    def _credit(self, pos: int, c: float) -> None:
        if pos >= HOT_FLAG:
            r = self._hot_decode(pos)
            if r is not None:
                self._hot_credit(r[0], c)
            return
        if pos < self.base_pos or pos >= self.end_pos:
            return
        seg = pos // HEAT_SEG
        i = self._seg_index(seg)
        slot = self._hot_slot_of_uid(int(self._seg_hot[i]))
        if slot is not None:  # already promoted: the heat keeps accruing on the hot copy
            self._hot_credit(slot, c)
            return
        self._seg_hot[i] = -1
        L = self._bump(float(self._seg_L[i]), c)
        self._seg_L[i] = L
        if self._hot_n and self._decayed(L, self._now()) >= self.promote_tokens:
            self._promote(seg, i)

    def _credit_range(self, r0: int, n: int, w: float) -> None:
        r1 = r0 + n
        seg = r0 // HEAT_SEG
        while seg * HEAT_SEG < r1:
            a, b = max(r0, seg * HEAT_SEG), min(r1, (seg + 1) * HEAT_SEG)
            if b > a:
                self._credit(a, (b - a) * w)
            seg += 1

    def _hot_credit(self, slot: int, c: float) -> None:
        L = self._bump(float(self._hot_L[slot]), c)
        self._hot_L[slot] = L
        heapq.heappush(self._hot_heap, (L, self._hot_uid(slot)))
        live = self._hot_n - len(self._hot_free)
        if len(self._hot_heap) > 4 * live + 1024:  # drop stale heap entries (amortised O(1) per credit)
            self._hot_heap = [(float(self._hot_L[s]), self._hot_uid(s))
                              for s in np.flatnonzero(self._hot_len > 0).tolist()]
            heapq.heapify(self._hot_heap)

    def _hot_drop(self, slot: int) -> None:
        uid = self._hot_uid(slot)
        seg = int(self._hot_seg[slot])
        if seg >= 0:
            j = seg & (self._nseg - 1)
            if int(self._seg_tag[j]) == seg and int(self._seg_hot[j]) == uid:
                self._seg_hot[j] = -1
                self._seg_L[j] = -np.inf  # a demoted segment needs fresh evidence to come back
        self.hot_ids -= int(self._hot_len[slot])
        self._hot_len[slot] = 0
        self._hot_L[slot] = -np.inf
        self._hot_seg[slot] = -1
        self._hot_gen[slot] += 1  # every position / index slot that named the old entry is now stale
        self._hot_free.append(int(slot))

    def _hot_alloc(self) -> Optional[int]:
        if not self._hot_free:
            heap = self._hot_heap
            while heap:
                L, uid = heapq.heappop(heap)
                slot = self._hot_slot_of_uid(uid)
                if slot is None or float(self._hot_L[slot]) != L:
                    continue  # stale heap entry
                self._hot_drop(slot)
                self.hot_evictions += 1
                break
        if not self._hot_free:
            return None
        return self._hot_free.pop()

    def _hot_put(self, ids: np.ndarray, L: float, seg: int) -> Optional[int]:
        slot = self._hot_alloc()
        if slot is None:
            return None
        k = int(ids.size)
        self._hot_buf[slot, :k] = ids
        self._hot_len[slot] = k
        self._hot_L[slot] = L
        self._hot_seg[slot] = seg
        self.hot_ids += k
        heapq.heappush(self._hot_heap, (L, self._hot_uid(slot)))
        self._hot_index(slot)
        return slot

    def _promote(self, seg: int, i: int) -> None:
        a = max(self.base_pos, seg * HEAT_SEG - HOT_PRE)
        b = min(self.end_pos, (seg + 1) * HEAT_SEG + HOT_POST)
        if b - a < WINDOW + MIN_TAIL:
            return
        ids = self._read(a, b)
        if ids is None:
            return
        slot = self._hot_put(ids, float(self._seg_L[i]), seg)
        if slot is None:
            return
        self._seg_hot[i] = self._hot_uid(slot)
        self.promotions += 1

    def _hot_index(self, slot: int) -> None:
        k = int(self._hot_len[slot])
        span = WINDOW + MIN_TAIL
        m = k - span + 1
        if m <= 0:
            return
        e = self._hot_buf[slot, :k].astype(np.int64)
        h = _window_hashes(e[:m + WINDOW - 1])
        cs = np.concatenate(([0], np.cumsum(e < 0, dtype=np.int64)))
        idx = np.flatnonzero((cs[span:span + m] - cs[:m]) == 0)
        if idx.size == 0:
            return
        h = h[idx]
        pos = (np.int64(slot) << HOT_OFF_BITS) + idx.astype(np.int64) + WINDOW
        s = (_mix_np(h) & np.uint64(self._hmask)).astype(np.int64)
        self._hkeys[s] = (h & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        self._hpos[s] = pos

    def _hot_lookup(self, t: np.ndarray, h: int, max_back: int) -> Optional[Tuple[int, int]]:
        s = _mix_int(h) & self._hmask
        if int(self._hkeys[s]) != (h & 0xFFFFFFFF):
            return None
        q = int(self._hpos[s])
        if q < 0:
            return None
        slot, off = q >> HOT_OFF_BITS, q & ((1 << HOT_OFF_BITS) - 1)
        k = int(self._hot_len[slot])
        if k == 0 or off < WINDOW or off > k:
            return None
        lim = max(WINDOW, min(int(max_back), int(t.size), off))
        back = self._hot_buf[slot, off - lim:off]
        mm = np.flatnonzero(back[::-1] != t[t.size - lim:][::-1])  # window check and back-extension at once
        L = int(mm[0]) if mm.size else lim
        if L < WINDOW:
            return None
        return HOT_FLAG + (self._hot_uid(slot) << HOT_OFF_BITS) + off, L

    # -- index -------------------------------------------------------------------------------------
    def _index_range(self, g0: int, g1: int) -> None:
        """Index every window that ends inside [g0, g1) and is followed by >= MIN_TAIL ids of its segment.

        A window with (almost) nothing after it - the end of a prompt - is left out, so it cannot shadow an
        older occurrence of the same ids that has a continuation (e.g. the earlier reply to the same prompt)."""
        s0 = max(self.base_pos, g0 - (WINDOW - 1))
        g_ext = min(self.end_pos, g1 + MIN_TAIL)
        if g_ext - s0 < WINDOW + MIN_TAIL:
            return
        a = self._read(s0, g_ext)
        if a is None:
            return
        span = WINDOW + MIN_TAIL
        m = min(int(a.size) - span + 1, g1 - WINDOW - s0 + 1)  # windows that end inside [g0, g1)
        if m <= 0:
            return
        h = _window_hashes(a[:m + WINDOW - 1])
        bad = a < 0
        cs = np.concatenate(([0], np.cumsum(bad, dtype=np.int64)))
        idx = np.flatnonzero((cs[span:span + m] - cs[:m]) == 0)
        if idx.size == 0:
            return
        h = h[idx]
        pos = s0 + idx.astype(np.int64) + WINDOW
        slots = (_mix_np(h) & np.uint64(self._mask)).astype(np.int64)
        self._keys[slots] = h
        self._pos[slots] = pos

    def _purge_index(self) -> None:
        """Drop the index slots whose positions fell out of the record (evicted ids)."""
        stale = self._pos < (self.base_pos + WINDOW)
        if not stale.any():
            return
        self._pos[stale] = -1
        self._keys[stale] = 0

    def _evict(self) -> None:
        n = int(self.store.evict())
        self.evicted_chunks += n
        if n:  # the full-index scan costs O(slots); only stale slots after a real eviction need it
            self._purge_index()

    def _extend_run(self, a: np.ndarray, s: int, e: int, p_s: int, p_e: int) -> Tuple[int, int]:
        """Token range [ts, te) of window run s..e, widened by direct comparison with the record (bridges
        index slots that a later window overwrote)."""
        ts, rs = s, p_s - WINDOW
        k = min(256, ts, rs - self.base_pos)
        if k > 0:
            rec = self._read(rs - k, rs)
            if rec is not None:
                mm = np.flatnonzero(rec[::-1] != a[ts - k:ts][::-1])
                ts -= int(mm[0]) if mm.size else k
        te, re_ = e + WINDOW, p_e
        k = min(256, int(a.size) - te, self.end_pos - re_)
        if k > 0:
            rec = self._read(re_, re_ + k)
            if rec is not None:
                mm = np.flatnonzero(rec != a[te:te + k])
                te += int(mm[0]) if mm.size else k
        return ts, te

    def _novel_mask(self, a: np.ndarray, seen: Optional[list] = None) -> np.ndarray:
        """False for ids inside runs of >= dedup_run ids the record already holds (or the whole input when
        the record already holds all of it). ``seen`` collects (record position, length) of skipped runs."""
        keep = np.ones(a.size, dtype=bool)
        if self.end_pos - self.base_pos < WINDOW:
            return keep
        h = _window_hashes(a)
        slots = (_mix_np(h) & np.uint64(self._mask)).astype(np.int64)
        k = self._keys[slots]
        p = self._pos[slots]
        valid = (k == h) & (p - WINDOW >= self.base_pos) & (p <= self.end_pos)
        if not valid.any():
            return keep
        m = valid.size
        chain = np.zeros(m, dtype=bool)
        chain[1:] = valid[1:] & valid[:-1] & (p[1:] == p[:-1] + 1)
        nxt = np.zeros(m, dtype=bool)
        nxt[:-1] = chain[1:]
        starts = np.flatnonzero(valid & ~chain)
        ends = np.flatnonzero(valid & ~nxt)
        cover = (ends - starts + 1) + (WINDOW - 1)
        sel = cover >= 2 * WINDOW
        if a.size < 2 * WINDOW:
            sel = cover >= WINDOW
        done = 0  # end of the input range already reported to ``seen`` (runs widen into each other)
        for s, e in zip(starts[sel], ends[sel]):
            ts, te = self._extend_run(a, int(s), int(e), int(p[s]), int(p[e]))
            if te - ts >= self.dedup_run or (ts == 0 and te == a.size):
                keep[ts:te] = False
                if seen is not None and te > done:
                    u = max(ts, done)
                    seen.append((int(p[s]) - WINDOW - (int(s) - u), te - u))
                    done = te
        return keep

    def _append_segment(self, seg: np.ndarray) -> int:
        if self.end_pos > self.base_pos:
            last = self._read(self.end_pos - 1, self.end_pos)
            if last is not None and int(last[0]) != SENTINEL:
                self._write_ids(np.array([SENTINEL], dtype=np.int64))
        start = self.end_pos
        self._write_ids(seg)
        self._index_range(start, self.end_pos)
        return int(seg.size)

    # -- public ------------------------------------------------------------------------------------
    def add(self, ids) -> int:
        """Append the novel part of ``ids``; returns how many ids were appended."""
        if not self.writer or self.add_blocked:
            return 0
        a = np.asarray(ids, dtype=np.int64).ravel()
        if a.size and int(a.min()) < 0:
            a = a[a >= 0]
        if a.size < WINDOW:
            return 0
        if self.backend == "disk":
            try:
                if shutil.disk_usage(self.dir).free < self.min_free_bytes:
                    self.skipped_low_disk += 1
                    return 0
            except OSError:
                pass
        with self._wlock:
            seen: Optional[list] = None
            if self._heat_on:
                self._clock += int(a.size)
                seen = [] if self.heat_seen_weight > 0 else None
            keep = self._novel_mask(a, seen)
            if seen:
                for r0, n in seen:  # the record already held this content: it came back, so it is warmer
                    self._credit_range(r0, n, self.heat_seen_weight)
            added = 0
            for i, j in _segments(keep, OVERLAP):
                if j - i >= WINDOW:
                    added += self._append_segment(a[i:j])
            self.deduped_ids += int(a.size) - int(np.count_nonzero(keep))
            if added:
                self.store.flush()
                self._evict()
            self.added_ids += added
            self.adds += 1
            return added

    def lookup(self, tail, max_back: int = MAX_BACK, h: Optional[int] = None) -> Optional[Tuple[int, int]]:
        """(continuation position, match length) for the newest occurrence of tail's last 16 ids.

        In heat mode the hot zone is asked too and wins when its match is at least as long; its positions are
        >= HOT_FLAG and work with :meth:`continuation` and :meth:`credit` like cold ones."""
        t = np.asarray(tail, dtype=np.int64)
        if t.size < WINDOW:
            return None
        if h is None:
            h = _hash_window_int(t[-WINDOW:].tolist())
        rh = self._hot_lookup(t, h, max_back) if (self._hot_n and self.hot_ids) else None
        if rh is not None and rh[1] >= min(int(max_back), int(t.size)):
            self.hot_hits += 1  # already the longest match possible: the cold ring cannot beat it
            return rh
        rc = self._cold_lookup(t, max_back, h)
        if rh is not None and (rc is None or rh[1] >= rc[1]):
            self.hot_hits += 1
            return rh
        if rc is not None and self._heat_on:
            self.cold_hits += 1
        return rc

    def _cold_lookup(self, t: np.ndarray, max_back: int, h: int) -> Optional[Tuple[int, int]]:
        if self.end_pos - self.base_pos < WINDOW:
            return None
        slot = _mix_int(h) & self._mask
        if int(self._keys[slot]) != h:
            return None
        p = int(self._pos[slot])
        if p - WINDOW < self.base_pos or p > self.end_pos:
            return None
        rec = self._read(p - WINDOW, p)
        if rec is None or not np.array_equal(rec, t[-WINDOW:]):
            return None
        L = WINDOW
        lim = min(int(max_back), int(t.size))
        if lim > WINDOW:
            g0 = max(self.base_pos, p - lim)
            back = self._read(g0, p - WINDOW)
            if back is not None and back.size:
                tb = t[t.size - WINDOW - back.size:t.size - WINDOW]
                mm = np.flatnonzero(back[::-1] != tb[::-1])
                L += int(mm[0]) if mm.size else int(back.size)
        return p, L

    def continuation(self, pos: int, n: int, min_n: Optional[int] = None) -> Optional[np.ndarray]:
        """Up to ``n`` ids that followed position ``pos`` (at least ``min_n``, default n), stopping at a sentinel."""
        if min_n is None:
            min_n = n
        if pos >= HOT_FLAG:
            r = self._hot_decode(pos) if self._hot_n else None
            if r is None:
                return None
            slot, off = r
            seg = self._hot_buf[slot, off:min(off + n, int(self._hot_len[slot]))].astype(np.int64)
            if r != self._hot_decode(pos):  # the entry was replaced while we copied
                return None
        else:
            end = min(pos + n, self.end_pos)
            if end - pos < min_n:
                return None
            seg = self._read(pos, end)
        if seg is None:
            return None
        neg = np.flatnonzero(seg < 0)
        if neg.size:
            seg = seg[:int(neg[0])]
        if seg.size < min_n:
            return None
        return seg

    def clear(self) -> None:
        if not self.writer:
            raise PermissionError("recall record %s is read-only here" % (self.dir or self.backend))
        with self._wlock:
            failed = self.store.clear()
            self._keys[:] = 0
            self._pos[:] = -1
            self._init_heat()
            self.added_ids = self.deduped_ids = self.adds = self.evicted_chunks = 0
            self.skipped_low_disk = 0
            if failed:
                self.add_blocked = True
                self.last_error = "could not delete %s" % ", ".join(failed)
                raise OSError(self.last_error)
            self.add_blocked = False
            self.last_error = ""

    def request_clear(self) -> None:
        """Ask the record to be cleared at the next tick (in-process request; the disk backend also
        honours a ``clear.request`` file in its directory)."""
        self._clear_requested = True

    def poll_clear_request(self) -> bool:
        """True when a clear was requested; the disk backend also watches its ``clear.request`` file."""
        req = self._clear_requested
        self._clear_requested = False
        if self.dir:
            ctl = os.path.join(self.dir, "clear.request")
            if os.path.exists(ctl):
                try:
                    os.remove(ctl)
                except OSError:
                    pass
                req = True
        return req

    def disk_bytes(self) -> int:
        return int(self.store.disk_bytes())

    def stats(self) -> dict:
        return {
            "dir": self.dir if self.dir is not None else ("memory" if self.backend == "memory" else "cuda"),
            "backend": self.backend,
            "state": self.state,
            "writer": self.writer,
            "ids": int(self.end_pos - self.base_pos),
            "chunks": int(self.store.chunk_count()),
            "bytes_disk": self.disk_bytes(),
            "bytes_mem": int(self.store.mem_bytes()) + self._keys.nbytes + self._pos.nbytes + self._heat_bytes(),
            "bytes_gpu": int(self.store.gpu_bytes()),
            "cap_ids": int(self.cap_ids),
            "added_ids": int(self.added_ids),
            "deduped_ids": int(self.deduped_ids),
            "adds": int(self.adds),
            "evicted_chunks": int(self.evicted_chunks),
            "skipped_low_disk": int(self.skipped_low_disk),
            "add_blocked": bool(self.add_blocked),
            "last_error": self.last_error,
            "evict_mode": self.evict_mode,
            "dedup_run": int(self.dedup_run),
            "hot_ids": int(self.hot_ids),
            "hot_cap_ids": int(self.hot_cap_ids),
            "hot_segments": int(self._hot_n - len(self._hot_free)) if self._hot_n else 0,
            "promotions": int(self.promotions),
            "hot_evictions": int(self.hot_evictions),
            "hot_hits": int(self.hot_hits),
            "cold_hits": int(self.cold_hits),
            "index_density": float(self.index_density),
            "index_slots": int(self._slots),
            "hot_index_slots": int(self._hot_slots),
        }

    def close(self) -> None:
        with self._wlock:
            self.store.close()

    # -- snapshots (memory / cuda backends) -------------------------------------------------------
    def save(self, path: str) -> str:
        """Write a npz snapshot (token ids + identity) of the record to ``path``."""
        ids = self._export_ids()
        meta = {
            "identity": json.dumps(self.identity, ensure_ascii=False, sort_keys=True),
            "base_pos": int(self.base_pos),
            "end_pos": int(self.end_pos),
            "chunk_ids": int(self.chunk_ids),
            "backend": self.backend,
            "index_density": float(self.index_density),
            "format": 1,
        }
        path = str(path)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        extra = self._export_heat()
        if extra:
            meta["heat"] = {"clock": int(self._clock), "halflife": float(self.heat_halflife)}
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:  # a file object keeps numpy from appending ".npz" to the name
            np.savez(f, ids=ids.astype(np.int32), meta=np.array(json.dumps(meta, ensure_ascii=False)), **extra)
        os.replace(tmp, path)
        return path

    def _export_heat(self) -> dict:
        """Hot entries and the live heat table, for :meth:`save` (empty in fifo mode)."""
        if not self._heat_on:
            return {}
        with self._wlock:
            live = np.flatnonzero(self._hot_len > 0)
            lens = self._hot_len[live].astype(np.int32)
            ids = (np.concatenate([self._hot_buf[s, :int(self._hot_len[s])] for s in live.tolist()])
                   if live.size else np.zeros(0, dtype=np.int32))
            segs = np.flatnonzero((self._seg_tag >= 0) & np.isfinite(self._seg_L))
            return {"hot_ids": ids.astype(np.int32), "hot_lens": lens, "hot_L": self._hot_L[live].copy(),
                    "hot_seg": self._hot_seg[live].copy(), "seg_tag": self._seg_tag[segs].copy(),
                    "seg_L": self._seg_L[segs].copy()}

    def _import_heat(self, z, clock: int) -> None:
        """Re-insert saved hot entries (hottest first, as many as fit) and the heat of still-live segments."""
        if not self._heat_on or "hot_lens" not in z:
            return
        self._clock = int(clock)
        lo, hi = self.base_pos // HEAT_SEG, (self.end_pos - 1) // HEAT_SEG
        for tag, L in zip(np.array(z["seg_tag"]).tolist(), np.array(z["seg_L"]).tolist()):
            if lo <= tag <= hi:
                self._seg_L[self._seg_index(int(tag))] = L
        lens = np.array(z["hot_lens"]).astype(np.int64)
        offs = np.concatenate(([0], np.cumsum(lens)))
        ids = np.array(z["hot_ids"])
        Ls = np.array(z["hot_L"])
        segs = np.array(z["hot_seg"])
        for k in np.argsort(-Ls, kind="stable").tolist()[:self._hot_n]:
            slot = self._hot_put(ids[offs[k]:offs[k + 1]], float(Ls[k]), int(segs[k]))
            if slot is None:
                break
            seg = int(segs[k])
            if lo <= seg <= hi:
                self._seg_hot[self._seg_index(seg)] = self._hot_uid(slot)

    def _export_ids(self) -> np.ndarray:
        seg = self._read(self.base_pos, self.end_pos)
        if seg is None:
            return np.zeros(0, dtype=np.int32)
        return seg.astype(np.int32)

    @classmethod
    def load(cls, path: str, identity: Optional[dict] = None, backend: str = "memory",
             device: Optional[str] = None, chunk_ids: Optional[int] = None,
             gib: Optional[float] = None, evict: Optional[str] = None, hot_frac: Optional[float] = None,
             dedup_run: Optional[int] = None, index_density: Optional[float] = None) -> "RecallRecord":
        """Restore a record from a npz snapshot written by :meth:`save` (heat mode also restores the hot zone)."""
        backend = str(backend or "memory").strip().lower()
        if backend in ("gpu", "cuda"):
            backend = "cuda"
        if backend not in BACKENDS:
            raise ValueError("unknown recall backend %r" % (backend,))
        with np.load(str(path), allow_pickle=False) as z:
            ids = np.array(z["ids"])
            meta = json.loads(str(z["meta"]))
            heat = {k: np.array(z[k]) for k in ("hot_ids", "hot_lens", "hot_L", "hot_seg", "seg_tag", "seg_L")
                    if k in z.files}
        saved = json.loads(meta.get("identity") or "{}")
        ident = dict(identity or saved)
        sd, nd = saved.get("tokenizer_digest"), ident.get("tokenizer_digest")
        sv, nv = saved.get("vocab_size"), ident.get("vocab_size")
        if (sd and nd and sd != nd) or (sv is not None and nv is not None and int(sv) != int(nv)):
            raise RecallIdentityError(
                "recall snapshot %s was written by tokenizer %s (vocab %s); this server's tokenizer is %s "
                "(vocab %s)" % (path, str(sd)[:16], sv, str(nd)[:16], nv))
        cid = int(chunk_ids or meta.get("chunk_ids") or CHUNK_IDS)
        dens = index_density if index_density is not None else meta.get("index_density")
        rec = cls(None, ident, gib=float(gib if gib is not None else 1.0 / 16), chunk_ids=cid,
                  backend=backend, device=device, evict=evict, hot_frac=hot_frac, dedup_run=dedup_run,
                  index_density=None if dens is None else float(dens))
        rec.store.restore(ids, int(meta.get("base_pos", 0)))
        rec._index_range(rec.base_pos, rec.end_pos)
        if heat and rec.end_pos > rec.base_pos:
            with rec._wlock:
                rec._import_heat(heat, int((meta.get("heat") or {}).get("clock", 0)))
        return rec


# ----------------------------------------------------------------------------------------------- engine
@dataclass
class Proposal:
    tokens: np.ndarray  # exactly n ids (padded with the last real id when the copy is shorter)
    source: str  # "own" | "record"
    tier: int  # 0: 16-31, 1: 32-63, 2: >= 64
    match_len: int
    probe: bool = False
    real: int = 0  # how many of tokens came from the copy


class _Stat:
    __slots__ = ("n", "s", "offers", "landed")

    def __init__(self) -> None:
        self.n = 0.0
        self.s = 0.0
        self.offers = 0
        self.landed = 0


class _Stream:
    __slots__ = ("ctx", "n", "h", "own", "prev", "cooldown", "pending", "own_ema", "prompt_tail", "offers",
                 "out_len", "last_seen")

    def __init__(self, own_ema: float) -> None:
        self.ctx = np.empty(1024, dtype=np.int64)
        self.n = 0
        self.h = 0
        self.own: Dict[int, int] = {}
        self.prev = -1
        self.cooldown = 0
        self.pending: Deque[tuple] = deque()  # one entry per offered copy, oldest first (overlap: 1-2 deep)
        self.own_ema = own_ema
        self.prompt_tail = np.zeros(0, dtype=np.int64)
        self.offers = 0
        self.out_len = 0  # output ids appended through extend_stream
        self.last_seen = time.monotonic()

    def append(self, toks) -> None:
        if isinstance(toks, np.ndarray):
            toks = toks.tolist()
        n = self.n
        need = n + len(toks)
        if need > self.ctx.size:
            cap = self.ctx.size
            while cap < need:
                cap *= 2
            nc = np.empty(cap, dtype=np.int64)
            nc[:n] = self.ctx[:n]
            self.ctx = nc
        ctx, h, own, prev = self.ctx, self.h, self.own, self.prev
        for t in toks:
            t = int(t)
            ctx[n] = t
            n += 1
            if n > WINDOW:
                h = _roll_int(h, int(ctx[n - WINDOW - 1]), t)
            elif n == WINDOW:
                h = _hash_window_int(ctx[:WINDOW].tolist())
            else:
                continue
            prev = own.get(h, -1)
            own[h] = n
        self.n, self.h, self.prev = n, h, prev


class RecallEngine:
    """Copy-draft decisions per stream on top of a RecallRecord. Single caller thread (the scheduler)."""

    SOURCES = ("own", "record")

    def __init__(self, record: RecallRecord, gamma: int, min_match: int = WINDOW, cooldown_rounds: int = 4,
                 probe_start: int = 16, probe_max: int = 4096, async_add: bool = True, prior_frac: float = 0.8,
                 prior_weight: float = 2.0, decay: float = 0.98, min_n: int = 2, own_alpha: float = 0.25):
        self.record = record
        self.gamma = int(gamma)
        self.min_match = max(WINDOW, int(min_match))
        self.cooldown_rounds = int(cooldown_rounds)
        self.probe_start = int(probe_start)
        self.probe_max = int(probe_max)
        self.prior = float(prior_frac) * self.gamma
        self.prior_w = float(prior_weight)
        self.decay = float(decay)
        self.min_n = max(1, int(min_n))
        self.own_alpha = float(own_alpha)
        self._streams: Dict[object, _Stream] = {}
        self._stats: Dict[Tuple[str, int], _Stat] = {(s, t): _Stat() for s in self.SOURCES for t in range(3)}
        self._probe_iv = {s: self.probe_start for s in self.SOURCES}
        self._probe_left = {s: self.probe_start for s in self.SOURCES}
        self._own_global = 0.5 * self.gamma
        self.offers = 0
        self.probes = 0
        self.landed_tokens = 0
        self.unused = 0
        self.lookups = 0
        self.rounds_own = 0
        self.streams_begun = 0
        self.gc_streams = 0
        self.last_clear = ""
        self._last_tick = 0.0
        self._q: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        if async_add:
            self._q = queue.Queue()
            self._thread = threading.Thread(target=self._bg, name="recall-writer", daemon=True)
            self._thread.start()

    # -- background ------------------------------------------------------------------------------
    def _bg(self) -> None:
        q = self._q
        assert q is not None
        while True:
            item = q.get()
            try:
                if item is None:
                    return
                kind, payload = item
                if kind == "add":
                    self.record.add(payload)
                elif kind == "credit":
                    self.record.credit(*payload)
                elif kind == "stats":
                    self._dump_stats(payload)
                elif kind == "clear":
                    self._do_clear()
            except Exception as e:  # never take the server down
                self.record.last_error = "%s: %s" % (type(e).__name__, e)
            finally:
                q.task_done()

    def _submit(self, kind: str, payload) -> None:
        if kind == "credit" and not getattr(self.record, "_heat_on", False):
            return
        if self._q is not None:
            self._q.put((kind, payload))
            return
        if kind == "add":
            self.record.add(payload)
        elif kind == "credit":
            self.record.credit(*payload)
        elif kind == "stats":
            self._dump_stats(payload)
        elif kind == "clear":
            self._do_clear()

    def _do_clear(self) -> None:
        try:
            self.record.clear()
            for s in self._stats.values():
                s.n = s.s = 0.0
                s.offers = s.landed = 0
            self.offers = self.probes = self.landed_tokens = self.unused = self.lookups = 0
            self.last_clear = "ok %s" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        except Exception as e:
            self.last_clear = "error %s: %s" % (type(e).__name__, e)

    def _dump_stats(self, data: dict) -> None:
        d = self.record.dir
        if not d or not os.path.isdir(d):
            return
        tmp = os.path.join(d, ".stats.json.tmp")
        _write_private(tmp, json.dumps(data, ensure_ascii=False, sort_keys=True))
        os.replace(tmp, os.path.join(d, "stats.json"))

    def flush(self, timeout: float = 30.0) -> None:
        if self._q is None:
            return
        t0 = time.time()
        while self._q.unfinished_tasks and time.time() - t0 < timeout:
            time.sleep(0.002)

    def tick(self, now: Optional[float] = None) -> None:
        """Call about once per scheduler step: every 2 s dumps stats.json and honours a clear.request file."""
        now = time.monotonic() if now is None else now
        if now - self._last_tick < 2.0:
            return
        self._last_tick = now
        if self.record.poll_clear_request():
            if self.record.writer:
                self._submit("clear", None)
            else:
                self.last_clear = "refused: read-only"
        self._submit("stats", self.stats())
        # streams the scheduler never ended (aborted in the queue, lost on retract): drop after 10 min idle
        stale = [rid for rid, st in self._streams.items() if now - st.last_seen > 600.0]
        for rid in stale:
            self._streams.pop(rid, None)
        self.gc_streams += len(stale)

    # -- streams ---------------------------------------------------------------------------------
    def begin_stream(self, rid, prompt_ids, cached_len: int = 0) -> None:
        p = np.asarray(prompt_ids, dtype=np.int64).ravel()
        st = _Stream(self._own_global)
        st.prompt_tail = p[-64:].copy()
        st.append(p[-MAX_BACK:])
        self._streams[rid] = st
        self.streams_begun += 1
        new = p[max(0, int(cached_len)):]
        if new.size >= WINDOW and self.record.writer:
            self._submit("add", new.copy())

    def extend_stream(self, rid, new_ids) -> None:
        st = self._streams.get(rid)
        if st is not None and len(new_ids):
            st.append(new_ids)
            st.out_len += len(new_ids)
            st.last_seen = time.monotonic()

    def stream_out_len(self, rid) -> int:
        """Output ids appended to the stream so far, -1 when the stream is unknown."""
        st = self._streams.get(rid)
        return -1 if st is None else st.out_len

    def end_stream(self, rid, output_ids=None, finished: bool = True) -> None:
        st = self._streams.pop(rid, None)
        if st is None or not finished or output_ids is None or not self.record.writer:
            return
        out = np.asarray(output_ids, dtype=np.int64).ravel()
        if out.size == 0:
            return
        seg = np.concatenate((st.prompt_tail, out))
        if seg.size >= WINDOW:
            self._submit("add", seg)

    def has_stream(self, rid) -> bool:
        return rid in self._streams

    def drop_pending(self, rid) -> None:
        """Forget copies offered to a stream whose rounds will never report (retracted, aborted)."""
        st = self._streams.get(rid)
        if st is not None:
            st.pending.clear()

    # -- drafting --------------------------------------------------------------------------------
    def _expect(self, src: str, tier: int) -> float:
        s = self._stats[(src, tier)]
        return (s.s + self.prior * self.prior_w) / (s.n + self.prior_w)

    def _own_candidate(self, st: _Stream, n_tok: int):
        p = st.prev
        n = st.n
        if p < WINDOW or p >= n:
            return None
        ctx = st.ctx
        if not np.array_equal(ctx[p - WINDOW:p], ctx[n - WINDOW:n]):
            return None
        back = min(MAX_BACK - WINDOW, p - WINDOW)
        ext = 0
        if back > 0:
            a = ctx[p - WINDOW - back:p - WINDOW][::-1]
            b = ctx[n - WINDOW - back:n - WINDOW][::-1]
            mm = np.flatnonzero(a != b)
            ext = int(mm[0]) if mm.size else back
        L = WINDOW + ext
        if L < self.min_match:
            return None
        period = n - p
        if period >= n_tok:
            toks = ctx[p:p + n_tok].copy()
        else:  # the copy runs into the stream's own tail: repeat with that period
            toks = np.empty(n_tok, dtype=np.int64)
            toks[:period] = ctx[p:n]
            for j in range(period, n_tok):
                toks[j] = toks[j - period]
        return ("own", tier_of(L), L, toks, n_tok)

    def _record_candidate(self, st: _Stream, n_tok: int):
        rec = self.record
        if rec.end_pos - rec.base_pos < WINDOW:
            return None
        self.lookups += 1
        r = rec.lookup(st.ctx[max(0, st.n - MAX_BACK):st.n], MAX_BACK, h=st.h)
        if r is None:
            return None
        p, L = r
        if L < self.min_match:
            return None
        cont = rec.continuation(p, n_tok, min_n=min(self.min_n, n_tok))
        if cont is None:
            return None
        real = int(cont.size)
        if real < n_tok:
            cont = np.concatenate((cont, np.full(n_tok - real, cont[-1], dtype=np.int64)))
        return ("record", tier_of(L), L, cont, real, p)

    def propose(self, rid, own_ema: Optional[float] = None, n: Optional[int] = None) -> Optional[Proposal]:
        st = self._streams.get(rid)
        if st is None or st.n < WINDOW:
            return None
        if st.cooldown > 0:
            st.cooldown -= 1
            return None
        n_tok = int(n or self.gamma)
        own = st.own_ema if own_ema is None else float(own_ema)
        best = None
        best_e = -1.0
        for cand in (self._own_candidate(st, n_tok), self._record_candidate(st, n_tok)):
            if cand is None:
                continue
            e = self._expect(cand[0], cand[1])
            if e > best_e:  # strict: the stream's own context (asked first) wins ties
                best, best_e = cand, e
        if best is None:
            return None
        src, tier, L, toks, real = best[:5]
        pos = best[5] if len(best) > 5 else -1
        probe = False
        if best_e <= own:
            left = self._probe_left[src] - 1
            if left > 0:
                self._probe_left[src] = left
                return None
            self._probe_left[src] = self._probe_iv[src]
            probe = True
            self.probes += 1
        st.pending.append((src, tier, probe, real, own, pos))
        st.offers += 1
        st.last_seen = time.monotonic()
        self.offers += 1
        return Proposal(tokens=toks, source=src, tier=tier, match_len=L, probe=probe, real=real)

    def feedback(self, rid, landed: int, used: bool = True) -> None:
        """Report the oldest copy offered to ``rid``: landed = copied tokens the target accepted that round;
        used=False when the copy was not verified after all (the lane's own drafts ran instead)."""
        st = self._streams.get(rid)
        if st is None or not st.pending:
            return
        src, tier, probe, real, own, pos = st.pending.popleft()
        if not used:
            self.unused += 1
            return
        landed = max(0, min(int(landed), int(real)))
        if landed and src == "record" and pos >= 0:
            self._submit("credit", (pos, landed))
        s = self._stats[(src, tier)]
        s.s = s.s * self.decay + landed
        s.n = s.n * self.decay + 1.0
        s.offers += 1
        s.landed += landed
        self.landed_tokens += landed
        if landed == 0:
            st.cooldown = self.cooldown_rounds
        if probe:
            if landed > own:
                self._probe_iv[src] = self.probe_start
            else:
                self._probe_iv[src] = min(self._probe_iv[src] * 2, self.probe_max)
            self._probe_left[src] = self._probe_iv[src]

    def observe_own(self, rid, accepted: float) -> None:
        """The stream's own drafts landed ``accepted`` tokens this round (a round without a copy)."""
        a = self.own_alpha
        st = self._streams.get(rid)
        if st is not None:
            st.own_ema = (1.0 - a) * st.own_ema + a * float(accepted)
        self._own_global = 0.99 * self._own_global + 0.01 * float(accepted)
        self.rounds_own += 1

    # -- reporting -------------------------------------------------------------------------------
    def stats(self) -> dict:
        tiers = {}
        for (src, t), s in self._stats.items():
            tiers["%s_%s" % (src, ("16-31", "32-63", "64+")[t])] = {
                "offers": s.offers, "landed": s.landed,
                "rate": round(s.landed / s.offers, 4) if s.offers else None,
                "expect": round(self._expect(src, t), 4)}
        d = self.record.stats()
        d.update({
            "gamma": self.gamma, "offers": self.offers, "probes": self.probes, "landed_tokens": self.landed_tokens,
            "unused": self.unused, "lookups": self.lookups, "own_rounds": self.rounds_own,
            "own_global_ema": round(self._own_global, 4), "active_streams": len(self._streams),
            "streams_begun": self.streams_begun, "gc_streams": self.gc_streams,
            "probe_interval": dict(self._probe_iv), "tiers": tiers,
            "last_clear": self.last_clear, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return d

    def clear(self) -> None:
        self._submit("clear", None)
        self.flush()

    def close(self) -> None:
        if self._q is not None and self._thread is not None:
            self.flush()
            self._q.put(None)
            self._thread.join(timeout=10)
            self._q = None
        self.record.close()


# ----------------------------------------------------------------------------------------------- helpers
def tokenizer_identity(tokenizer, model_name: str = "") -> dict:
    h = hashlib.sha256()
    vocab = tokenizer.get_vocab()
    for tok, idx in sorted(vocab.items(), key=lambda kv: (kv[1], kv[0])):
        h.update(("%d\t%s\n" % (idx, tok)).encode("utf-8", "surrogatepass"))
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    try:
        for idx in sorted(added):
            h.update(("A%d\t%s\n" % (idx, getattr(added[idx], "content", added[idx]))).encode("utf-8", "surrogatepass"))
    except Exception:
        pass
    try:
        js = json.loads(tokenizer.backend_tokenizer.to_str())
        merges = (js.get("model") or {}).get("merges")
        if merges:
            h.update(json.dumps(merges, ensure_ascii=False).encode("utf-8", "surrogatepass"))
    except Exception:
        pass
    return {"tokenizer_digest": h.hexdigest(), "vocab_size": len(vocab), "model": str(model_name)}


_ENGINE: Optional[RecallEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> Optional[RecallEngine]:
    return _ENGINE


def engine_from_env(gamma: int, identity: dict, environ=None) -> Optional[RecallEngine]:
    """Process singleton configured by SGLANG_D_RECALL* variables (DSV41_RECALL* as fallback);
    None when Recall is off."""
    global _ENGINE
    env = os.environ if environ is None else environ
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE
        if _env(ENV_NEW["on"], ENV_OLD["on"], "0", env).strip().lower() not in _TRUE:
            return None
        digest = str(identity.get("tokenizer_digest") or "unknown")
        directory = _env(ENV_NEW["dir"], ENV_OLD["dir"], "", env) or os.path.join(
            os.path.expanduser("~/.cache/sglang_d_recall"), digest[:16])
        backend = _env(ENV_NEW["backend"], ENV_OLD["backend"], "disk", env).strip().lower()
        device = _env(ENV_NEW["device"], ENV_OLD["device"], "", env) or None
        gib = float(_env(ENV_NEW["gib"], ENV_OLD["gib"], "0.0625", env))
        read_only = _env(ENV_NEW["read_only"], ENV_OLD["read_only"], "0", env).strip().lower() in _TRUE
        min_free_gib = float(_env(ENV_NEW["min_free_gib"], ENV_OLD["min_free_gib"], "4", env))
        if backend in ("gpu", "cuda"):
            try:
                rec = RecallRecord(directory, identity, gib=gib, read_only=read_only,
                                   min_free_gib=min_free_gib, backend="cuda", device=device, environ=env)
            except Exception as e:  # noqa: BLE001
                print("[recall] cuda backend unavailable (%r); falling back to the memory backend" % (e,), flush=True)
                rec = RecallRecord(directory, identity, gib=gib, read_only=read_only,
                                   min_free_gib=min_free_gib, backend="memory", environ=env)
        elif backend == "memory":
            rec = RecallRecord(directory, identity, gib=gib, read_only=read_only,
                               min_free_gib=min_free_gib, backend="memory", environ=env)
        else:
            rec = RecallRecord(directory, identity, gib=gib, read_only=read_only,
                               min_free_gib=min_free_gib, backend="disk", environ=env)
        _ENGINE = RecallEngine(
            rec, gamma,
            min_match=int(_env(ENV_NEW["min_match"], ENV_OLD["min_match"], str(WINDOW), env)),
            prior_frac=float(_env(ENV_NEW["prior"], ENV_OLD["prior"], "0.8", env)))
        return _ENGINE
