"""影子仪表核心模块：只观测、不改行为。

设计约束（生产推理服务对探针的硬要求）：
  * 主线程调用 ``record()`` 只做一次 deque.append，无 I/O、无 json 序列化、无锁；
  * 落盘由后台守护线程按固定周期完成，写失败只计数、绝不抛回主线程；
  * 环境变量 ``VLLM_D_RECALL_SHADOW`` 一键开关，关闭时 ``get_recorder()``
    返回 ``NullRecorder``，所有调用都是空操作、零开销、不起线程。

只依赖标准库，Python 3.10+。
"""

from __future__ import annotations

import atexit
import collections
import json
import os
import random
import re
import tempfile
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

__all__ = [
    "ENV_ENABLE",
    "ENV_DIR",
    "ENV_TOPK_RATE",
    "DEFAULT_DIR_NAME",
    "DEFAULT_TOPK_RATE",
    "KIND_STEP",
    "KIND_REQ",
    "KIND_ACC",
    "KIND_WREQ",
    "KIND_WSTEP",
    "KIND_TOPK",
    "SCHEMA",
    "NullRecorder",
    "ShadowRecorder",
    "enabled",
    "get_recorder",
    "default_out_dir",
    "sample",
    "seed",
    "load",
    "validate",
    "_reset_for_tests",
]

# --------------------------------------------------------------------------
# 环境变量与常量
# --------------------------------------------------------------------------

ENV_ENABLE = "VLLM_D_RECALL_SHADOW"
ENV_DIR = "VLLM_D_RECALL_SHADOW_DIR"
ENV_TOPK_RATE = "VLLM_D_RECALL_SHADOW_TOPK_RATE"

DEFAULT_DIR_NAME = "vllm_d_recall_shadow"
DEFAULT_TOPK_RATE = 0.01

_TRUTHY = frozenset({"1", "true", "on", "yes"})
_WRITE_BATCH = 4096

# --------------------------------------------------------------------------
# 记录 schema（只用于文档与离线校验，record() 路径不做任何校验）
# --------------------------------------------------------------------------

KIND_STEP = "step"
KIND_REQ = "req"
KIND_ACC = "acc"
KIND_WREQ = "wreq"
KIND_WSTEP = "wstep"
KIND_TOPK = "topk"

SCHEMA: Dict[str, Tuple[str, ...]] = {
    # 调度器每步一条：step(int) n_dec(int 本步解码请求数) n_pre(int 预填请求数)
    # tok_pre(int 预填 token 数) bt(int 整批送验 token 数，等于各解码请求 w+1 之和)
    # dt_ms(float 距上一步的墙钟间隔)
    KIND_STEP: ("step", "n_dec", "n_pre", "tok_pre", "bt", "dt_ms"),
    # 调度器每步每个解码请求一条：step rid(str) pos(int 该请求本步第一个草稿位
    # 在序列里的位置) temp(float) top_p(float) top_k(int，-1 表示不限)
    # st(bool 是否结构化输出) w(int 送验宽度)
    KIND_REQ: ("step", "rid", "pos", "temp", "top_p", "top_k", "st", "w"),
    # 调度器拿到验收结果时一条：step(送验那一步) rid pos v(int 送验个数)
    # a(int 接受个数，不含 bonus)
    KIND_ACC: ("step", "rid", "pos", "v", "a"),
    # worker 侧每步每请求一条：wstep(int worker 本地步号) rid pos w
    # rec(int 位掩码，第 j 位为 1 表示第 j 个草稿来自召回)
    # qmax(list[float] 逐位草稿分布最大概率) qx(list[float] 逐位草稿 token
    # 在草稿分布里的概率) conf(list[float] 置信度头输出，没有就空列表)
    KIND_WREQ: ("wstep", "rid", "pos", "w", "rec", "qmax", "qx", "conf"),
    # worker 侧每步一条：wstep n bt fwd_ms(float) drf_ms(float)
    KIND_WSTEP: ("wstep", "n", "bt", "fwd_ms", "drf_ms"),
    # worker 侧抽样：wstep rid pos j(int 草稿位)
    # tgt(list[[int, float]] 目标分布前 32) drf(list[[int, float]] 草稿分布前 32)
    KIND_TOPK: ("wstep", "rid", "pos", "j", "tgt", "drf"),
}


def validate(kind: str, rec: Any) -> List[str]:
    """返回 rec 相对 kind schema 缺失的键列表；未知 kind 返回标记项。"""
    keys = SCHEMA.get(kind)
    if keys is None:
        return ["__unknown_kind__:" + str(kind)]
    if not isinstance(rec, dict):
        return ["__not_a_dict__"]
    return [k for k in keys if k not in rec]


# --------------------------------------------------------------------------
# 开关
# --------------------------------------------------------------------------

_ENABLED_CACHE: Optional[bool] = None
_RNG = random.Random()


def enabled() -> bool:
    """环境变量 VLLM_D_RECALL_SHADOW 为 1/true/on/yes（不分大小写）时为真。进程内缓存。"""
    global _ENABLED_CACHE
    if _ENABLED_CACHE is None:
        raw = os.environ.get(ENV_ENABLE, "")
        _ENABLED_CACHE = str(raw).strip().lower() in _TRUTHY
    return _ENABLED_CACHE


def default_out_dir() -> str:
    """默认输出目录：环境变量 VLLM_D_RECALL_SHADOW_DIR，否则 temp 目录同名子目录。"""
    d = os.environ.get(ENV_DIR)
    if d:
        return d
    return os.path.join(tempfile.gettempdir(), DEFAULT_DIR_NAME)


def seed(value: int) -> None:
    """给模块级随机实例设定种子（用于可复现的抽样）。"""
    _RNG.seed(value)


def sample(rate: Optional[float] = None) -> bool:
    """按 VLLM_D_RECALL_SHADOW_TOPK_RATE（默认 0.01）抽样，返回是否抽中。"""
    if rate is None:
        raw = os.environ.get(ENV_TOPK_RATE)
        try:
            rate = float(raw) if raw is not None else DEFAULT_TOPK_RATE
        except (TypeError, ValueError):
            rate = DEFAULT_TOPK_RATE
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        return False
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    return _RNG.random() < rate


# --------------------------------------------------------------------------
# 记录器
# --------------------------------------------------------------------------

class NullRecorder:
    """关闭态记录器：所有方法都是空操作，不产生文件、不起线程。"""

    __slots__ = ()

    def record(self, kind: str, rec: dict) -> None:
        return None

    def stats(self) -> Dict[str, int]:
        return {"recorded": 0, "written": 0, "dropped": 0, "write_errors": 0, "files": 0}

    def flush(self, timeout_s: float = 5.0) -> bool:
        return True

    def close(self) -> None:
        return None

    def sample(self, rate: Optional[float] = None) -> bool:
        return False


_NULL_RECORDER = NullRecorder()


class ShadowRecorder:
    """后台刷盘的观测记录器。

    主线程只做 ``deque.append``；后台守护线程周期性把缓冲写成 JSONL。
    """

    def __init__(
        self,
        out_dir: Optional[str] = None,
        role: str = "unknown",
        rank: int = 0,
        capacity: int = 200000,
        flush_interval_s: float = 1.0,
        max_file_mb: float = 256,
    ) -> None:
        self.out_dir = os.path.abspath(out_dir or default_out_dir())
        self.role = str(role)
        self.rank = int(rank)
        self.capacity = max(1, int(capacity))
        self.flush_interval_s = max(0.001, float(flush_interval_s))
        self.max_file_mb = float(max_file_mb)
        self._max_bytes = max(1, int(self.max_file_mb * 1024 * 1024))

        self._buf: "collections.deque" = collections.deque(maxlen=self.capacity)
        self._wlock = threading.Lock()  # 只保护写侧，不进主线程 record 路径
        self._stop = threading.Event()
        self._wake = threading.Event()

        self._fh = None
        self._seq = 0
        self._bytes = 0

        self._recorded = 0
        self._written = 0
        self._dropped = 0
        self._write_errors = 0
        self._files = 0
        self._closed = False

        self._thread = threading.Thread(
            target=self._run, name="shadow-%s-r%d" % (self.role, self.rank), daemon=True
        )
        self._thread.start()

    # -- 主线程热路径 ------------------------------------------------------

    def record(self, kind: str, rec: dict) -> None:
        """O(1) 追加一条观测；不做 I/O、不做序列化、不加锁。"""
        buf = self._buf
        if len(buf) >= self.capacity:
            self._dropped += 1
        buf.append((time.time(), kind, rec))
        self._recorded += 1

    def sample(self, rate: Optional[float] = None) -> bool:
        return sample(rate)

    # -- 统计 --------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {
            "recorded": self._recorded,
            "written": self._written,
            "dropped": self._dropped,
            "write_errors": self._write_errors,
            "files": self._files,
        }

    # -- 落盘 --------------------------------------------------------------

    def _filename(self, seq: int) -> str:
        return "%s-r%d-p%d-%04d.jsonl" % (self.role, self.rank, os.getpid(), seq)

    def _ensure_file(self):
        if self._fh is not None:
            return self._fh
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, self._filename(self._seq))
        fh = open(path, "ab", buffering=1024 * 1024)
        self._fh = fh
        try:
            self._bytes = fh.tell()
        except OSError:
            self._bytes = 0
        self._files += 1
        return fh

    def _rotate(self) -> None:
        fh = self._fh
        self._fh = None
        self._bytes = 0
        self._seq += 1
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass

    def _write_batch(self, batch: List[Tuple[float, str, dict]]) -> None:
        try:
            fh = self._ensure_file()
        except Exception:
            self._write_errors += 1
            return
        if fh is None:
            self._write_errors += 1
            return
        for ts, kind, rec in batch:
            try:
                obj = {"t": ts, "k": kind}
                if rec:
                    obj.update(rec)
                data = (
                    json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"
                ).encode("utf-8")
                if self._bytes and (self._bytes + len(data)) > self._max_bytes:
                    self._rotate()
                    fh = self._ensure_file()
                fh.write(data)
                self._bytes += len(data)
                self._written += 1
            except Exception:
                self._write_errors += 1

    def _drain_locked(self) -> None:
        """调用方必须持有 ``self._wlock``。把缓冲全部排空（分批以限制峰值内存）。"""
        buf = self._buf
        while buf:
            batch: List[Tuple[float, str, dict]] = []
            while buf and len(batch) < _WRITE_BATCH:
                batch.append(buf.popleft())
            if batch:
                self._write_batch(batch)

    def _drain(self) -> None:
        with self._wlock:
            self._drain_locked()

    def _run(self) -> None:
        while True:
            self._wake.wait(self.flush_interval_s)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self._drain()
            except Exception:
                self._write_errors += 1
        try:
            self._drain()
        except Exception:
            self._write_errors += 1

    # -- 生命周期 ----------------------------------------------------------

    def flush(self, timeout_s: float = 5.0) -> bool:
        """同步把缓冲里的记录落盘。返回是否在超时内完成。"""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        for _ in range(1024):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            if not self._wlock.acquire(timeout=remaining):
                return False
            try:
                self._drain_locked()
                if self._fh is not None:
                    try:
                        self._fh.flush()
                    except OSError:
                        self._write_errors += 1
            finally:
                self._wlock.release()
            if not self._buf:
                return True
        return not self._buf

    def close(self) -> None:
        """停后台线程并同步刷盘（可重复调用）。"""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.flush_interval_s + 1.0))
        try:
            self.flush(timeout_s=10.0)
        except Exception:
            self._write_errors += 1
        with self._wlock:
            fh = self._fh
            self._fh = None
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass


# --------------------------------------------------------------------------
# 单例注册
# --------------------------------------------------------------------------

_REGISTRY: Dict[Tuple[str, int], ShadowRecorder] = {}
_REGISTRY_LOCK = threading.Lock()


def get_recorder(role: str, rank: int = 0) -> Any:
    """开启时返回 (role, rank) 唯一 ShadowRecorder；关闭时返回 NullRecorder。"""
    if not enabled():
        return _NULL_RECORDER
    key = (str(role), int(rank))
    with _REGISTRY_LOCK:
        rec = _REGISTRY.get(key)
        if rec is None:
            rec = ShadowRecorder(out_dir=default_out_dir(), role=role, rank=rank)
            _REGISTRY[key] = rec
        return rec


def _close_all() -> None:
    with _REGISTRY_LOCK:
        recorders = list(_REGISTRY.values())
        _REGISTRY.clear()
    for rec in recorders:
        try:
            rec.close()
        except Exception:
            pass


atexit.register(_close_all)


def _reset_for_tests() -> None:
    """清掉模块级缓存（开关结果、记录器单例），供测试切换环境变量用。"""
    global _ENABLED_CACHE
    with _REGISTRY_LOCK:
        recorders = list(_REGISTRY.values())
        _REGISTRY.clear()
    for rec in recorders:
        try:
            rec.close()
        except Exception:
            pass
    _ENABLED_CACHE = None


# --------------------------------------------------------------------------
# 读回
# --------------------------------------------------------------------------

_FILE_RE = re.compile(r"^(?P<role>.+)-r(?P<rank>\d+)-p(?P<pid>\d+)-(?P<seq>\d+)\.jsonl$")


def load(out_dir: str, role: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
    """按文件 seq 顺序读回所有记录（dict）。role 为 None 时读全部 role。"""
    try:
        names = os.listdir(out_dir)
    except OSError:
        return
    files: List[Tuple[int, str]] = []
    for name in names:
        m = _FILE_RE.match(name)
        if m is None:
            continue
        if role is not None and m.group("role") != role:
            continue
        files.append((int(m.group("seq")), name))
    files.sort(key=lambda item: (item[0], item[1]))
    for _, name in files:
        path = os.path.join(out_dir, name)
        try:
            fh = open(path, "r", encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    yield obj
