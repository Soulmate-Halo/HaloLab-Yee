#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解码步级耗时分解（本地分析 torch profiler chrome trace，只用标准库）。

把同一次 torch profiler 采集的多 rank trace（文件名里带 ``TP-0..TP-3``）切成解码步，
按类别统计 kernel 耗时，并做跨 rank 对照。

步的界定
--------
1. 首选：CUDA graph 启动（``cudaGraphLaunch``）+ ``args.correlation`` 把 kernel 归到某次
   图启动；kernel 数最多的那次图（>= 最大值的一半）当作 verify 图，每次 verify 图的首个
   kernel 开始时刻就是一个步边界。一步 = [第 k 个 verify 图开始, 第 k+1 个 verify 图开始)；
   最后一个不完整的步丢弃。
2. 其次：GPU 时间轴上"每步都出现"的固定 kernel（出现 >= 3 次且间隔稳定），用它的各次
   出现当步边界。
3. 再次：``gpu_user_annotation`` 里名字以 ``step[`` 开头的长事件当步边界。

类别
----
通信：``roce_allreduce``（RoCE 单发 all-reduce）、``roce_allgather``（RoCE all-gather）、
``nccl``、``comm_other``（其他通信）。
计算：``moe``、``gemm``（稠密 GEMM）、``attention``、``draft``（草稿图）、``compute_other``。
通信按 kernel 名优先判定（草稿图里的通信 kernel 仍算通信），然后才是草稿图，最后是计算子类。

跨 rank 对照
------------
逐步比较各 rank 的非通信计算时长，最慢 rank = 计算最长的 rank。

* ``comm_delta_ms`` = 本 rank 通信时长 - 最慢 rank 通信时长（<= 0，即比最慢 rank 早/晚多少）
* ``comm_wait_ms`` = 最慢 rank 通信时长 - 本 rank 通信时长（>= 0，即本 rank 通信先完成后要等多久）

用法
----
::

    python -m yee.tools.step_breakdown TRACE_OR_DIR [更多 ...] [--rank R ...] --out OUT_DIR

输出 ``OUT_DIR/breakdown.json`` 与 ``OUT_DIR/breakdown.md``；stdout 末行形如::

    STEP_BREAKDOWN: ranks=4 steps=10 wall_ms=... comm_ms=... idle_ms=...

steps 是各 rank 共同拥有的步数（跨 rank 对齐后的步数），wall/comm/idle 是所有 rank、
这些共同步的合计。

退出码：0 正常；2 输入不存在或参数不合法；3 没找到可用 trace。
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import os
import re
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# trace 的时间单位是微秒
US_PER_MS = 1000.0
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")

COMM_BUCKETS = ("roce_allreduce", "roce_allgather", "nccl", "comm_other")
COMPUTE_BUCKETS = ("moe", "gemm", "attention", "draft", "compute_other")
ALL_BUCKETS = COMM_BUCKETS + COMPUTE_BUCKETS

# ---------------------------------------------------------------- 名称匹配
RE_NCCL = re.compile(r"nccl", re.I)
RE_ROCE = re.compile(r"roce|b12x_?comm", re.I)  # b12x 的 MoE/稠密 GEMM kernel 名也带 b12x，单凭 b12x 不算通信
RE_ALLREDUCE = re.compile(
    r"all[_-]?reduce|reduce[_-]?all|red[_-]?scat|one[_-]?shot|single[_-]?shot", re.I)
RE_ALLGATHER = re.compile(r"all[_-]?gather|gath", re.I)
RE_COMM_GEN = re.compile(
    r"all[_-]?reduce|all[_-]?gather|broadcast|reduce[_-]?scatter|all[_-]?to[_-]?all|"
    r"send[_-]?recv|send[_-]|recv[_-]", re.I)
RE_MOE = re.compile(
    r"moe|expert|grouped[_-]?gemm|group[_-]?gemm|fused[_-]?moe|router|swiglu|silu|"
    r"deep[_-]?gemm|topk[_-]?gate", re.I)
RE_GEMM = re.compile(
    r"gemm|cutlass|nvjet|scaled[_-]?mm|cublas|matmul|sgemm|hgemm|bmm|w8a8|w4a8|mxfp", re.I)
RE_ATTN = re.compile(
    r"attn|attention|flash[_-]?mla|sparse|indexer|mqa|page[_-]?split|page[_-]?mark|"
    r"candidate|softmax", re.I)

RE_RANK = re.compile(r"TP[-_]?(\d+)", re.I)


# ---------------------------------------------------------------- 小工具

def _num(value: Any) -> Optional[float]:
    """把 trace 里的数值字段读成 float；读不出返回 None。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _ms(us: float) -> float:
    return round(us / US_PER_MS, 4)


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(statistics.mean(values), 4) if values else None


def _union(intervals: Sequence[Tuple[float, float]]) -> float:
    """区间并集长度。"""
    if not intervals:
        return 0.0
    ordered = sorted(iv for iv in intervals if iv[1] > iv[0])
    if not ordered:
        return 0.0
    total = 0.0
    cur_s, cur_e = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_e:
            if end > cur_e:
                cur_e = end
        else:
            total += cur_e - cur_s
            cur_s, cur_e = start, end
    return total + cur_e - cur_s


def classify_comm(name: str) -> Optional[str]:
    """按 kernel 名判通信桶；不是通信返回 None。"""
    if RE_NCCL.search(name):
        return "nccl"
    if RE_ROCE.search(name):
        if RE_ALLGATHER.search(name):
            return "roce_allgather"
        if RE_ALLREDUCE.search(name):
            return "roce_allreduce"
        return "comm_other"
    if RE_COMM_GEN.search(name):
        return "comm_other"
    return None


def classify_compute(name: str) -> str:
    """按 kernel 名判计算子类（不含草稿图，草稿图靠 CUDA graph 归属判定）。"""
    if RE_MOE.search(name):
        return "moe"
    if RE_GEMM.search(name):
        return "gemm"
    if RE_ATTN.search(name):
        return "attention"
    return "compute_other"


# ---------------------------------------------------------------- 读 trace

def load_events(path: str) -> List[dict]:
    """读 chrome trace（.json / .json.gz），返回事件字典列表。"""
    if path.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
    if isinstance(data, dict):
        events = data.get("traceEvents")
        if events is None:
            events = data.get("trace_events")
        if not isinstance(events, list):
            raise ValueError("trace 里没有 traceEvents 列表")
    elif isinstance(data, list):
        events = data
    else:
        raise ValueError("不支持的 trace json 结构")
    return [ev for ev in events if isinstance(ev, dict)]


def gpu_events(events: Iterable[dict]) -> List[dict]:
    """挑出 GPU 时间轴上的事件（kernel / memcpy / memset），按 ts 排序。"""
    out = []
    for ev in events:
        if ev.get("cat") not in GPU_CATS:
            continue
        ph = ev.get("ph", "X")
        if ph != "X":
            continue
        ts = _num(ev.get("ts"))
        if ts is None:
            continue
        dur = _num(ev.get("dur"))
        if dur is None or dur < 0:
            dur = 0.0
        out.append({
            "name": str(ev.get("name") or ""),
            "cat": ev.get("cat"),
            "ts": ts,
            "dur": dur,
            "end": ts + dur,
            "args": ev.get("args") if isinstance(ev.get("args"), dict) else {},
        })
    out.sort(key=lambda e: e["ts"])
    return out


# ---------------------------------------------------------------- 切步

def graph_marks(events: Sequence[dict], gpu: Sequence[dict]) -> Tuple[List[float], set, Dict[str, Any]]:
    """用 CUDA graph 启动切步。

    返回 (verify 图的开始时刻列表, 草稿图 kernel 的 id 集合, 诊断信息)。
    """
    info: Dict[str, Any] = {"method": "graph_correlation"}
    runtime = [ev for ev in events if ev.get("cat") in ("cuda_runtime", "cuda_driver")]
    corr2rt = {}
    for ev in runtime:
        corr = (ev.get("args") or {}).get("correlation")
        if corr is not None:
            corr2rt[corr] = ev
    launches = [ev for ev in runtime if "GraphLaunch" in str(ev.get("name") or "")]
    launches.sort(key=lambda ev: _num(ev.get("ts")) or 0.0)
    info["n_graph_launches"] = len(launches)
    if not launches:
        return [], set(), info
    launch_idx = {id(ev): i for i, ev in enumerate(launches)}
    per: Dict[int, List[dict]] = {}
    owner: Dict[int, int] = {}
    for ev in gpu:
        if ev.get("cat") != "kernel":
            continue
        corr = ev["args"].get("correlation")
        rt = corr2rt.get(corr)
        if rt is None or id(rt) not in launch_idx:
            continue
        idx = launch_idx[id(rt)]
        per.setdefault(idx, []).append(ev)
        owner[id(ev)] = idx
    info["n_kernels_mapped"] = len(owner)
    if not per:
        return [], set(), info
    groups = []
    for idx, kernels in per.items():
        groups.append({
            "launch": idx,
            "n": len(kernels),
            "start": min(k["ts"] for k in kernels),
            "end": max(k["end"] for k in kernels),
        })
    groups.sort(key=lambda g: g["start"])
    nmax = max(g["n"] for g in groups)
    target = {g["launch"] for g in groups if g["n"] >= 0.5 * nmax}
    draft = {g["launch"] for g in groups if g["launch"] not in target}
    draft_ids = {kid for kid, gid in owner.items() if gid in draft}
    marks = [g["start"] for g in groups if g["launch"] in target]
    info["n_graph_groups"] = len(groups)
    info["n_verify_graphs"] = len(marks)
    info["n_draft_graphs"] = len(draft)
    info["verify_kernels_max"] = nmax
    return marks, draft_ids, info


def repeated_kernel_marks(gpu: Sequence[dict]) -> List[float]:
    """回退口径：找每步都出现的固定 kernel，用它的各次出现当步边界。"""
    kernels = [ev for ev in gpu if ev.get("cat") == "kernel"]
    if len(kernels) < 3:
        return []
    by_name: Dict[str, List[float]] = {}
    for ev in kernels:
        by_name.setdefault(ev["name"], []).append(ev["ts"])
    best = None
    for name, stamps in by_name.items():
        if len(stamps) < 3:
            continue
        stamps = sorted(stamps)
        deltas = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
        if any(d <= 0 for d in deltas):
            continue
        med = statistics.median(deltas)
        if med <= 0:
            continue
        if max(deltas) > 1.25 * med or min(deltas) < 0.75 * med:
            continue
        cand = (stamps[0], -len(stamps), name, stamps)
        if best is None or cand[:2] < best[:2]:
            best = cand
    return list(best[3]) if best else []


def annotation_marks(events: Sequence[dict]) -> List[float]:
    """回退口径：gpu_user_annotation 里名字以 ``step[`` 开头的长事件当步边界。"""
    ann = []
    for ev in events:
        if ev.get("cat") != "gpu_user_annotation":
            continue
        if not str(ev.get("name") or "").startswith("step["):
            continue
        ts = _num(ev.get("ts"))
        if ts is None:
            continue
        dur = _num(ev.get("dur")) or 0.0
        ann.append((ts, dur))
    if not ann:
        return []
    dmax = max(d for _t, d in ann)
    return sorted(ts for ts, d in ann if d >= 0.5 * dmax)


def cut_steps(events: Sequence[dict]) -> Tuple[List[Tuple[float, float]], set, List[dict], Dict[str, Any]]:
    """切出完整解码步。

    返回 (窗口列表, 草稿 kernel id 集合, GPU 事件列表, 诊断信息)。
    """
    gpu = gpu_events(events)
    marks, draft_ids, info = graph_marks(events, gpu)
    if len(marks) < 2:
        marks = repeated_kernel_marks(gpu)
        draft_ids = set()
        info = {"method": "repeated_kernel"}
    if len(marks) < 2:
        marks = annotation_marks(events)
        draft_ids = set()
        info = {"method": "gpu_annotation"}
    info.setdefault("method", "graph_correlation")
    info["n_marks"] = len(marks)
    info["n_gpu_events"] = len(gpu)
    windows = [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)]
    info["n_steps"] = len(windows)
    return windows, draft_ids, gpu, info


# ---------------------------------------------------------------- 单步统计

def _window_slice(gpu: Sequence[dict], starts: Sequence[float], start: float, end: float) -> List[dict]:
    """取与 [start, end) 有重叠的 GPU 事件（含跨过边界的 kernel）。"""
    if not gpu:
        return []
    hi = bisect.bisect_left(starts, end)
    lo = bisect.bisect_left(starts, start)
    # 向前回退，纳入跨过 step 起点的 kernel
    back = lo
    limit = max(0, lo - 5000)
    while back > limit and gpu[back - 1]["end"] > start:
        back -= 1
    return [ev for ev in gpu[back:hi] if ev["end"] > start and ev["ts"] < end]


def step_stats(windows: Sequence[Tuple[float, float]], gpu: Sequence[dict],
               draft_ids: set) -> List[Dict[str, Any]]:
    """逐窗口统计墙钟、GPU 忙/空闲、各类别 kernel 时长与通信 kernel 次数。"""
    steps: List[Dict[str, Any]] = []
    starts = [ev["ts"] for ev in gpu]
    for idx, (start, end) in enumerate(windows):
        inside = _window_slice(gpu, starts, start, end)
        buckets: Dict[str, float] = {b: 0.0 for b in ALL_BUCKETS}
        counts: Dict[str, int] = {b: 0 for b in ALL_BUCKETS}
        comm_names: Dict[str, float] = {}
        comm_calls: Dict[str, int] = {}
        intervals: List[Tuple[float, float]] = []
        compute_iv: List[Tuple[float, float]] = []
        for ev in inside:
            s = max(ev["ts"], start)
            e = min(ev["end"], end)
            if e <= s:
                continue
            dur = e - s
            intervals.append((s, e))
            name = ev["name"]
            comm = classify_comm(name)
            if comm is not None:
                buckets[comm] += dur
                counts[comm] += 1
                comm_names[name] = comm_names.get(name, 0.0) + dur
                comm_calls[name] = comm_calls.get(name, 0) + 1
                continue
            compute_iv.append((s, e))
            if id(ev) in draft_ids:
                bucket = "draft"
            else:
                bucket = classify_compute(name)
            buckets[bucket] += dur
            counts[bucket] += 1
        busy = _union(intervals)
        compute_union = _union(compute_iv)
        wall = end - start
        comm_ms = sum(buckets[b] for b in COMM_BUCKETS)
        compute_ms = sum(buckets[b] for b in COMPUTE_BUCKETS)
        n_comm = sum(counts[b] for b in COMM_BUCKETS)
        steps.append({
            "i": idx,
            "wall_ms": _ms(wall),
            "busy_ms": _ms(busy),
            "idle_ms": _ms(wall - busy),
            "comm_ms": _ms(comm_ms),
            # 暴露在关键路径上的通信：GPU 忙但没有任何计算 kernel 在跑的时间（通信与计算重叠的部分不算）
            "comm_exposed_ms": _ms(max(busy - compute_union, 0.0)),
            "compute_union_ms": _ms(compute_union),
            "compute_ms": _ms(compute_ms),
            "n_kernels": len(inside),
            "n_comm_kernels": n_comm,
            "cats_ms": {b: _ms(buckets[b]) for b in ALL_BUCKETS},
            "cat_counts": {b: counts[b] for b in ALL_BUCKETS},
            "comm_kernel_names": comm_names,
            "comm_kernel_calls": comm_calls,
        })
    return steps


# ---------------------------------------------------------------- 单个 rank

def analyze_rank(path: str, rank: Optional[int]) -> Dict[str, Any]:
    """分析一个 trace 文件。"""
    events = load_events(path)
    windows, draft_ids, gpu, info = cut_steps(events)
    steps = step_stats(windows, gpu, draft_ids)

    comm_kernels: Dict[str, Dict[str, Any]] = {}
    for step in steps:
        for name, dur in step["comm_kernel_names"].items():
            rec = comm_kernels.setdefault(name, {
                "name": name, "count": 0, "total_ms": 0.0,
                "bucket": classify_comm(name) or "comm_other",
            })
            rec["count"] += int(step.get("comm_kernel_calls", {}).get(name, 1))  # 调用次数，不是出现的步数
            rec["total_ms"] = round(rec["total_ms"] + _ms(dur), 4)
    for rec in comm_kernels.values():
        rec["calls_per_step"] = round(rec["count"] / len(steps), 2) if steps else None
        rec["mean_us"] = round(rec["total_ms"] * 1000.0 / rec["count"], 2) if rec["count"] else None
    top_comm = sorted(comm_kernels.values(), key=lambda r: (-r["total_ms"], r["name"]))[:20]

    cats_all = sorted({b for s in steps for b, v in s["cats_ms"].items() if v})
    summary = {
        "n_steps": len(steps),
        "wall_ms_mean": _mean([s["wall_ms"] for s in steps]),
        "busy_ms_mean": _mean([s["busy_ms"] for s in steps]),
        "idle_ms_mean": _mean([s["idle_ms"] for s in steps]),
        "comm_ms_mean": _mean([s["comm_ms"] for s in steps]),
        "comm_exposed_ms_mean": _mean([s["comm_exposed_ms"] for s in steps]),
        "compute_ms_mean": _mean([s["compute_ms"] for s in steps]),
        "n_comm_kernels_mean": _mean([float(s["n_comm_kernels"]) for s in steps]),
        "cats_ms_mean": {b: _mean([s["cats_ms"][b] for s in steps]) for b in cats_all},
    }
    return {
        "rank": rank,
        "trace": os.path.basename(path),
        "trace_path": path,
        "method": info.get("method"),
        "diag": info,
        "steps": steps,
        "summary": summary,
        "matched_comm_kernels": top_comm,
    }


# ---------------------------------------------------------------- 跨 rank

def cross_rank(rank_data: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """跨 rank 对照：最慢 rank、非通信计算时长、通信等待估计。"""
    usable = [r for r in rank_data if r.get("steps")]
    if not usable:
        return {"n_steps_common": 0, "per_step": [], "summary": {}}
    n_common = min(len(r["steps"]) for r in usable)
    per_step = []
    slow_counter: Dict[str, int] = {}
    comm_wait_sum: Dict[str, float] = {}
    compute_sum: Dict[str, float] = {}
    for i in range(n_common):
        rows = {}
        for r in usable:
            step = r["steps"][i]
            rows[str(r["rank"])] = {
                "wall_ms": step["wall_ms"],
                "busy_ms": step["busy_ms"],
                "idle_ms": step["idle_ms"],
                "comm_ms": step["comm_ms"],
                "compute_ms": step["compute_ms"],
            }
        slow = max(usable, key=lambda r: (r["steps"][i]["compute_ms"], -int(r["rank"])))
        slow_rank = slow["rank"]
        slow_comm = slow["steps"][i]["comm_ms"]
        wait: Dict[str, float] = {}
        delta: Dict[str, float] = {}
        for r in usable:
            key = str(r["rank"])
            comm = r["steps"][i]["comm_ms"]
            wait[key] = round(slow_comm - comm, 4)
            delta[key] = round(comm - slow_comm, 4)
            comm_wait_sum[key] = comm_wait_sum.get(key, 0.0) + wait[key]
            compute_sum[key] = compute_sum.get(key, 0.0) + r["steps"][i]["compute_ms"]
        slow_counter[str(slow_rank)] = slow_counter.get(str(slow_rank), 0) + 1
        per_step.append({
            "i": i,
            "slow_rank": slow_rank,
            "slow_compute_ms": slow["steps"][i]["compute_ms"],
            "slow_comm_ms": slow_comm,
            "ranks": rows,
            "comm_wait_ms": wait,
            "comm_delta_ms": delta,
        })
    summary = {
        "n_steps_common": n_common,
        "slow_rank_counts": slow_counter,
        "slow_rank_by_step": {str(s["i"]): s["slow_rank"] for s in per_step},
        "comm_wait_ms_mean": {k: round(v / n_common, 4) for k, v in sorted(comm_wait_sum.items())},
        "comm_delta_ms_mean": {k: round(-v / n_common, 4) for k, v in sorted(comm_wait_sum.items())},
        "compute_ms_mean": {k: round(v / n_common, 4) for k, v in sorted(compute_sum.items())},
        "note": ("comm_delta_ms = 本 rank 通信时长 - 最慢 rank 通信时长（<=0）；"
                 "comm_wait_ms = 最慢 rank 通信时长 - 本 rank 通信时长（>=0）"),
    }
    return {"n_steps_common": n_common, "per_step": per_step, "summary": summary}


# ---------------------------------------------------------------- 输入收集

def collect_traces(path: str) -> List[str]:
    """把文件或目录展开成 trace 文件列表（目录递归找 .json / .json.gz）。"""
    if os.path.isfile(path):
        return [path]
    found = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            low = name.lower()
            if low.endswith(".json") or low.endswith(".json.gz"):
                found.append(os.path.join(root, name))
    return sorted(found)


def detect_rank(path: str) -> Optional[int]:
    """从文件名里的 ``TP-<n>`` 认 rank。"""
    m = RE_RANK.search(os.path.basename(path))
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def assign_ranks(files: Sequence[str], explicit: Optional[Sequence[int]]) -> List[Optional[int]]:
    """决定每个文件的 rank：命令行指定优先，其次文件名，最后按顺序补空位。"""
    if explicit is not None:
        return list(explicit)
    ranks: List[Optional[int]] = [detect_rank(f) for f in files]
    used = {r for r in ranks if r is not None}
    nxt = 0
    for i, r in enumerate(ranks):
        if r is None:
            while nxt in used:
                nxt += 1
            ranks[i] = nxt
            used.add(nxt)
    return ranks


# ---------------------------------------------------------------- 输出

def render_markdown(result: Dict[str, Any]) -> str:
    ranks = result["ranks"]
    cross = result["cross_rank"]
    lines: List[str] = []
    lines.append("# 解码步级耗时分解（step_breakdown）")
    lines.append("")
    lines.append("- 生成时间: %s" % result["generated_at"])
    lines.append("- trace 数: %d（rank: %s）" % (len(ranks), ", ".join(str(r["rank"]) for r in ranks)))
    lines.append("- 跨 rank 共同步数: %d" % cross.get("n_steps_common", 0))
    lines.append("- 步的界定: 每步重复出现的 verify CUDA graph 为界；"
                 "一步 = [第 k 个 verify 图开始, 第 k+1 个 verify 图开始)，最后一个不完整的步丢弃")
    lines.append("- 通信判定: 名字含 nccl -> NCCL；含 roce/b12x 且含 allreduce/all_reduce/oneshot -> "
                 "RoCE 单发 all-reduce；含 roce/b12x 且含 gather -> RoCE all-gather；其余通信关键字 -> 其他通信")
    lines.append("- 等待估计: comm_wait_ms = 最慢 rank 通信时长 - 本 rank 通信时长（>= 0）；"
                 "comm_delta_ms = 本 rank 通信时长 - 最慢 rank 通信时长（<= 0）")
    lines.append("")

    lines.append("## 每 rank 汇总（逐步均值）")
    lines.append("")
    lines.append("| rank | 步数 | 墙钟 ms | GPU 忙 ms | GPU 空闲 ms | 通信 ms | 暴露通信 ms | 计算 ms | 通信 kernel 数 | 方法 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in ranks:
        s = r["summary"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["rank"], s["n_steps"], s["wall_ms_mean"], s["busy_ms_mean"], s["idle_ms_mean"],
            s["comm_ms_mean"], s.get("comm_exposed_ms_mean"), s["compute_ms_mean"], s["n_comm_kernels_mean"],
            r["method"]))
    lines.append("")

    lines.append("## 跨 rank 对照（逐步）")
    lines.append("")
    rank_ids = [str(r["rank"]) for r in ranks]
    header = ("| 步 | " + " | ".join("rank%s 计算 ms" % rid for rid in rank_ids) +
              " | 最慢 rank | " + " | ".join("rank%s 通信等待 ms" % rid for rid in rank_ids) + " |")
    lines.append(header)
    lines.append("|" + "---|" * (2 + 2 * len(rank_ids)))
    for row in cross.get("per_step", [])[:20]:
        cells = [str(row["i"])]
        cells += [str(row["ranks"][rid]["compute_ms"]) for rid in rank_ids]
        cells.append(str(row["slow_rank"]))
        cells += [str(row["comm_wait_ms"][rid]) for rid in rank_ids]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    for r in ranks:
        lines.append("## rank %s 明细（%s）" % (r["rank"], r["trace"]))
        lines.append("")
        lines.append("| 步 | 墙钟 ms | 忙 ms | 空闲 ms | 通信 ms | 计算 ms | MoE | GEMM | 注意力 | 草稿图 | 其他计算 | 通信 kernel |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for step in r["steps"][:20]:
            c = step["cats_ms"]
            lines.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d |" % (
                step["i"], step["wall_ms"], step["busy_ms"], step["idle_ms"], step["comm_ms"],
                step["compute_ms"], c["moe"], c["gemm"], c["attention"], c["draft"],
                c["compute_other"], step["n_comm_kernels"]))
        if len(r["steps"]) > 20:
            lines.append("| ... | 共 %d 步 | | | | | | | | | | |" % len(r["steps"]))
        lines.append("")
        lines.append("### rank %s 通信 kernel 命中（top20，按总时长）" % r["rank"])
        lines.append("")
        if r["matched_comm_kernels"]:
            lines.append("| kernel | 归类 | 调用次数 | 每步次数 | 单次均值 µs | 总时长 ms |")
            lines.append("|---|---|---|---|---|---|")
            for rec in r["matched_comm_kernels"]:
                lines.append("| %s | %s | %d | %s | %s | %s |" % (
                    rec["name"], rec["bucket"], rec["count"], rec.get("calls_per_step"), rec.get("mean_us"),
                    rec["total_ms"]))
        else:
            lines.append("(没有命中通信关键字的 kernel)")
        lines.append("")

    lines.append("## 合计")
    lines.append("")
    lines.append("- wall_ms=%.3f comm_ms=%.3f idle_ms=%.3f（所有 rank、所有共同步合计）"
                 % (result["totals"]["wall_ms"], result["totals"]["comm_ms"], result["totals"]["idle_ms"]))
    return "\n".join(lines) + "\n"


def build_result(rank_data: Sequence[Dict[str, Any]], files: Sequence[str]) -> Dict[str, Any]:
    cross = cross_rank(rank_data)
    n_common = cross.get("n_steps_common", 0)
    wall = comm = idle = busy = 0.0
    for r in rank_data:
        for step in r["steps"][:n_common]:
            wall += step["wall_ms"]
            busy += step["busy_ms"]
            comm += step["comm_ms"]
            idle += step["idle_ms"]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "yee.tools.step_breakdown",
        "files": list(files),
        "ranks": list(rank_data),
        "cross_rank": cross,
        "totals": {
            "n_ranks": len(rank_data),
            "n_steps": n_common,
            "wall_ms": round(wall, 4),
            "busy_ms": round(busy, 4),
            "comm_ms": round(comm, 4),
            "idle_ms": round(idle, 4),
        },
    }


# ---------------------------------------------------------------- CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yee.tools.step_breakdown",
        description="分析 torch profiler chrome trace 的解码步级耗时分解（本地、只读）",
    )
    parser.add_argument("traces", nargs="+", metavar="TRACE",
                        help="trace 文件或目录（*.json / *.json.gz）")
    parser.add_argument("--out", required=True, help="输出目录（写 breakdown.json 与 breakdown.md）")
    parser.add_argument("--rank", type=int, action="append", default=None,
                        help="按命令行里 trace 的顺序指定 rank 号（个数必须与文件数一致）")
    args = parser.parse_args(argv)

    files: List[str] = []
    for path in args.traces:
        if not os.path.exists(path):
            sys.stderr.write("ERROR: 路径不存在: %s\n" % path)
            return 2
        files.extend(collect_traces(path))
    if not files:
        sys.stderr.write("ERROR: 没找到 trace 文件（需要 .json 或 .json.gz）\n")
        return 3
    if args.rank is not None and len(args.rank) != len(files):
        sys.stderr.write("ERROR: --rank 给了 %d 个，trace 文件有 %d 个\n" % (len(args.rank), len(files)))
        return 2
    ranks = assign_ranks(files, args.rank)

    rank_data: List[Dict[str, Any]] = []
    seen = set()
    for path, rank in zip(files, ranks):
        try:
            data = analyze_rank(path, rank)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write("ERROR: %s: %r\n" % (path, exc))
            continue
        if data["rank"] in seen:
            sys.stderr.write("WARN: rank %s 重复，忽略 %s\n" % (data["rank"], path))
            continue
        seen.add(data["rank"])
        rank_data.append(data)
    if not rank_data:
        sys.stderr.write("ERROR: 没有可分析的 trace\n")
        return 3
    rank_data.sort(key=lambda r: (r["rank"] is None, r["rank"]))

    result = build_result(rank_data, files)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "breakdown.json")
    md_path = os.path.join(out_dir, "breakdown.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(result))

    totals = result["totals"]
    sys.stdout.write("breakdown.json -> %s\n" % json_path)
    sys.stdout.write("breakdown.md   -> %s\n" % md_path)
    sys.stdout.write("STEP_BREAKDOWN: ranks=%d steps=%d wall_ms=%.3f comm_ms=%.3f idle_ms=%.3f\n" % (
        totals["n_ranks"], totals["n_steps"], totals["wall_ms"], totals["comm_ms"], totals["idle_ms"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
