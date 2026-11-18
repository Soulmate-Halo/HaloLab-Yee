# -*- coding: utf-8 -*-
"""yee.tools.step_breakdown 的自测：用代码合成的小 trace 验证切步、分桶、并集、慢 rank 与等待估计。

合成 trace 的形状（4 个 rank，每步）：
  每步一个 verify CUDA graph（6 个 kernel）和一个草稿图（2 个 kernel）；
  步边界 = 下一个 verify 图的首个 kernel 开始时刻；最后一个不完整的步丢弃。
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parent.parent
MOD_PATH = PKG_ROOT / "yee" / "tools" / "step_breakdown.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("step_breakdown", MOD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()

STEP_US = 10000.0          # 每步 10 ms
N_STEPS = 3                # 合成 3 个完整步（需要 4 个 verify 图来闭合最后一个窗口）

# 每 rank 的参数：(moe kernel 时长 us, nccl kernel 时长 us)
RANK_PARAMS = {
    0: (1000.0, 400.0),
    1: (1000.0, 400.0),
    2: (2500.0, 400.0),     # rank 2 计算最慢
    3: (1000.0, 200.0),     # rank 3 通信最短
}
SLOW_RANK = 2


def _kernel(name, ts, dur, corr):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur,
            "pid": 1000 + corr, "tid": 7, "args": {"correlation": corr}}


def _launch(ts, corr):
    return {"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": ts, "dur": 5.0,
            "pid": 1000 + corr, "tid": 7, "args": {"correlation": corr}}


def make_trace(rank: int, n_steps: int = N_STEPS) -> dict:
    """合成一个 rank 的 chrome trace。"""
    moe_dur, nccl_dur = RANK_PARAMS[rank]
    events = []
    corr = 1
    for i in range(n_steps + 1):
        base = i * STEP_US
        # ---- verify 图（6 个 kernel）
        events.append(_launch(base, corr))
        events.append(_kernel("fused_moe_grouped_gemm", base, moe_dur, corr))
        events.append(_kernel("cutlass_gemm_f16_128x128", base + 1000, 1500, corr))
        events.append(_kernel("flash_mla_decode_kernel", base + 2500, 800, corr))
        events.append(_kernel("ncclKernel_AllReduce_sum", base + 3300, nccl_dur, corr))
        events.append(_kernel("b12x_roce_oneshot_allreduce", base + 3700, 300, corr))
        events.append(_kernel("b12x_roce_allgather_draft_logits", base + 3800, 200, corr))
        corr += 1
        if i >= n_steps:
            break
        # ---- 草稿图（2 个 kernel，在 verify 之后、下一 verify 之前）
        events.append(_launch(base + 5000, corr))
        events.append(_kernel("dspark_draft_mlp_proj", base + 5000, 500, corr))
        events.append(_kernel("dspark_draft_sample", base + 5600, 300, corr))
        corr += 1
    return {"traceEvents": events, "distributedInfo": {"rank": rank}}


def write_traces(tmp_path: Path, n_steps: int = N_STEPS, ranks=(0, 1, 2, 3), gz=False):
    out = []
    for rank in ranks:
        data = make_trace(rank, n_steps)
        if gz:
            path = tmp_path / ("rank_TP-%d.trace.json.gz" % rank)
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(data, fh)
        else:
            path = tmp_path / ("rank_TP-%d.trace.json" % rank)
            path.write_text(json.dumps(data), encoding="utf-8")
        out.append(path)
    return out


# --------------------------------------------------------------------------
# 切步
# --------------------------------------------------------------------------

def test_cut_steps_uses_verify_graph_and_drops_last_incomplete(tmp_path):
    path = write_traces(tmp_path, ranks=(0,))[0]
    events = mod.load_events(str(path))
    windows, draft_ids, gpu, info = mod.cut_steps(events)
    # N_STEPS+1 个 verify 图 -> N_STEPS 个完整步
    assert len(windows) == N_STEPS
    assert info["method"] == "graph_correlation"
    assert info["n_verify_graphs"] == N_STEPS + 1
    assert info["n_draft_graphs"] == N_STEPS
    # 步边界就是 verify 图的首个 kernel 开始时刻
    assert windows[0][0] == pytest.approx(0.0)
    assert windows[1][0] == pytest.approx(STEP_US)
    assert windows[-1][1] == pytest.approx(N_STEPS * STEP_US)
    # 草稿图的 kernel 被单独标出来
    assert len(draft_ids) == 2 * N_STEPS


def test_cut_steps_falls_back_to_repeated_kernel(tmp_path):
    """没有 CUDA graph 时，退回"每步重复出现的固定 kernel"当步边界。"""
    events = []
    for i in range(4):
        base = i * 5000.0
        events.append({"ph": "X", "cat": "kernel", "name": "steady_marker_kernel",
                       "ts": base, "dur": 100.0, "args": {}})
        events.append({"ph": "X", "cat": "kernel", "name": "work_kernel",
                       "ts": base + 200.0, "dur": 900.0, "args": {}})
    path = tmp_path / "nograph.json"
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    windows, _draft, _gpu, info = mod.cut_steps(mod.load_events(str(path)))
    assert info["method"] == "repeated_kernel"
    assert len(windows) == 3
    assert windows[0][0] == pytest.approx(0.0)
    assert windows[1][0] == pytest.approx(5000.0)


# --------------------------------------------------------------------------
# 分桶
# --------------------------------------------------------------------------

def test_comm_bucket_by_kernel_name():
    assert mod.classify_comm("ncclKernel_AllReduce_sum") == "nccl"
    assert mod.classify_comm("b12x_roce_oneshot_allreduce") == "roce_allreduce"
    # 回归（r713 实采）：b12x 的 MoE 与稠密 GEMM kernel 名也带 b12x，不能算通信
    assert mod.classify_comm("kernel_cutlass_kernel_b12x_nextmoe_sharedkernelsw4a8_phase1W4A8MaterializedPhase1Kernel"
                             "_object_at__tensorptri32gmemalign16o1") is None
    assert mod.classify_comm("kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xDenseGemm"
                             "Kernel_object_at__CopyAtom") is None
    assert mod.classify_comm("kernel_cutlass_kernel_b12xcommroce_oneshot_cute_RoceOneshotLaunch_object_at__ptri32"
                             ) == "roce_allreduce"
    assert mod.classify_comm("kernel_cutlass_kernel_b12xcommroce_allgather_cute_RoceAllGatherLaunch_object_at__ptr"
                             ) == "roce_allgather"
    assert mod.classify_comm("b12x_roce_all_gather") == "roce_allgather"
    assert mod.classify_comm("roce_allgather_draft_logits") == "roce_allgather"
    assert mod.classify_comm("ncclDevKernel_AllGather") == "nccl"
    assert mod.classify_comm("some_broadcast_kernel") == "comm_other"
    assert mod.classify_comm("fused_moe_grouped_gemm") is None


def test_compute_bucket_by_kernel_name():
    assert mod.classify_compute("fused_moe_grouped_gemm") == "moe"
    assert mod.classify_compute("cutlass_gemm_f16_128x128") == "gemm"
    assert mod.classify_compute("flash_mla_decode_kernel") == "attention"
    assert mod.classify_compute("elementwise_add_kernel") == "compute_other"


def test_step_buckets_split_comm_and_compute(tmp_path):
    path = write_traces(tmp_path, ranks=(0,))[0]
    data = mod.analyze_rank(str(path), 0)
    assert data["summary"]["n_steps"] == N_STEPS
    step = data["steps"][0]
    cats = step["cats_ms"]
    # rank 0: moe 1000us, gemm 1500us, attn 800us, draft 500+300us, nccl 400us,
    #         roce allreduce 300us, roce allgather 200us
    assert cats["moe"] == pytest.approx(1.0)
    assert cats["gemm"] == pytest.approx(1.5)
    assert cats["attention"] == pytest.approx(0.8)
    assert cats["draft"] == pytest.approx(0.8)
    assert cats["nccl"] == pytest.approx(0.4)
    assert cats["roce_allreduce"] == pytest.approx(0.3)
    assert cats["roce_allgather"] == pytest.approx(0.2)
    assert cats["compute_other"] == pytest.approx(0.0)
    assert step["comm_ms"] == pytest.approx(0.9)
    assert step["compute_ms"] == pytest.approx(4.1)
    assert step["n_comm_kernels"] == 3


def test_comm_kernel_top_listing(tmp_path):
    path = write_traces(tmp_path, ranks=(0,))[0]
    data = mod.analyze_rank(str(path), 0)
    names = {rec["name"]: rec for rec in data["matched_comm_kernels"]}
    assert "ncclKernel_AllReduce_sum" in names
    assert "b12x_roce_oneshot_allreduce" in names
    assert "b12x_roce_allgather_draft_logits" in names
    assert names["b12x_roce_oneshot_allreduce"]["bucket"] == "roce_allreduce"
    assert names["b12x_roce_allgather_draft_logits"]["bucket"] == "roce_allgather"
    # 每个通信 kernel 在每步出现一次
    assert names["ncclKernel_AllReduce_sum"]["count"] == N_STEPS


# --------------------------------------------------------------------------
# 并集 / 空闲
# --------------------------------------------------------------------------

def test_busy_uses_union_not_sum(tmp_path):
    path = write_traces(tmp_path, ranks=(0,))[0]
    data = mod.analyze_rank(str(path), 0)
    step = data["steps"][0]
    # kernel 时长之和 = 5.0 ms，但两个通信 kernel 有重叠，并集 = 4.8 ms
    assert step["wall_ms"] == pytest.approx(10.0)
    assert step["busy_ms"] == pytest.approx(4.8)
    assert step["idle_ms"] == pytest.approx(10.0 - 4.8)
    assert step["busy_ms"] < 5.0
    assert mod._union([(0.0, 10.0), (5.0, 15.0)]) == pytest.approx(15.0)
    assert mod._union([]) == 0.0


# --------------------------------------------------------------------------
# 跨 rank：慢 rank 与等待估计
# --------------------------------------------------------------------------

def test_cross_rank_slow_rank_and_comm_wait(tmp_path):
    paths = write_traces(tmp_path)
    rank_data = [mod.analyze_rank(str(p), mod.detect_rank(str(p))) for p in paths]
    cross = mod.cross_rank(rank_data)
    assert cross["n_steps_common"] == N_STEPS
    row = cross["per_step"][0]
    # rank 2 的非通信计算最长（5.6 ms vs 4.1 ms）
    assert row["slow_rank"] == SLOW_RANK
    assert row["ranks"]["2"]["compute_ms"] == pytest.approx(5.6)
    assert row["ranks"]["0"]["compute_ms"] == pytest.approx(4.1)
    # 最慢 rank 通信 0.9 ms；rank 3 通信 0.7 ms
    assert row["slow_comm_ms"] == pytest.approx(0.9)
    assert row["comm_wait_ms"]["3"] == pytest.approx(0.2)
    assert row["comm_wait_ms"]["0"] == pytest.approx(0.0)
    assert row["comm_delta_ms"]["3"] == pytest.approx(-0.2)
    assert row["comm_delta_ms"]["2"] == pytest.approx(0.0)
    assert cross["summary"]["slow_rank_counts"] == {str(SLOW_RANK): N_STEPS}
    assert cross["summary"]["comm_wait_ms_mean"]["3"] == pytest.approx(0.2)


# --------------------------------------------------------------------------
# 输入收集 / rank 识别
# --------------------------------------------------------------------------

def test_detect_rank_from_filename():
    assert mod.detect_rank("/tmp/xxx_TP-0.trace.json") == 0
    assert mod.detect_rank("/tmp/xxx_TP-3.trace.json.gz") == 3
    assert mod.detect_rank("/tmp/rank_2.json") is None


def test_collect_traces_from_directory(tmp_path):
    paths = write_traces(tmp_path)
    found = mod.collect_traces(str(tmp_path))
    assert sorted(found) == sorted(str(p) for p in paths)


def test_gzip_trace_supported(tmp_path):
    paths = write_traces(tmp_path, gz=True)
    data = mod.analyze_rank(str(paths[0]), 0)
    assert data["summary"]["n_steps"] == N_STEPS
    assert data["steps"][0]["comm_ms"] == pytest.approx(0.9)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _run_cli(*args, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "yee.tools.step_breakdown"] + [str(a) for a in args],
        capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(cwd) if cwd else None,
    )


def test_cli_writes_outputs_and_summary_line(tmp_path):
    write_traces(tmp_path)
    out_dir = tmp_path / "out"
    proc = _run_cli(tmp_path, "--out", out_dir)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    m = re.match(r"^STEP_BREAKDOWN: ranks=(\d+) steps=(\d+) wall_ms=([\d.]+) "
                 r"comm_ms=([\d.]+) idle_ms=([\d.]+)$", lines[-1])
    assert m, lines[-1]
    assert int(m.group(1)) == 4
    assert int(m.group(2)) == N_STEPS
    # 4 rank x 3 步：墙钟 4*3*10=120ms，通信 4 rank 合计 (0.9+0.9+0.9+0.7)*3=10.2ms
    assert float(m.group(3)) == pytest.approx(120.0)
    assert float(m.group(4)) == pytest.approx(10.2)
    assert float(m.group(5)) > 0
    payload = json.loads((out_dir / "breakdown.json").read_text(encoding="utf-8"))
    assert payload["totals"]["n_ranks"] == 4
    assert payload["totals"]["n_steps"] == N_STEPS
    assert payload["cross_rank"]["summary"]["slow_rank_counts"] == {str(SLOW_RANK): N_STEPS}
    md = (out_dir / "breakdown.md").read_text(encoding="utf-8")
    assert "跨 rank 对照" in md
    assert "通信 kernel 命中" in md
    assert "b12x_roce_oneshot_allreduce" in md


def test_cli_missing_path_exit_code_2(tmp_path):
    proc = _run_cli(tmp_path / "nope.json", "--out", tmp_path / "out")
    assert proc.returncode == 2


def test_cli_empty_dir_exit_code_3(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = _run_cli(empty, "--out", tmp_path / "out")
    assert proc.returncode == 3
