"""vcap_width_report 的自测：合成 vcap_stats.jsonl 验证汇总、过滤、判定与退出码。"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

MOD_PATH = Path(__file__).resolve().parent.parent / "yee" / "tools" / "vcap_width_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("vcap_width_report", MOD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


# --------------------------------------------------------------------------
# 样例数据：4 个窗口 + 1 行坏 JSON + 1 行旧格式（没有 width）
# --------------------------------------------------------------------------

def _win1() -> dict:
    """tag=a, t=1000, bs=1/2。"""
    return {
        "t": 1000.0,
        "tag": "a",
        "mode": "legacy",
        "width": {
            "W0": 8,
            "widths": [4, 8, 13],
            "steps": 10,
            "timed": 10,
            "untimed": 0,
            "missing": 0,
            "switches": 2,
            "switches_le0": 1,
            "rho": [0.5, 0.5],
            "us_mean": 12.3,
            "t_w0": {"1": 10.0, "2": 20.0},
            "by_bs": {
                "1": {
                    "n": 6,
                    "act": [60, 10.0],
                    "sel": [66, 10.0],
                    "sel_lo": [63, 10.0],
                    "le0": [63, 10.0],
                    "oracle": [72, 10.0],
                    "static": {"4": [54, 10.0], "8": [60, 10.0], "13": [66, 10.0]},
                    "pick": {"4": 2, "8": 3, "13": 1},
                    "pick_le0": {"4": 1, "8": 5},
                    "pred": [66, 60],
                },
                "2": {
                    "n": 4,
                    "act": [80, 20.0],
                    "sel": [88, 20.0],
                    "sel_lo": [84, 20.0],
                    "le0": [84, 20.0],
                    "oracle": [96, 20.0],
                    "static": {"4": [72, 20.0], "8": [80, 20.0], "13": [88, 20.0]},
                    "pick": {"4": 1, "8": 2, "13": 1},
                    "pick_le0": {"4": 1, "8": 3},
                    "pred": [88, 80],
                },
            },
        },
    }


def _win2() -> dict:
    """tag=a, t=1010, 只有 bs=1。"""
    return {
        "t": 1010.0,
        "tag": "a",
        "mode": "legacy",
        "width": {
            "W0": 8,
            "widths": [4, 8, 13],
            "steps": 4,
            "timed": 4,
            "untimed": 0,
            "missing": 0,
            "switches": 1,
            "switches_le0": 0,
            "rho": [0.5, 0.5],
            "us_mean": 12.0,
            "t_w0": {"1": 8.0},
            "by_bs": {
                "1": {
                    "n": 4,
                    "act": [40, 8.0],
                    "sel": [44, 8.0],
                    "sel_lo": [42, 8.0],
                    "le0": [42, 8.0],
                    "oracle": [48, 8.0],
                    "static": {"4": [36, 8.0], "8": [40, 8.0], "13": [44, 8.0]},
                    "pick": {"4": 1, "8": 2, "13": 1},
                    "pick_le0": {"4": 1, "8": 3},
                    "pred": [44, 40],
                },
            },
        },
    }


def _win3() -> dict:
    """tag=b, t=1020, widths 只有 [8, 13]，bs=4。"""
    return {
        "t": 1020.0,
        "tag": "b",
        "mode": "legacy",
        "width": {
            "W0": 8,
            "widths": [8, 13],
            "steps": 2,
            "timed": 2,
            "untimed": 0,
            "missing": 0,
            "switches": 0,
            "switches_le0": 0,
            "rho": [0.4, 0.6],
            "us_mean": 11.0,
            "t_w0": {"4": 10.0},
            "by_bs": {
                "4": {
                    "n": 2,
                    "act": [40, 10.0],
                    "sel": [40, 10.0],
                    "sel_lo": [40, 10.0],
                    "le0": [40, 10.0],
                    "oracle": [40, 10.0],
                    "static": {"8": [40, 10.0], "13": [40, 10.0]},
                    "pick": {"8": 2},
                    "pick_le0": {"8": 2},
                    "pred": [40, 40],
                },
            },
        },
    }


def _win4() -> dict:
    """tag=b, t=1030, bs=1，静态宽度没有 4（缺失当 0）。"""
    return {
        "t": 1030.0,
        "tag": "b",
        "mode": "legacy",
        "width": {
            "W0": 8,
            "widths": [8, 13],
            "steps": 5,
            "timed": 5,
            "untimed": 0,
            "missing": 0,
            "switches": 0,
            "switches_le0": 0,
            "rho": [0.4, 0.6],
            "us_mean": 11.5,
            "t_w0": {"1": 10.0},
            "by_bs": {
                "1": {
                    "n": 5,
                    "act": [50, 10.0],
                    "sel": [50, 10.0],
                    "sel_lo": [50, 10.0],
                    "le0": [50, 10.0],
                    "oracle": [55, 10.0],
                    "static": {"8": [50, 10.0], "13": [50, 10.0]},
                    "pick": {"8": 5},
                    "pick_le0": {"8": 5},
                    "pred": [50, 50],
                },
            },
        },
    }


LEGACY_ROW = {"t": 1005.0, "tag": "legacy", "mode": "legacy", "steps": 3, "mean_bs": 1.0}
BAD_LINE = "{not json at all"


def write_sample(tmp_path: Path, name: str = "vcap_stats.jsonl") -> Path:
    path = tmp_path / name
    lines = []
    for win in (_win1(), _win2(), _win3(), _win4()):
        lines.append(json.dumps(win))
    lines.append(json.dumps(LEGACY_ROW))
    lines.append(BAD_LINE)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_cli(*args) -> subprocess.CompletedProcess:
    # 子进程按控制台编码输出（Windows 中文控制台是 GBK）：固定成 UTF-8 再按 UTF-8 读
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, str(MOD_PATH)] + [str(a) for a in args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )


def load_sample(tmp_path: Path):
    windows, bad = mod.load_windows([str(write_sample(tmp_path))])
    return windows, bad


# --------------------------------------------------------------------------
# 汇总求和
# --------------------------------------------------------------------------

def test_totals_are_sum_over_bs(tmp_path):
    windows, bad = load_sample(tmp_path)
    assert bad == 1
    assert len(windows) == 4
    rep = mod.aggregate(windows)
    total = rep["total"]
    assert rep["bs_order"] == ["1", "2", "4"]

    # 手算总计：act tok 60+80+40+40+50=270，ms 10+20+8+10+10=58
    assert total["act"] == [270.0, 58.0]
    assert total["sel"] == [288.0, 58.0]
    assert total["sel_lo"] == [279.0, 58.0]
    assert total["le0"] == [279.0, 58.0]
    assert total["oracle"] == [311.0, 58.0]
    assert total["n"] == 6 + 4 + 4 + 2 + 5
    # 总计 = 各档之和
    for bs in rep["bs_order"]:
        acc = rep["by_bs"][bs]
        assert total["act"][0] == pytest.approx(sum(rep["by_bs"][b]["act"][0] for b in rep["bs_order"]))
        assert total["act"][1] == pytest.approx(sum(rep["by_bs"][b]["act"][1] for b in rep["bs_order"]))
        assert acc["act"][0] >= 0
    # 分档手算
    assert rep["by_bs"]["1"]["act"] == [150.0, 28.0]
    assert rep["by_bs"]["2"]["act"] == [80.0, 20.0]
    assert rep["by_bs"]["4"]["act"] == [40.0, 10.0]
    # static 按并集求和，缺档当 0
    assert rep["by_bs"]["4"]["static"].get("4") is None
    assert total["static"]["4"] == [162.0, 38.0]
    assert total["static"]["8"] == [270.0, 58.0]
    assert total["static"]["13"] == [288.0, 58.0]
    assert rep["widths"] == [4, 8, 13]
    # 步数与切换
    assert rep["steps"] == 10 + 4 + 2 + 5
    assert rep["switches"] == 3
    assert rep["switches_le0"] == 1
    assert rep["tags"] == {"a": 2, "b": 2}
    # 选中分布
    assert total["pick"] == {"4": 4, "8": 14, "13": 3}
    assert total["pick_le0"] == {"4": 3, "8": 18}
    # 预测偏差
    assert total["pred"] == [288.0, 270.0]


def test_gain_percentages_match_manual(tmp_path):
    windows, _ = load_sample(tmp_path)
    rep = mod.aggregate(windows)
    total = rep["total"]
    # sel: (288/58) / (270/58) - 1 = 6.67%
    assert mod.gain(total["act"], total["sel"]) == pytest.approx(288.0 / 270.0 - 1.0)
    # sel_lo: 279/270 - 1 = 3.33%
    assert mod.gain(total["act"], total["sel_lo"]) == pytest.approx(279.0 / 270.0 - 1.0)
    # oracle: 311/270 - 1 = 15.19%
    assert mod.gain(total["act"], total["oracle"]) == pytest.approx(311.0 / 270.0 - 1.0)
    # static 4: (162/38)/(270/58) - 1
    assert mod.gain(total["act"], total["static"]["4"]) == pytest.approx((162.0 / 38.0) / (270.0 / 58.0) - 1.0)
    # 分档：bs=1 的 sel 是 160/150 - 1 = 6.67%
    assert mod.gain(rep["by_bs"]["1"]["act"], rep["by_bs"]["1"]["sel"]) == pytest.approx(160.0 / 150.0 - 1.0)
    # act 的 ms 为 0 时提速为空
    assert mod.gain([10.0, 0.0], [20.0, 1.0]) is None
    assert mod.throughput([10.0, 0.0]) is None


def test_zero_act_ms_gap_is_empty(tmp_path):
    path = tmp_path / "zero.jsonl"
    win = _win1()
    win["width"]["by_bs"]["1"]["act"] = [0, 0]
    path.write_text(json.dumps(win) + "\n", encoding="utf-8")
    proc = run_cli(path, "--json")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["by_bs"]["1"]["gain"]["sel"] is None


# --------------------------------------------------------------------------
# 过滤
# --------------------------------------------------------------------------

def test_tag_filter(tmp_path):
    path = write_sample(tmp_path)
    proc = run_cli(path, "--tag", "b", "--json")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["windows"] == 2
    assert payload["tags"] == {"b": 2}
    assert payload["gain"]["sel"] == pytest.approx(0.0)
    assert payload["verdict"] == "STOP"
    # tag 取并集
    proc = run_cli(path, "--tag", "a", "--tag", "b", "--json")
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["windows"] == 4


def test_since_until_filter(tmp_path):
    path = write_sample(tmp_path)
    proc = run_cli(path, "--since", "1015.0", "--json")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["windows"] == 2
    assert set(payload["tags"]) == {"b"}

    proc = run_cli(path, "--until", "1010.0", "--json")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["windows"] == 2
    assert set(payload["tags"]) == {"a"}

    proc = run_cli(path, "--since", "1010.0", "--until", "1020.0", "--json")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["windows"] == 2


# --------------------------------------------------------------------------
# 判定
# --------------------------------------------------------------------------

def test_verdict_go_and_stop(tmp_path):
    path = write_sample(tmp_path)
    # 全量：sel 提速 6.67% >= 3% -> GO
    proc = run_cli(path)
    assert proc.returncode == 0
    assert "VERDICT: GO" in proc.stdout
    assert "le0: GO" in proc.stdout
    assert "sel_lo: GO" in proc.stdout

    # 阈值提高到 20% -> STOP（sel 6.67% < 20%，oracle 14.81% 也不到，但 oracle 不参与判定）
    proc = run_cli(path, "--threshold", "0.2")
    assert proc.returncode == 0
    assert "VERDICT: STOP" in proc.stdout
    assert "le0: STOP" in proc.stdout
    assert "sel_lo: STOP" in proc.stdout

    # tag=b 时 sel 提速 0% -> STOP
    proc = run_cli(path, "--tag", "b")
    assert proc.returncode == 0
    assert "VERDICT: STOP" in proc.stdout


# --------------------------------------------------------------------------
# 文本 / JSON 输出
# --------------------------------------------------------------------------

def test_text_output_contains_sections(tmp_path):
    path = write_sample(tmp_path)
    proc = run_cli(path)
    assert proc.returncode == 0
    out = proc.stdout
    assert "总计" in out
    assert "bs=1" in out and "bs=2" in out and "bs=4" in out
    assert "每千步切换次数" in out
    assert "预测偏差" in out
    assert "窗口数: 4" in out
    assert "坏行数: 1" in out
    assert "各 tag 窗口数" in out
    assert "+6.7%" in out
    assert "VERDICT: GO" in out


def test_json_output_fields(tmp_path):
    path = write_sample(tmp_path)
    proc = run_cli(path, "--json")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "GO"
    assert payload["gain"]["sel"] == pytest.approx(288.0 / 270.0 - 1.0)
    assert payload["gain"]["sel_lo"] == pytest.approx(279.0 / 270.0 - 1.0)
    assert payload["gain"]["le0"] == pytest.approx(279.0 / 270.0 - 1.0)
    assert payload["gain"]["oracle"] == pytest.approx(311.0 / 270.0 - 1.0)
    assert payload["gain"]["static"]["4"] == pytest.approx((162.0 / 38.0) / (270.0 / 58.0) - 1.0)
    assert payload["gain"]["static"]["13"] == pytest.approx((288.0 / 58.0) / (270.0 / 58.0) - 1.0)
    assert payload["by_bs"]["1"]["n"] == 6 + 4 + 5
    assert payload["by_bs"]["1"]["act_tok_s"] == pytest.approx(150.0 / 28.0 * 1000.0)
    assert payload["by_bs"]["1"]["gain"]["sel"] == pytest.approx(160.0 / 150.0 - 1.0)
    assert payload["windows"] == 4
    assert payload["bad_lines"] == 1
    assert payload["tags"] == {"a": 2, "b": 2}
    assert payload["switches_per_1k"]["sel"] == pytest.approx(1000.0 * 3 / 21)
    assert payload["pred"]["ratio"] == pytest.approx(288.0 / 270.0)
    assert payload["total"]["n"] == 6 + 4 + 4 + 2 + 5


# --------------------------------------------------------------------------
# 退出码
# --------------------------------------------------------------------------

def test_exit_code_3_when_no_records(tmp_path):
    path = write_sample(tmp_path)
    proc = run_cli(path, "--tag", "does-not-exist")
    assert proc.returncode == 3
    assert "没有" in (proc.stdout + proc.stderr)


def test_exit_code_2_when_file_missing(tmp_path):
    proc = run_cli(tmp_path / "nope.jsonl")
    assert proc.returncode == 2
    assert "不存在" in (proc.stdout + proc.stderr)


def test_multiple_files_are_merged(tmp_path):
    first = tmp_path / "a.jsonl"
    first.write_text(json.dumps(_win1()) + "\n", encoding="utf-8")
    second = tmp_path / "b.jsonl"
    second.write_text(json.dumps(_win3()) + "\n", encoding="utf-8")
    proc = run_cli(first, second, "--json")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["windows"] == 2
    assert payload["total"]["n"] == 6 + 4 + 2
