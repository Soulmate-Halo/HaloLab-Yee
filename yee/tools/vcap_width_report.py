"""vcap 选宽度器影子模式统计报告。

输入
----
``vcap_stats.jsonl``：每行一个 JSON 对象（一个 10 秒统计窗口）。只处理含
``width`` 键的行；旧格式（没有 ``width`` 键）跳过；不是 JSON 的坏行跳过并计数。

``width`` 结构::

    {"W0": 8, "widths": [4, 8, 13], "steps": 0, "timed": 0, "untimed": 0,
     "missing": 0, "switches": 0, "switches_le0": 0, "rho": [0.5, 0.5],
     "us_mean": 12.3, "t_w0": {"1": 45.1},
     "by_bs": {"1": {"n": 步数, "act": [tok, ms], "sel": [tok, ms],
                     "sel_lo": [tok, ms], "le0": [tok, ms], "oracle": [tok, ms],
                     "static": {"4": [tok, ms]}, "pick": {"4": 0},
                     "pick_le0": {"4": 0}, "pred": [pred_tok, real_tok]}}}

用法
----
::

    python -m yee.tools.vcap_width_report STATS.jsonl [更多文件 ...] \
        [--tag T] [--since TS] [--until TS] [--threshold 0.03] [--json]

退出码：0 正常；2 输入文件不存在；3 过滤后没有可用的统计窗口。

只用标准库。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 参与「相对 act 提速」计算的方案
SCHEMES = ("sel", "sel_lo", "le0", "oracle")
# by_bs 里成对出现（tok, ms）的字段
PAIR_KEYS = ("act", "sel", "sel_lo", "le0", "oracle")


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------

def _new_bs() -> Dict[str, Any]:
    """空的 bs 档累加器。"""
    return {
        "n": 0,
        "act": [0.0, 0.0],
        "sel": [0.0, 0.0],
        "sel_lo": [0.0, 0.0],
        "le0": [0.0, 0.0],
        "oracle": [0.0, 0.0],
        "static": {},
        "pick": {},
        "pick_le0": {},
        "pred": [0.0, 0.0],
    }


def _pair(value: Any) -> List[float]:
    """把 ``[tok, ms]`` 读成两个浮点数；格式不对就当 0。"""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return [0.0, 0.0]
    return [0.0, 0.0]


def _add_pair(acc: List[float], value: Any) -> None:
    pair = _pair(value)
    acc[0] += pair[0]
    acc[1] += pair[1]


def _add_counts(acc: Dict[str, int], value: Any) -> None:
    """累加 ``{"4": 3}`` 这类计数字典。"""
    if not isinstance(value, dict):
        return
    for key, val in value.items():
        try:
            acc[str(key)] = acc.get(str(key), 0) + int(val)
        except (TypeError, ValueError):
            continue


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _bs_sort(key: str) -> Tuple[int, str]:
    """bs 档按数值升序，非数字的排后面按字符串。"""
    try:
        return (0, "%020d" % int(key))
    except (TypeError, ValueError):
        return (1, key)


def _disp_width(text: str) -> int:
    """显示宽度：东亚宽字符算 2 列。"""
    total = 0
    for ch in text:
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def _cell(text: str, width: int, right: bool = False) -> str:
    pad = max(0, width - _disp_width(text))
    return (" " * pad + text) if right else (text + " " * pad)


def _fmt_gain(gain: Optional[float]) -> str:
    """提速百分比，保留 1 位小数；算不出来（除零）记为空。"""
    if gain is None:
        return "-"
    return "%+.1f%%" % (gain * 100.0)


def _fmt_num(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "-"
    return ("%." + str(digits) + "f") % value


# --------------------------------------------------------------------------
# 读取与过滤
# --------------------------------------------------------------------------

def load_windows(paths: Sequence[str]) -> Tuple[List[Dict[str, Any]], int]:
    """读所有 jsonl，返回 (含 width 的窗口列表, 坏行数)。文件不存在抛 FileNotFoundError。"""
    windows: List[Dict[str, Any]] = []
    bad = 0
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                if not isinstance(obj, dict):
                    bad += 1
                    continue
                if "width" not in obj:
                    continue
                windows.append(obj)
    return windows, bad


def filter_windows(
    windows: Sequence[Dict[str, Any]],
    tags: Optional[Sequence[str]],
    since: Optional[float],
    until: Optional[float],
) -> List[Dict[str, Any]]:
    """按 tag（取并集）、时间区间过滤窗口。"""
    wanted = set(tags) if tags else None
    out: List[Dict[str, Any]] = []
    for win in windows:
        if wanted is not None:
            if str(win.get("tag", "")) not in wanted:
                continue
        if since is not None or until is not None:
            stamp = win.get("t")
            if not isinstance(stamp, (int, float)) or isinstance(stamp, bool):
                continue
            if since is not None and float(stamp) < since:
                continue
            if until is not None and float(stamp) > until:
                continue
        out.append(win)
    return out


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------

def _merge_bs(acc: Dict[str, Any], data: Dict[str, Any]) -> None:
    acc["n"] += _as_int(data.get("n", 0))
    for key in PAIR_KEYS:
        _add_pair(acc[key], data.get(key))
    static = data.get("static")
    if isinstance(static, dict):
        for width, val in static.items():
            slot = acc["static"].setdefault(str(width), [0.0, 0.0])
            _add_pair(slot, val)
    _add_counts(acc["pick"], data.get("pick"))
    _add_counts(acc["pick_le0"], data.get("pick_le0"))
    _add_pair(acc["pred"], data.get("pred"))


def _sum_bs(accs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = _new_bs()
    for acc in accs:
        total["n"] += acc["n"]
        for key in PAIR_KEYS:
            _add_pair(total[key], acc[key])
        _add_pair(total["pred"], acc["pred"])
        for width, slot in acc["static"].items():
            dest = total["static"].setdefault(width, [0.0, 0.0])
            _add_pair(dest, slot)
        for width, cnt in acc["pick"].items():
            total["pick"][width] = total["pick"].get(width, 0) + cnt
        for width, cnt in acc["pick_le0"].items():
            total["pick_le0"][width] = total["pick_le0"].get(width, 0) + cnt
    return total


def aggregate(windows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把窗口按 bs 档逐项求和，再算总计。"""
    by_bs: Dict[str, Dict[str, Any]] = {}
    widths = set()
    steps = 0
    switches = 0
    switches_le0 = 0

    for win in windows:
        block = win.get("width")
        if not isinstance(block, dict):
            continue
        for raw in block.get("widths") or []:
            widths.add(_as_int(raw))
        steps += _as_int(block.get("steps", 0))
        switches += _as_int(block.get("switches", 0))
        switches_le0 += _as_int(block.get("switches_le0", 0))
        bs_map = block.get("by_bs")
        if not isinstance(bs_map, dict):
            continue
        for bs, data in bs_map.items():
            if not isinstance(data, dict):
                continue
            acc = by_bs.setdefault(str(bs), _new_bs())
            _merge_bs(acc, data)
            static = data.get("static")
            if isinstance(static, dict):
                for raw in static:
                    widths.add(_as_int(raw))

    order = sorted(by_bs.keys(), key=_bs_sort)
    total = _sum_bs([by_bs[bs] for bs in order])

    tags: Dict[str, int] = {}
    for win in windows:
        tag = str(win.get("tag", ""))
        tags[tag] = tags.get(tag, 0) + 1

    return {
        "by_bs": by_bs,
        "bs_order": order,
        "total": total,
        "widths": sorted(widths),
        "steps": steps,
        "switches": switches,
        "switches_le0": switches_le0,
        "windows": len(windows),
        "tags": tags,
    }


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------

def throughput(pair: Sequence[float]) -> Optional[float]:
    """tok / ms * 1000，单位 tok/s。"""
    tok, ms = pair[0], pair[1]
    if ms <= 0:
        return None
    return tok / ms * 1000.0


def gain(act: Sequence[float], scheme: Sequence[float]) -> Optional[float]:
    """(方案 tok/ms) / (act tok/ms) - 1；分母为 0 或 act 无吞吐时返回 None。"""
    if act[1] <= 0 or scheme[1] <= 0 or act[0] <= 0:
        return None
    try:
        return (scheme[0] / scheme[1]) / (act[0] / act[1]) - 1.0
    except ZeroDivisionError:
        return None


def _scheme_gain(acc: Dict[str, Any], scheme: str) -> Optional[float]:
    return gain(acc["act"], acc[scheme])


def _static_gain(acc: Dict[str, Any], width: int) -> Optional[float]:
    return gain(acc["act"], acc["static"].get(str(width), [0.0, 0.0]))


def _acc_gain_block(acc: Dict[str, Any], widths: Sequence[int]) -> Dict[str, Any]:
    block = {scheme: _scheme_gain(acc, scheme) for scheme in SCHEMES}
    block["static"] = {str(w): _static_gain(acc, w) for w in widths}
    return block


def _pick_share(counts: Dict[str, int], widths: Sequence[int]) -> List[str]:
    total = sum(counts.values())
    parts = []
    for width in widths:
        cnt = counts.get(str(width), 0)
        if total > 0:
            parts.append("W%d=%.1f%%" % (width, 100.0 * cnt / total))
        else:
            parts.append("W%d=-" % width)
    return parts


def _verdict(value: Optional[float], threshold: float) -> str:
    return "GO" if (value is not None and value >= threshold) else "STOP"


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------

def _render_table(rep: Dict[str, Any]) -> str:
    widths = rep["widths"]
    header = ["范围", "n", "act tok/s", "sel%", "sel_lo%", "le0%", "oracle%"]
    header += ["static%d%%" % w for w in widths]
    rows: List[List[str]] = []
    items: List[Tuple[str, Dict[str, Any]]] = [("总计", rep["total"])]
    items += [("bs=%s" % bs, rep["by_bs"][bs]) for bs in rep["bs_order"]]
    for label, acc in items:
        row = [label, str(acc["n"]), _fmt_num(throughput(acc["act"]))]
        for scheme in SCHEMES:
            row.append(_fmt_gain(_scheme_gain(acc, scheme)))
        for width in widths:
            row.append(_fmt_gain(_static_gain(acc, width)))
        rows.append(row)

    cols = len(header)
    col_w = []
    for c in range(cols):
        widest = _disp_width(header[c])
        for row in rows:
            widest = max(widest, _disp_width(row[c]))
        col_w.append(widest)
    lines = ["  ".join(_cell(header[c], col_w[c], right=(c != 0)) for c in range(cols)).rstrip(),
             "  ".join("-" * col_w[c] for c in range(cols))]
    for row in rows:
        lines.append("  ".join(_cell(row[c], col_w[c], right=(c != 0)) for c in range(cols)).rstrip())
    return "\n".join(lines)


def render_text(rep: Dict[str, Any], threshold: float) -> str:
    total = rep["total"]
    widths = rep["widths"]
    lines: List[str] = []
    lines.append("vcap 选宽度器影子模式报告")
    lines.append("阈值 threshold = %.3f" % threshold)
    lines.append("")
    lines.append("总表（act tok/s 为现网吞吐；百分比为相对 act 的提速）")
    lines.append(_render_table(rep))
    lines.append("")
    lines.append("选中分布（按选中步数占比）")
    lines.append("  sel  " + "  ".join(_pick_share(total["pick"], widths)))
    lines.append("  le0  " + "  ".join(_pick_share(total["pick_le0"], widths)))
    lines.append("")
    steps = rep["steps"]
    sw_sel = (1000.0 * rep["switches"] / steps) if steps > 0 else None
    sw_le0 = (1000.0 * rep["switches_le0"] / steps) if steps > 0 else None
    lines.append("每千步切换次数: sel=%s le0=%s (switches=%d switches_le0=%d steps=%d)"
                 % (_fmt_num(sw_sel), _fmt_num(sw_le0), rep["switches"], rep["switches_le0"], steps))
    pred_tok, real_tok = total["pred"][0], total["pred"][1]
    ratio = (pred_tok / real_tok) if real_tok > 0 else None
    lines.append("预测偏差: pred_tok=%s real_tok=%s ratio=%s"
                 % (_fmt_num(pred_tok), _fmt_num(real_tok),
                    ("%.4f" % ratio) if ratio is not None else "-"))
    lines.append("窗口数: %d  坏行数: %d" % (rep["windows"], rep["bad_lines"]))
    tags = rep["tags"]
    if tags:
        lines.append("各 tag 窗口数: " + "  ".join("%s=%d" % (k, tags[k]) for k in sorted(tags)))
    else:
        lines.append("各 tag 窗口数: -")
    lines.append("")
    g_sel = _scheme_gain(total, "sel")
    g_le0 = _scheme_gain(total, "le0")
    g_lo = _scheme_gain(total, "sel_lo")
    lines.append("VERDICT: %s" % _verdict(g_sel, threshold))
    lines.append("le0: %s" % _verdict(g_le0, threshold))
    lines.append("sel_lo: %s" % _verdict(g_lo, threshold))
    return "\n".join(lines)


def build_json(rep: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    total = rep["total"]
    widths = rep["widths"]
    gain_block = _acc_gain_block(total, widths)
    by_bs: Dict[str, Any] = {}
    for bs in rep["bs_order"]:
        acc = rep["by_bs"][bs]
        by_bs[bs] = {
            "n": acc["n"],
            "act_tok_s": throughput(acc["act"]),
            "gain": _acc_gain_block(acc, widths),
        }
    steps = rep["steps"]
    pred_tok, real_tok = total["pred"][0], total["pred"][1]
    return {
        "verdict": _verdict(gain_block["sel"], threshold),
        "verdict_le0": _verdict(gain_block["le0"], threshold),
        "verdict_sel_lo": _verdict(gain_block["sel_lo"], threshold),
        "threshold": threshold,
        "gain": gain_block,
        "by_bs": by_bs,
        "total": {
            "n": total["n"],
            "act_tok_s": throughput(total["act"]),
            "gain": gain_block,
        },
        "widths": widths,
        "windows": rep["windows"],
        "bad_lines": rep["bad_lines"],
        "tags": rep["tags"],
        "pick": total["pick"],
        "pick_le0": total["pick_le0"],
        "steps": steps,
        "switches": rep["switches"],
        "switches_le0": rep["switches_le0"],
        "switches_per_1k": {
            "sel": (1000.0 * rep["switches"] / steps) if steps > 0 else None,
            "le0": (1000.0 * rep["switches_le0"] / steps) if steps > 0 else None,
        },
        "pred": {
            "pred_tok": pred_tok,
            "real_tok": real_tok,
            "ratio": (pred_tok / real_tok) if real_tok > 0 else None,
        },
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vcap_width_report",
        description="汇总 vcap 选宽度器影子模式写出的 vcap_stats.jsonl",
    )
    parser.add_argument("files", nargs="+", help="vcap_stats.jsonl 路径，可给多个")
    parser.add_argument("--tag", action="append", default=None,
                        help="只统计该 tag 的窗口，可重复给（取并集）")
    parser.add_argument("--since", type=float, default=None, help="只统计 t >= 该时间戳的窗口")
    parser.add_argument("--until", type=float, default=None, help="只统计 t <= 该时间戳的窗口")
    parser.add_argument("--threshold", type=float, default=0.03, help="判定阈值（默认 0.03）")
    parser.add_argument("--json", action="store_true", dest="as_json", help="输出 JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        windows, bad = load_windows(args.files)
    except FileNotFoundError as exc:
        sys.stderr.write("文件不存在: %s\n" % exc)
        return 2
    selected = filter_windows(windows, args.tag, args.since, args.until)
    if not selected:
        sys.stderr.write(
            "过滤后没有可用的统计窗口（读入 %d 个含 width 的窗口，%d 行坏数据）\n"
            % (len(windows), bad))
        return 3
    rep = aggregate(selected)
    rep["bad_lines"] = bad
    if args.as_json:
        sys.stdout.write(json.dumps(build_json(rep, args.threshold), ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(render_text(rep, args.threshold) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
