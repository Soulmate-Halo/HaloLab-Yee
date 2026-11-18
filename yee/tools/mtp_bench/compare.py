#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare.py — 汇总多个 bench.py 输出，生成 markdown 对比表 + 一致性检查。

用法示例：
    python3 compare.py --inputs base.json gpu_k1.json --baseline base --out compare.md
"""

import argparse
import json
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="compare bench outputs")
    p.add_argument("--inputs", nargs="+", required=True, help="多个 bench 输出的 json")
    p.add_argument("--baseline", required=True, help="作为基准的 tag")
    p.add_argument("--out", required=True, help="输出 markdown 文件")
    return p.parse_args(argv)


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(v, nd=2, suffix=""):
    if v is None:
        return "n/a"
    return ("%." + str(nd) + "f%s") % (v, suffix)


def main(argv=None):
    args = parse_args(argv)
    runs = []
    for p in args.inputs:
        obj = load(p)
        runs.append(obj)

    base_obj = None
    for obj in runs:
        if obj.get("tag") == args.baseline:
            base_obj = obj
            break
    if base_obj is None:
        sys.stderr.write("[compare] baseline tag %r not found in inputs\n" % args.baseline)
        return 2

    base_total = base_obj.get("summary", {}).get("total", {})
    base_tok_s = base_total.get("agg_tok_s")

    kinds = []
    for obj in runs:
        for k in obj.get("summary", {}).get("by_kind", {}):
            if k not in kinds:
                kinds.append(k)

    lines = []
    lines.append("# bench compare (baseline = %s)" % args.baseline)
    lines.append("")

    header = ["tag", "tok/s", "倍数", "接受率", "请求数", "失败数"] + [k + " tok/s" for k in kinds]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for obj in runs:
        tot = obj.get("summary", {}).get("total", {})
        tok_s = tot.get("agg_tok_s")
        acc = tot.get("accept_rate")
        mult = (tok_s / base_tok_s) if (tok_s is not None and base_tok_s) else None
        row = [
            str(obj.get("tag")),
            fmt(tok_s),
            fmt(mult, 3, "x"),
            fmt(acc, 4),
            str(tot.get("requests", 0)),
            str(tot.get("fail", 0)),
        ]
        byk = obj.get("summary", {}).get("by_kind", {})
        for k in kinds:
            row.append(fmt(byk.get(k, {}).get("agg_tok_s")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # 一致性：按 (id, rep) 比较各 tag 与 baseline 的 content_sha16
    # 只在两边都存在的 (id, rep) 上比对；任一侧请求失败记入 fail_pair，不算不一致。
    def sha_map(obj):
        m = {}
        for r in obj.get("records", []):
            key = (r.get("id"), r.get("rep"))
            m[key] = (r.get("content_sha16"), r.get("error"))
        return m

    base_map = sha_map(base_obj)
    lines.append("## 一致性（与 baseline `%s` 按 (id, rep) 比较 content_sha16）" % args.baseline)
    lines.append("")
    lines.append("| tag | 不一致条数 | 不一致 id |")
    lines.append("| --- | --- | --- |")
    total_mismatch = 0
    for obj in runs:
        tag = obj.get("tag")
        if tag == args.baseline:
            continue
        m = sha_map(obj)
        bad_ids = []
        for key in sorted(set(m.keys()) & set(base_map.keys()), key=lambda x: (str(x[0]), x[1])):
            sha, err = m[key]
            bsha, berr = base_map[key]
            if err or berr or sha is None or bsha is None:
                continue
            if sha != bsha:
                bad_ids.append(key[0])
        total_mismatch += len(bad_ids)
        uniq = sorted(set(bad_ids))
        lines.append(
            "| %s | %d | %s |" % (tag, len(bad_ids), ", ".join(uniq) if uniq else "-")
        )
    lines.append("")
    lines.append("总不一致条数：%d" % total_mismatch)
    lines.append("")

    text = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
