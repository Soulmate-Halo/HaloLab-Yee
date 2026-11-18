#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench.py — llama-server 投机解码测速客户端（仅标准库）。

用法示例：
    python3 bench.py --base http://127.0.0.1:18670 --prompts prompts.jsonl \
        --tag gpu_k1 --out gpu_k1.json --max-tokens 256 --repeat 1 --timeout 600 --warmup
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="llama-server speculative decoding bench")
    p.add_argument("--base", required=True, help="服务基址，例如 http://127.0.0.1:18670")
    p.add_argument("--prompts", required=True, help="prompts jsonl 文件")
    p.add_argument("--tag", required=True, help="本次运行的标签")
    p.add_argument("--out", required=True, help="输出 json 文件")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--warmup", action="store_true", help="先发一个短请求，不计入统计")
    return p.parse_args(argv)


def load_prompts(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows


def post_chat(base, messages, max_tokens, timeout):
    """POST {base}/v1/chat/completions，返回 (resp_dict_or_None, wall_s, error_or_None)。"""
    body = {
        "model": "bench",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "seed": 1,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
        wall = time.monotonic() - t0
        return json.loads(raw), wall, None
    except urllib.error.HTTPError as e:
        wall = time.monotonic() - t0
        try:
            detail = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            detail = ""
        return None, wall, "HTTPError %s %s" % (e.code, detail)
    except Exception as e:  # 超时 / 连接失败等
        wall = time.monotonic() - t0
        return None, wall, "%s: %s" % (type(e).__name__, e)


def _num(v):
    """数值字段：能转 float 就转，否则 None。"""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int_or_none(v):
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def extract_timings(resp):
    t = resp.get("timings") if isinstance(resp, dict) else None
    if not isinstance(t, dict):
        return {
            "predicted_n": None,
            "predicted_ms": None,
            "tok_s": None,
            "prompt_n": None,
            "prompt_ms": None,
            "draft_n": None,
            "draft_n_accepted": None,
        }
    pn = _num(t.get("predicted_n"))
    pms = _num(t.get("predicted_ms"))
    tok_s = (pn / pms * 1000.0) if (pn is not None and pms) else None
    return {
        "predicted_n": pn,
        "predicted_ms": pms,
        "tok_s": tok_s,
        "prompt_n": _int_or_none(t.get("prompt_n")),
        "prompt_ms": _num(t.get("prompt_ms")),
        "draft_n": _int_or_none(t.get("draft_n")),
        "draft_n_accepted": _int_or_none(t.get("draft_n_accepted")),
    }


def run_one(base, row, max_tokens, timeout):
    """跑一条 prompt，返回记录 dict。"""
    messages = row.get("messages", [])
    resp, wall, err = post_chat(base, messages, max_tokens, timeout)
    rec = {
        "tag": None,
        "id": row.get("id"),
        "kind": row.get("kind"),
        "rep": 0,
        "predicted_n": None,
        "predicted_ms": None,
        "tok_s": None,
        "prompt_n": None,
        "prompt_ms": None,
        "draft_n": None,
        "draft_n_accepted": None,
        "accept_rate": None,
        "content_sha16": None,
        "head80": None,
        "wall_s": round(wall, 3),
        "error": err,
    }
    if err is not None or resp is None:
        return rec
    try:
        content = resp["choices"][0]["message"]["content"] or ""
    except Exception:
        rec["error"] = "bad_response_shape: %s" % json.dumps(resp)[:400]
        return rec
    rec.update(extract_timings(resp))
    rec["content_sha16"] = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    rec["head80"] = content[:80]
    if rec["draft_n"]:
        rec["accept_rate"] = (rec["draft_n_accepted"] or 0) / rec["draft_n"]
    return rec


def summarize(records):
    kinds = {}
    total = {
        "predicted_n": 0.0,
        "predicted_ms": 0.0,
        "draft_n": 0,
        "draft_n_accepted": 0,
        "requests": 0,
        "fail": 0,
    }
    for r in records:
        k = r.get("kind")
        slot = kinds.setdefault(
            k,
            {
                "predicted_n": 0.0,
                "predicted_ms": 0.0,
                "draft_n": 0,
                "draft_n_accepted": 0,
                "requests": 0,
                "fail": 0,
                "agg_tok_s": None,
                "accept_rate": None,
            },
        )
        slot["requests"] += 1
        total["requests"] += 1
        if r.get("error"):
            slot["fail"] += 1
            total["fail"] += 1
            continue
        pn = r.get("predicted_n")
        pms = r.get("predicted_ms")
        if pn is not None:
            slot["predicted_n"] += pn
            total["predicted_n"] += pn
        if pms is not None:
            slot["predicted_ms"] += pms
            total["predicted_ms"] += pms
        dn = r.get("draft_n")
        da = r.get("draft_n_accepted")
        if dn is not None:
            slot["draft_n"] += dn
            total["draft_n"] += dn
        if da is not None:
            slot["draft_n_accepted"] += da
            total["draft_n_accepted"] += da

    def finish(slot):
        if slot["predicted_ms"]:
            slot["agg_tok_s"] = slot["predicted_n"] / slot["predicted_ms"] * 1000.0
        else:
            slot["agg_tok_s"] = None
        if slot["draft_n"]:
            slot["accept_rate"] = slot["draft_n_accepted"] / slot["draft_n"]
        else:
            slot["accept_rate"] = None
        return slot

    for slot in kinds.values():
        finish(slot)
    finish(total)
    return {"by_kind": kinds, "total": total}

def main(argv=None):
    args = parse_args(argv)
    rows = load_prompts(args.prompts)
    records = []
    for row in rows:
        for rep in range(1, args.repeat + 1):
            rec = run_one(args.base, row, args.max_tokens, args.timeout)
            rec["tag"] = args.tag
            rec["rep"] = rep
            records.append(rec)
            if rec["error"]:
                sys.stderr.write(
                    "[bench] %s rep%d FAIL %s\n" % (rec["id"], rep, rec["error"])
                )
    summary = summarize(records)
    out_obj = {
        "tag": args.tag,
        "args": {
            "base": args.base,
            "prompts": args.prompts,
            "tag": args.tag,
            "out": args.out,
            "max_tokens": args.max_tokens,
            "repeat": args.repeat,
            "timeout": args.timeout,
            "warmup": bool(args.warmup),
        },
        "records": records,
        "summary": summary,
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
        f.write("\n")

    t = summary["total"]
    tok_s = t["agg_tok_s"]
    acc = t["accept_rate"]
    print(
        "SUMMARY tag=%s tok_s=%s accept=%s n=%d fail=%d"
        % (
            args.tag,
            ("%.2f" % tok_s) if tok_s is not None else "null",
            ("%.4f" % acc) if acc is not None else "null",
            t["requests"],
            t["fail"],
        )
    )
    return 0 if t["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
