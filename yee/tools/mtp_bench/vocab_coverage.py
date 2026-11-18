#!/usr/bin/env python3
"""Yee 3.5.2 mtp_bench: which share of the generated tokens has an id below N (to choose YEE_MTP_VOCAB_N).

Needs a running llama-server of the target model (speculation not needed). For every prompt it applies the chat
template, generates greedily with ``return_tokens`` and counts the token ids.

    python3 vocab_coverage.py --base http://127.0.0.1:8080 --out coverage.json
"""
import argparse
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NS = "16384,32768,49152,65536,98304,131072,151936,200000"


def post(base, path, obj, timeout):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(obj).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def coverage(tokens, ns):
    n = max(1, len(tokens))
    return {str(k): round(sum(1 for t in tokens if t < k) / n, 4) for k in ns}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="llama-server URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--prompts", default=os.path.join(HERE, "prompts.jsonl"))
    ap.add_argument("--out", default="vocab_coverage.json")
    ap.add_argument("--n-predict", type=int, default=192)
    ap.add_argument("--ns", default=NS, help="comma-separated N values")
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args(argv)
    ns = [int(x) for x in a.ns.split(",") if x.strip()]
    prompts = [json.loads(line) for line in open(a.prompts, encoding="utf-8") if line.strip()]
    every, per_kind = [], {}
    for p in prompts:
        t = post(a.base, "/apply-template", {"messages": p["messages"],
                                             "chat_template_kwargs": {"enable_thinking": False}}, a.timeout)
        r = post(a.base, "/completion", {"prompt": t["prompt"], "n_predict": a.n_predict, "temperature": 0,
                                         "top_k": 1, "seed": 1, "cache_prompt": False, "return_tokens": True},
                 a.timeout)
        toks = r.get("tokens") or []
        every += toks
        per_kind.setdefault(p.get("kind", "?"), []).extend(toks)
        print("%s tokens=%d max_id=%s" % (p.get("id"), len(toks), max(toks) if toks else None), flush=True)
    out = {"total_tokens": len(every), "all": coverage(every, ns),
           "per_kind": {k: coverage(v, ns) for k, v in per_kind.items()}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("ALL " + json.dumps(out["all"]))
    for k, v in out["per_kind"].items():
        print("%s %s" % (k, json.dumps(v)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
