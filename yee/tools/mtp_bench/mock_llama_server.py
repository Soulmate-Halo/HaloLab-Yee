#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mock_llama_server.py — 标准库 http.server 模拟 llama-server，用于离线联调 bench/compare。

用法示例：
    python3 mock_llama_server.py --port 18670 --tok-s 40 --draft \
        --diff-id qa-2 --diff-text "这是被篡改的回答内容。"
    python3 mock_llama_server.py --port 18671 --tok-s 90 --no-draft

接口：
    GET  /health                  -> 200 {"status":"ok"}
    POST /v1/chat/completions     -> OpenAI 风格响应 + llama.cpp timings 扩展字段

确定性：同一 (id, tag) 一定返回同样内容；--diff-id 指定的 id 返回不同内容。
"""

import argparse
import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="mock llama-server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18670)
    p.add_argument("--tok-s", type=float, default=40.0, help="模拟预测速度 token/s")
    p.add_argument("--prompt-tok-s", type=float, default=200.0, help="模拟预填充速度 token/s")
    p.add_argument("--draft", dest="draft", action="store_true", default=True,
                   help="timings 里带 draft_n / draft_n_accepted")
    p.add_argument("--no-draft", dest="draft", action="store_false",
                   help="timings 不带 draft 字段")
    p.add_argument("--accept-rate", type=float, default=0.72, help="mock 的草稿接受率（仅 --draft 时生效）")
    p.add_argument("--tag", default="mock", help="本次 mock 实例的标签，参与内容哈希，可模拟不同服务/不同配置")
    p.add_argument("--diff-id", default=None, help="该 id 返回 --diff-text 指定内容")
    p.add_argument("--diff-text", default=None, help="--diff-id 对应的替身内容")
    p.add_argument("--max-tokens", type=int, default=256, help="mock 实际生成的 token 数上限")
    return p.parse_args(argv)


def user_id_of(messages):
    """从 user 消息首行 [id:xxx] 提取 prompt id；没有则返回 unknown。"""
    for m in messages:
        if m.get("role") == "user":
            c = (m.get("content") or "").strip()
            if c.startswith("[id:"):
                end = c.find("]")
                if end > 4:
                    return c[4:end].strip()
            return "unknown"
    return "unknown"


def chat_of(messages, max_tokens, tag):
    """根据 messages 生成确定性回答：内容只由 tag 与 prompt id 决定，
    与服务端参数无关，因此同 tag 的两次运行内容必然一致。"""
    pid = user_id_of(messages)
    seed = "%s|%s" % (tag, pid)
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return (
        "mock answer for %s seed=%s. " % (pid, h[:12])
        + ("确定性输出，用于校验一致性。" * 8)
        + " tail=" + h[-8:]
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "mock-llama-server/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 静音
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/health") or self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/v1/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            self._json(400, {"error": "bad json: %s" % e})
            return

        cfg = self.server.cfg
        messages = req.get("messages") or []
        pid = user_id_of(messages)
        max_tokens = int(req.get("max_tokens") or 256)

        # 确定性内容：默认由 (tag, id) 派生，同 tag 两次运行恒定；
        # --diff-id 命中时返回 --diff-text，用来制造内容不一致。
        if cfg.diff_id and pid == cfg.diff_id:
            content = cfg.diff_text
        else:
            content = chat_of(messages, max_tokens, cfg.tag)

        # 确定性 timings
        prompt_n = 32 + (len(json.dumps(messages, ensure_ascii=False)) % 64)
        predicted_n = min(cfg.max_tokens, max_tokens)
        predicted_ms = predicted_n / cfg.tok_s * 1000.0 if cfg.tok_s > 0 else 0.0
        prompt_ms = prompt_n / cfg.prompt_tok_s * 1000.0 if cfg.prompt_tok_s > 0 else 0.0
        timings = {
            "prompt_n": prompt_n,
            "prompt_ms": round(prompt_ms, 3),
            "predicted_n": predicted_n,
            "predicted_ms": round(predicted_ms, 3),
            "predicted_per_second": round(cfg.tok_s, 3),
        }
        if cfg.draft:
            dn = predicted_n * 4
            da = int(round(dn * cfg.accept_rate))
            timings["draft_n"] = dn
            timings["draft_n_accepted"] = da

        # 模拟耗时，但不真的等那么久
        self._json(200, {
            "id": "chatcmpl-mock-%s" % hashlib.sha256(("%s|%s" % (cfg.tag, pid)).encode("utf-8")).hexdigest()[:12],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model") or "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "timings": timings,
        })


def main(argv=None):
    cfg = parse_args(argv)
    if cfg.diff_id and cfg.diff_text is None:
        sys.stderr.write("[mock] --diff-id 需要同时给 --diff-text\n")
        return 2
    srv = ThreadingHTTPServer((cfg.host, cfg.port), Handler)
    srv.cfg = cfg
    print("[mock] listening on http://%s:%d tok_s=%.1f draft=%s tag=%s"
          % (cfg.host, cfg.port, cfg.tok_s, cfg.draft, cfg.tag), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
