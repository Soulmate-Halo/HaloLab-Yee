"""Yee 3.5.2: the mtp_bench tools (bench.py, compare.py and their mock llama-server)."""
import json
import os
import py_compile
import socket
import subprocess
import sys
import time
import urllib.request

import yee

TOOLS = os.path.join(os.path.dirname(os.path.abspath(yee.__file__)), "tools", "mtp_bench")
FILES = ("bench.py", "compare.py", "vocab_coverage.py", "mock_llama_server.py", "prompts.jsonl", "run_matrix.sh",
         "contention.sh", "README.md", "__init__.py")


def test_files_present_and_compile():
    for name in FILES:
        assert os.path.isfile(os.path.join(TOOLS, name)), name
    for name in FILES:
        if name.endswith(".py"):
            py_compile.compile(os.path.join(TOOLS, name), doraise=True)
    with open(os.path.join(TOOLS, "prompts.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 15
    assert {r["kind"] for r in rows} == {"code", "copy", "rewrite", "qa", "creative"}
    with open(os.path.join(TOOLS, "run_matrix.sh"), encoding="utf-8") as f:
        sh = f.read()
    assert "bench.py" in sh and "compare.py" in sh and "prompts.jsonl" in sh
    for old in ("bench670", "compare670", "prompts670", "run670"):
        assert old not in sh, old


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run(args, cwd):
    return subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=300)


def test_bench_and_compare_roundtrip(tmp_path):
    port = _free_port()
    mock = subprocess.Popen([sys.executable, os.path.join(TOOLS, "mock_llama_server.py"), "--port", str(port),
                             "--tag", "same", "--tok-s", "50", "--no-draft"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = "http://127.0.0.1:%d" % port
        for _ in range(100):
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.1)
        prompts = os.path.join(TOOLS, "prompts.jsonl")
        for tag in ("a", "b"):
            p = _run([os.path.join(TOOLS, "bench.py"), "--base", base, "--prompts", prompts, "--tag", tag, "--out",
                      str(tmp_path / (tag + ".json")), "--max-tokens", "8"], str(tmp_path))
            assert p.returncode == 0, p.stdout + p.stderr
            assert "SUMMARY tag=%s" % tag in p.stdout
        b = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
        b["records"][2]["content_sha16"] = "ffffffffffffffff"
        changed = b["records"][2]["id"]
        (tmp_path / "b.json").write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
        out = tmp_path / "compare.md"
        p = _run([os.path.join(TOOLS, "compare.py"), "--inputs", str(tmp_path / "a.json"), str(tmp_path / "b.json"),
                  "--baseline", "a", "--out", str(out)], str(tmp_path))
        assert p.returncode == 0, p.stdout + p.stderr
        md = out.read_text(encoding="utf-8")
        assert "| b | 1 | %s |" % changed in md, md
    finally:
        mock.terminate()
        try:
            mock.wait(timeout=10)
        except Exception:
            mock.kill()
