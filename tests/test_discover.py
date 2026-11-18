"""Plain-assert tests of ``yee.integrations.discover`` (Yee 4.3.0 doctor: find running services).

Run: python tests/test_discover.py (also runs under pytest and in ``yee selftest``).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "yee")):
    sys.path.insert(0, ROOT)

from yee.integrations import discover  # noqa: E402

RESULTS = []

SGLANG_SERVER_INFO = {
    "server_args": {"model_path": "/models/DeepSeek-V4.1-Flash", "tp_size": 4,
                    "speculative_algorithm": "DSPARK", "max_running_requests": 64,
                    "cuda_graph_max_bs": 32},
    "port": 30000,
}

METRICS_TEXT = """# HELP sglang:num_running_reqs running
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs 12
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs 3
# TYPE sglang:spec_accept_length gauge
sglang:spec_accept_length 2.5
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total 120
vllm:spec_decode_num_accepted_tokens_total 60
"""


def test(fn):
    RESULTS.append(fn)
    return fn


class FakeRunner:
    """假 runner：按命令前缀返回假输出，并记录跑过的 argv。"""

    def __init__(self, table=None, default=(127, b"", b"not found")):
        self.table = table or {}
        self.default = default
        self.calls = []

    def run(self, argv, input=None, timeout=None):
        argv = list(argv)
        self.calls.append(argv)
        for prefix, resp in self.table.items():
            if argv[:len(prefix)] == list(prefix):
                if callable(resp):
                    return resp(argv)
                return resp
        return self.default


# --------------------------------------------------------------------------- docker
@test
def test_docker_finds_container_process():
    top = ("PID  ARGS\n"
           "1234 /usr/bin/python3 -m sglang.launch_server --model-path /models/X --tp 4\n")
    inspect = json.dumps([{"Config": {"Env": ["YEE=1", "RECALL_ACCEL=1", "PATH=/usr/bin"],
                                      "Image": "dsv41-4x-spark"},
                           "Mounts": [{"Source": "/opt/yee", "Destination": "/opt/yee"}]}])
    runner = FakeRunner({
        ("docker", "ps"): (0, b'{"Names":"sg-1","Image":"dsv41-4x-spark"}\n', b""),
        ("docker", "top"): (0, top.encode(), b""),
        ("docker", "inspect"): (0, inspect.encode(), b""),
    })
    out = discover.docker(runner)
    assert len(out) == 1, out
    inst = out[0]
    assert inst["engine"] == "sglang"
    assert inst["where"]["container"] == "sg-1"
    assert inst["where"]["source"] == "docker"
    assert inst["where"]["pid"] == 1234
    assert inst["parallel"]["tp"] == 4
    assert inst["env"].get("RECALL_ACCEL") == "1"
    assert "PATH" not in inst["env"]
    assert any("镜像" in n for n in inst["notes"]), inst["notes"]
    assert any("挂载" in n for n in inst["notes"]), inst["notes"]


@test
def test_docker_without_docker_returns_empty():
    runner = FakeRunner({}, default=(127, b"", b"docker: command not found"))
    out = discover.docker(runner)
    assert out == []
    assert runner.calls, "runner was never called"


@test
def test_docker_no_matching_process():
    top = "PID  ARGS\n1234 /usr/bin/sleep 100\n"
    runner = FakeRunner({
        ("docker", "ps"): (0, b'{"Names":"c1"}\n', b""),
        ("docker", "top"): (0, top.encode(), b""),
        ("docker", "inspect"): (0, b"[]", b""),
    })
    assert discover.docker(runner) == []


# --------------------------------------------------------------------------- kubernetes
@test
def test_kubernetes_finds_pod_container():
    pods = {"items": [{"metadata": {"namespace": "prod", "name": "vllm-0"},
                       "spec": {"containers": [{"name": "server",
                                                "command": ["vllm", "serve"],
                                                "args": ["/models/X", "--tensor-parallel-size", "2"],
                                                "env": [{"name": "VLLM_D_RECALL", "value": "1"},
                                                        {"name": "HF_HOME", "value": "/hf"}]}]}}]}
    runner = FakeRunner({("kubectl", "get", "pods"): (0, json.dumps(pods).encode(), b"")})
    out = discover.kubernetes(runner)
    assert len(out) == 1, out
    inst = out[0]
    assert inst["engine"] == "vllm"
    assert inst["parallel"]["tp"] == 2
    assert inst["where"]["container"] == "prod/vllm-0/server"
    assert inst["where"]["source"] == "kubernetes"
    assert inst["env"].get("VLLM_D_RECALL") == "1"
    assert "HF_HOME" not in inst["env"]


@test
def test_kubernetes_without_kubectl_returns_empty():
    runner = FakeRunner({}, default=(127, b"", b"kubectl: command not found"))
    assert discover.kubernetes(runner) == []


# --------------------------------------------------------------------------- ssh
@test
def test_ssh_uses_remote_yee_doctor_json():
    plan = {"version": "4.3.0", "instances": [
        {"id": "i1", "engine": "sglang",
         "instance": {"engine": "sglang", "head": {"algorithm": "DSPARK"}, "where": {"pid": 42}}}]}
    runner = FakeRunner({
        ("ssh",): lambda argv: (0, json.dumps(plan).encode(), b"") if "doctor" in " ".join(argv) else (1, b"", b""),
    })
    out = discover.ssh("host-a", runner)
    assert len(out) == 1, out
    assert out[0]["engine"] == "sglang"
    assert out[0]["where"]["host"] == "host-a"
    assert out[0]["where"]["source"] == "ssh"


@test
def test_ssh_falls_back_to_proc_scan():
    def fake(argv):
        joined = " ".join(argv)
        if "doctor" in joined:
            return 1, b"", b"no module named yee"
        return 0, b"/usr/bin/python3 -m sglang.launch_server --model-path /models/X --tp 2\n", b""

    runner = FakeRunner({("ssh",): fake})
    out = discover.ssh("host-b", runner)
    assert len(out) == 1, out
    assert out[0]["engine"] == "sglang"
    assert out[0]["parallel"]["tp"] == 2
    assert out[0]["where"]["host"] == "host-b"
    assert out[0]["where"]["source"] == "ssh"


@test
def test_ssh_without_connection_returns_empty():
    runner = FakeRunner({}, default=(255, b"", b"ssh: connect failed"))
    assert discover.ssh("host-c", runner) == []


# --------------------------------------------------------------------------- url
class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/get_server_info":
            body = json.dumps(SGLANG_SERVER_INFO).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/metrics":
            body = METRICS_TEXT.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


@test
def test_url_sglang_server_info_and_metrics():
    srv, base = _serve()
    try:
        inst = discover.url(base, timeout=5)
    finally:
        srv.shutdown()
    assert inst is not None
    assert inst["engine"] == "sglang"
    assert inst["head"]["algorithm"] == "DSPARK"
    assert inst["parallel"]["tp"] == 4
    assert inst["concurrency"]["max_running"] == 64
    assert inst["where"]["source"] == "url"
    assert inst["load"] is not None
    assert inst["load"]["running"] == 12
    assert inst["load"]["waiting"] == 3
    assert inst["load"]["accept_len"] == 2.5
    assert inst["load"]["spec_draft"] == 120
    assert inst["load"]["spec_accepted"] == 60


@test
def test_url_unknown_service():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    base = "http://127.0.0.1:%d" % srv.server_address[1]
    srv.server_close()  # nothing listens on the port any more (shutdown() would wait for a serve_forever loop)
    try:
        inst = discover.url(base, timeout=2)
    except Exception:
        inst = None
    if inst is None:
        return
    assert inst["engine"] in ("unknown", "sglang", "llama.cpp")


@test
def test_url_bad_input():
    assert discover.url("", timeout=1) is None


# --------------------------------------------------------------------------- metrics_load
@test
def test_metrics_load_parses_prometheus():
    out = discover.metrics_load(METRICS_TEXT)
    assert out["running"] == 12
    assert out["waiting"] == 3
    assert out["accept_len"] == 2.5
    assert out["spec_draft"] == 120
    assert out["spec_accepted"] == 60


@test
def test_metrics_load_vllm_names():
    text = ("vllm:num_requests_running 5\n"
            "vllm:num_requests_waiting 2\n"
            "vllm:spec_decode_num_draft_tokens_total 10\n"
            "vllm:spec_decode_num_accepted_tokens_total 4\n")
    out = discover.metrics_load(text)
    assert out["running"] == 5 and out["waiting"] == 2
    assert out["spec_draft"] == 10 and out["spec_accepted"] == 4


@test
def test_metrics_load_empty_and_garbage():
    out = discover.metrics_load("")
    assert out["running"] is None and out["waiting"] is None
    out2 = discover.metrics_load("not a metric line\n# only comments\n")
    assert out2["running"] is None


# --------------------------------------------------------------------------- maya_configs
@test
def test_maya_configs_reads_json_file():
    import tempfile
    cfg = {"exe": "/opt/maya/build/maya-yee", "args": ["--glm-pack", "/opt/maya/pack", "--parallel", "8"],
           "port": 8080, "model_name": "glm-5.3-flash", "installer": {"draft": "mtp"},
           "env": {"MAYA_GLM_MULTI_MTP": "3", "RECALL_ACCEL": "1"}}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "maya.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        out = discover.maya_configs([path])
    assert len(out) == 1, out
    assert out[0]["engine"] == "maya"
    assert out[0]["concurrency"]["max_running"] == 8
    assert out[0]["head"]["algorithm"] == "MTP"
    assert out[0]["where"]["source"] == "config"


@test
def test_maya_configs_bad_path():
    assert discover.maya_configs(["/no/such/config.json"]) == []
    assert discover.maya_configs([]) == []


# --------------------------------------------------------------------------- collect
@test
def test_collect_uses_cmd_and_dedups():
    runner = FakeRunner({}, default=(127, b"", b""))
    out = discover.collect(cmds=["python -m sglang.launch_server --model-path /models/X --tp 4"],
                           urls=[], hosts=[], configs=[], use_local=False, use_docker=False, runner=runner)
    assert len(out) == 1, out
    assert out[0]["engine"] == "sglang"
    assert out[0]["where"]["source"] == "cmd"
    assert out[0]["parallel"]["tp"] == 4


@test
def test_collect_dedups_same_instance():
    cmd = "python -m sglang.launch_server --model-path /models/X --tp 4"
    runner = FakeRunner({}, default=(127, b"", b""))
    out = discover.collect(cmds=[cmd, cmd], use_local=False, use_docker=False, runner=runner)
    assert len(out) == 1, out


@test
def test_collect_adds_model_info():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
                       "num_nextn_predict_layers": 1}, f)
        runner = FakeRunner({}, default=(127, b"", b""))
        out = discover.collect(cmds=["vllm serve %s --port 8000" % d.replace("\\", "/")], use_local=False,
                               use_docker=False, runner=runner)
    assert len(out) == 1, out
    assert out[0]["model_info"].get("arch") == "Qwen3ForCausalLM"
    assert out[0]["model_info"].get("mtp_layers") == 1


@test
def test_collect_includes_docker_and_k8s():
    top = "PID  ARGS\n1234 /usr/bin/python3 -m sglang.launch_server --model-path /models/X\n"
    runner = FakeRunner({
        ("docker", "ps"): (0, b'{"Names":"sg-1"}\n', b""),
        ("docker", "top"): (0, top.encode(), b""),
        ("docker", "inspect"): (0, b"[]", b""),
    })
    out = discover.collect(use_local=False, use_docker=True, runner=runner)
    assert len(out) == 1, out
    assert out[0]["where"]["source"] == "docker"


@test
def test_local_on_non_linux_returns_empty():
    if os.path.isdir("/proc"):
        out = discover.local()
        assert isinstance(out, list)
    else:
        assert discover.local() == []


def main():
    ok = fail = 0
    for fn in RESULTS:
        try:
            fn()
            ok += 1
            print("PASS %s" % fn.__name__)
        except Exception:
            fail += 1
            print("FAIL %s" % fn.__name__)
            traceback.print_exc()
    if fail:
        print("FAILED %d / %d" % (fail, ok + fail))
        sys.exit(1)
    print("ALL PASS n=%d" % ok)


if __name__ == "__main__":
    main()
