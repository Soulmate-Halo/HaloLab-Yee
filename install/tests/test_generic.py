# -*- coding: utf-8 -*-
"""generic 模式（任何部署：本机进程 / docker / 多机 / K8s）的测试。

用假的 discover / plan 模块与假的 runner：不碰真的 docker、ssh，也不装真的东西。
"""
import importlib.util
import json
import os
import zipfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location("yee_install", os.path.join(KIT, "yee_install.py"))
yi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yi)


# ------------------------------------------------------------------ fakes
class FakeRunner(yi.Runner):
    """记录 argv，按命令前缀给假输出。"""

    def __init__(self):
        self.calls = []

    def run(self, argv, input=None, cwd=None, env=None, timeout=None):
        argv = list(argv)
        self.calls.append(argv)
        line = " ".join(argv)
        if "command -v python3" in line:
            return 0, b"/usr/bin/python3\n", b""
        if "--json" in line and "config" in line:
            return 0, b'{"version": "4.3.0", "recall": true}\n', b""
        if "install-hook" in line or "uninstall-hook" in line:
            return 0, b"hook ok\n", b""
        return 0, b"ok\n", b""

    def lines(self):
        return [" ".join(c) for c in self.calls]

    def has(self, needle):
        return any(needle in l for l in self.lines())


class FakeDiscover:
    def __init__(self, instances):
        self.instances = list(instances)
        self.kwargs = None
        self.calls = 0

    def collect(self, **kw):
        self.kwargs = kw
        self.calls += 1
        return list(self.instances)


class FakePlan:
    def __init__(self, exit_code=0):
        self.exit_code = exit_code
        self.analyzed = []

    def analyze(self, instances, workload=None):
        self.analyzed.append(list(instances))
        return {"version": "4.3.0",
                "instances": [{"id": "i%d" % (i + 1), "label": "x", "engine": "sglang", "role": "server",
                               "verdict": "needs-change", "features": {}, "changes": [], "notes": []}
                              for i, _ in enumerate(instances)],
                "cluster": {"groups": [], "changes": [], "notes": []},
                "summary": {"accelerated": 0, "needs_change": 1, "needs_patch": 0, "unsupported": 0,
                            "not_needed": 0},
                "exit": self.exit_code}

    def render(self, plan):
        return "Yee 4.3.0 部署诊断：找到 %d 个实例\n合计：可加速 0 · 要改参数 1。退出码 %d" % (
            len(plan.get("instances", [])), plan.get("exit", 0))


def make_wheel(d, version="4.3.0", with_integrations=False):
    p = os.path.join(d, "yee-%s-py3-none-any.whl" % version)
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("yee/__init__.py", "__version__ = %r\n" % version)
        z.writestr(yi.TEMPLATE, "# yee sitecustomize template\n")
        z.writestr("yee-%s.dist-info/METADATA" % version, "Metadata-Version: 2.1\nName: yee\nVersion: %s\n" % version)
        if with_integrations:
            z.writestr("yee/integrations/__init__.py", "")
            z.writestr("yee/integrations/discover.py", "def collect(**kw):\n    return []\n")
            z.writestr("yee/integrations/plan.py", "def analyze(i, workload=None):\n    return {'exit': 0}\n")
    return p


def inst(engine="sglang", host="", container=None, pid=None, argv=None):
    return {"engine": engine,
            "argv": list(argv or ["/usr/bin/python3", "-m", "sglang.launch_server"]),
            "where": {"host": host, "pid": pid, "container": container, "source": "test"},
            "head": {"algorithm": "NONE"}, "parallel": {}, "env": {}, "notes": []}


def test_local_python_prefers_proc_exe(tmp_path, monkeypatch):
    """argv[0] 是绝对路径的 python 时直接用它（venv 的 bin/python 不会被 /proc/<pid>/exe 解析成底层解释器）；
    argv[0] 不是绝对路径时才取 /proc/<pid>/exe 的 readlink。"""
    whl = make_wheel(str(tmp_path))
    seen = {}

    def fake_readlink(path):
        seen["path"] = path
        return "/opt/conda/bin/python3.11"

    monkeypatch.setattr(os, "readlink", fake_readlink)
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--plan-out",
                       str(tmp_path / "plan.json")], [inst("sglang", pid=4242, argv=["/opt/venv/bin/python3", "-m",
                                                                                       "sglang.launch_server"])])
    assert "path" not in seen
    assert r.has("/opt/venv/bin/python3 -m pip install"), r.lines()
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--plan-out",
                       str(tmp_path / "plan.json")], [inst("sglang", pid=4242, argv=["python3", "-m",
                                                                                       "sglang.launch_server"])])
    assert seen.get("path") == "/proc/4242/exe"
    assert r.has("/opt/conda/bin/python3.11 -m pip install"), r.lines()


def test_json_tail_skips_log_lines():
    assert yi.json_tail('[yee] 4.3.0 defaults: A=1\n[recall-accel] no hooks\n{"a": 1}') == {"a": 1}
    assert yi.json_tail('[yee] x\n[\n 1\n]') == [1]
    assert yi.json_tail('{"b": 2}') == {"b": 2}


def run(argv, instances, exit_code=0, tmp_path=None):
    """跑一次 generic 模式，返回 (退出码, runner, discover, plan)。"""
    r = FakeRunner()
    d = FakeDiscover(instances)
    p = FakePlan(exit_code)
    rc = yi.main(argv, runner=r, modules=(d, p))
    return rc, r, d, p


# ------------------------------------------------------------------ cases
def test_local_process_target_gets_installed(tmp_path):
    whl = make_wheel(str(tmp_path))
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--plan-out",
                       str(tmp_path / "plan.json")], [inst("sglang", pid=1234)], exit_code=1)
    assert rc == 1  # 假 plan 的 exit 是 1
    assert r.has("pip install --no-deps --force-reinstall"), r.lines()
    assert r.has("install-hook")
    assert r.has("config --json")
    # 三条命令都在同一个解释器上
    joined = "\n".join(r.lines())
    assert "-m pip install --no-deps --force-reinstall" in joined
    assert "-m yee install-hook" in joined
    assert "-m yee config --json" in joined
    # 没有 docker / ssh
    assert not r.has("docker")
    assert not r.has("ssh")


def test_docker_target_copies_and_installs(tmp_path, capsys):
    whl = make_wheel(str(tmp_path))
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--plan-out",
                       str(tmp_path / "plan.json")], [inst("vllm", container="vllm1")])
    out = capsys.readouterr().out
    assert r.has("docker cp"), r.lines()
    assert "vllm1:/tmp/yee-4.3.0-py3-none-any.whl" in "\n".join(r.lines())
    assert r.has("docker exec vllm1 /usr/bin/python3 -m pip install")
    assert r.has("docker exec vllm1 /usr/bin/python3 -m yee install-hook")
    assert r.has("docker exec vllm1 /usr/bin/python3 -m yee config --json")
    assert "持久做法" in out and "PYTHONPATH=/opt/yee" in out


def test_ssh_target_uses_scp_and_ssh(tmp_path):
    whl = make_wheel(str(tmp_path))
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--plan-out",
                       str(tmp_path / "plan.json")], [inst("sglang", host="h1")])
    joined = "\n".join(r.lines())
    assert "scp" in joined and "h1:/tmp/yee-4.3.0-py3-none-any.whl" in joined
    assert "ssh" in joined
    assert "pip install --no-deps --force-reinstall /tmp/yee-4.3.0-py3-none-any.whl" in joined
    assert "yee install-hook" in joined and "yee config --json" in joined
    assert "docker" not in joined


def test_patch_engine_is_not_installed(tmp_path, capsys):
    whl = make_wheel(str(tmp_path))
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--plan-out",
                       str(tmp_path / "plan.json")], [inst("maya", argv=["/opt/maya/build/maya-yee", "--parallel", "8"])])
    out = capsys.readouterr().out
    assert not r.has("pip install")
    assert not r.has("install-hook")
    assert "补丁引擎" in out


def test_check_does_not_install(tmp_path):
    whl = make_wheel(str(tmp_path))
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--check", "--plan-out",
                       str(tmp_path / "plan.json")], [inst("sglang", pid=7), inst("vllm", container="c1")])
    assert not r.has("pip install")
    assert not r.has("install-hook")
    assert not r.has("docker cp")
    assert not r.has("docker exec")
    assert not r.has("ssh")


def test_uninstall_runs_uninstall_hook(tmp_path):
    whl = make_wheel(str(tmp_path))
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--uninstall", "--plan-out",
                       str(tmp_path / "plan.json")], [inst("sglang", pid=7)])
    joined = "\n".join(r.lines())
    assert "yee uninstall-hook" in joined
    assert "pip uninstall -y yee" in joined
    assert "pip install" not in joined


def test_restart_restarts_docker_container(tmp_path):
    whl = make_wheel(str(tmp_path))
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--restart", "--plan-out",
                       str(tmp_path / "plan.json")], [inst("vllm", container="vllm1")])
    assert r.has("docker restart vllm1"), r.lines()
    # 重启后再诊断一次
    assert d.calls == 2


def test_restart_cmd_runs_locally(tmp_path):
    whl = make_wheel(str(tmp_path))
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--restart", "--restart-cmd",
                       "systemctl restart sglang", "--plan-out", str(tmp_path / "plan.json")],
                      [inst("sglang", pid=7)])
    assert r.has("systemctl restart sglang"), r.lines()


def test_exit_code_equals_plan_exit(tmp_path):
    whl = make_wheel(str(tmp_path))
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--plan-out",
                       str(tmp_path / "plan.json")], [inst("sglang", pid=7)], exit_code=0)
    assert rc == 0
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--plan-out",
                       str(tmp_path / "plan.json")], [inst("sglang", pid=7)], exit_code=3)
    assert rc == 3


def test_install_failure_returns_1(tmp_path):
    whl = make_wheel(str(tmp_path))
    r = FakeRunner()
    d = FakeDiscover([inst("sglang", pid=7)])
    p = FakePlan(0)
    real_run = r.run

    def failing(argv, input=None, cwd=None, env=None, timeout=None):
        argv = list(argv)
        r.calls.append(argv)
        if "pip" in " ".join(argv):
            return 1, b"", b"boom\n"
        return real_run(argv, input=input, cwd=cwd, env=env, timeout=timeout)

    r.run = failing
    rc = yi.main(["--mode", "generic", "--wheel", whl, "--yes", "--plan-out", str(tmp_path / "plan.json")],
                 runner=r, modules=(d, p))
    assert rc == 1


def test_wheel_without_integrations_exits_1(tmp_path, capsys):
    whl = make_wheel(str(tmp_path), with_integrations=False)
    rc = yi.main(["--mode", "generic", "--wheel", whl, "--yes", "--plan-out", str(tmp_path / "plan.json")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "discover" in out


def test_plan_json_written(tmp_path, monkeypatch):
    whl = make_wheel(str(tmp_path))
    monkeypatch.chdir(str(tmp_path))
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes"], [inst("sglang", pid=7)], exit_code=1)
    path = tmp_path / "yee-plan.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["exit"] == 1 and data["version"] == "4.3.0"


def test_plan_out_custom_path(tmp_path):
    whl = make_wheel(str(tmp_path))
    out = tmp_path / "sub" / "my.json"
    out.parent.mkdir()
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--plan-out", str(out)],
                      [inst("sglang", pid=7)])
    assert out.is_file()


def test_same_interpreter_installed_once(tmp_path):
    whl = make_wheel(str(tmp_path))
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--plan-out",
                       str(tmp_path / "plan.json")],
                      [inst("sglang"), inst("vllm")])
    assert len([l for l in r.lines() if "pip install --no-deps --force-reinstall" in l]) == 1


def test_collect_arguments_are_forwarded(tmp_path):
    whl = make_wheel(str(tmp_path))
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--cmd", "python -m sglang.launch_server",
                       "--cmd", "python -m vllm.entrypoints.openai.api_server", "--url", "http://127.0.0.1:8000",
                       "--hosts", "h1,h2", "--config", "maya.json", "--k8s", "--plan-out",
                       str(tmp_path / "plan.json")], [])
    kw = d.kwargs
    assert kw["cmds"] == ["python -m sglang.launch_server", "python -m vllm.entrypoints.openai.api_server"]
    assert kw["urls"] == ["http://127.0.0.1:8000"]
    assert kw["hosts"] == ["h1", "h2"]
    assert kw["configs"] == ["maya.json"]
    assert kw["use_local"] is True and kw["use_docker"] is True and kw["use_k8s"] is True


def test_no_docker_flag(tmp_path):
    whl = make_wheel(str(tmp_path))
    rc, r, d, p = run(["--mode", "generic", "--wheel", whl, "--yes", "--no-docker", "--plan-out",
                       str(tmp_path / "plan.json")], [])
    assert d.kwargs["use_docker"] is False


def test_auto_mode_falls_back_to_generic(tmp_path, monkeypatch):
    whl = make_wheel(str(tmp_path))
    monkeypatch.setattr(yi, "find_recipes", lambda *a, **k: [])
    r = FakeRunner()
    d = FakeDiscover([])
    p = FakePlan(0)
    rc = yi.main(["--wheel", whl, "--yes", "--plan-out", str(tmp_path / "plan.json")], runner=r, modules=(d, p))
    assert rc == 0
    assert d.calls == 1


def test_pip_flag_keeps_old_path(tmp_path):
    whl = make_wheel(str(tmp_path))
    r = FakeRunner()
    rc = yi.main(["--pip", "--wheel", whl, "--yes", "--python", "py"], runner=r)
    assert rc == 0
    assert r.calls[0] == ["py", "-m", "pip", "install", "--upgrade", os.path.abspath(whl)]
    assert r.calls[1] == ["py", "-m", "yee", "install-hook"]
