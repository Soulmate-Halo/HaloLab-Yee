# -*- coding: utf-8 -*-
"""yee_install.py 的单元与集成测试。集成用例需要 Linux bash（WSL 也可），用假的 ssh / docker / 启动脚本跑完整流程。"""
import importlib.util
import os
import shutil
import socket
import stat
import subprocess
import sys
import textwrap
import zipfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location("yee_install", os.path.join(KIT, "yee_install.py"))
yi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yi)

LINUX_BASH = sys.platform.startswith("linux") and shutil.which("bash") is not None
need_bash = pytest.mark.skipif(not LINUX_BASH, reason="needs Linux bash")
ORIG = "/root/.cache/xy_yee31:/opt/dsv41/adapter:/opt/b12x"


def make_wheel(d, version="4.3.0"):
    p = os.path.join(d, "yee-%s-py3-none-any.whl" % version)
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("yee/__init__.py", "__version__ = %r\n" % version)
        z.writestr(yi.TEMPLATE, "# yee sitecustomize template\n")
        z.writestr(yi.VCAP_FILES[0], "# verify_cap\n")
        z.writestr(yi.VCAP_FILES[1], "# vcap_policy\n")
        z.writestr("yee-%s.dist-info/METADATA" % version, "Metadata-Version: 2.1\nName: yee\nVersion: %s\n" % version)
    return p


# ------------------------------------------------------------------ pure helpers
def test_split_extra_last_wins_first_line_only():
    got = yi.split_extra("A=1 B=2 bad A=3\nC=4")
    assert got == {"A": "3", "B": "2"}
    assert list(got) == ["A", "B"]


def test_new_pythonpath_strips_old_yee_dirs():
    assert yi.new_pythonpath(ORIG, "yee43") == "/root/.cache/yee43:/opt/dsv41/adapter:/opt/b12x"
    assert yi.new_pythonpath("/root/.cache/yee42/:/root/.cache/yee43:/x", "yee43") == "/root/.cache/yee43:/x"
    assert yi.new_pythonpath("", "yee43") == "/root/.cache/yee43"
    once = yi.new_pythonpath(ORIG, "yee43")
    assert yi.new_pythonpath(once, "yee43") == once


def test_blocks_roundtrip_and_single():
    base = "A=1\nEXTRA_CONTAINER_ENV=\"X=1\"\n"
    one = yi.with_block(base, "4.3.0", "/root/.cache/yee43:/opt")
    two = yi.with_block(one, "4.3.0", "/root/.cache/yee43:/opt")
    assert two.count("# >>> yee 4.3.0 (install.sh) >>>") == 1
    assert "PYTHONPATH=/root/.cache/yee43:/opt YEE=1" in two
    assert yi.strip_blocks(two) == base
    other = yi.with_block(one, "4.4.0", "/root/.cache/yee44")
    assert "yee 4.3.0" not in other and other.count(">>> yee") == 1


def test_worker_hosts_rules():
    assert yi.worker_hosts({"WORKER_IPS": "a,b c"}) == ["a", "b", "c"]
    assert yi.worker_hosts({"WORKER_IPS": "a b", "WORKER_HOSTS": "h1,h2"}) == ["h1", "h2"]
    assert yi.worker_hosts({}) == ["10.0.0.2", "10.0.0.3"]
    assert yi.worker_hosts({"WORKER_IPS": "a b", "WORKER2_HOST": "h2"}) == ["a", "h2"]


def test_dir_name_and_mask():
    assert yi.dir_name_for("4.3.0") == "yee43"
    assert yi.dir_name_for("10.12.3") == "yee1012"
    assert yi.mask("API_KEY", "s") == "<已隐藏>" and yi.mask("YEE_P", "0.9") == "0.9"


def test_wheel_stage(tmp_path):
    w = yi.Wheel(make_wheel(str(tmp_path)))
    assert w.version == "4.3.0" and set(w.vcap) == {"verify_cap.py", "vcap_policy.py"}
    d1, d2 = tmp_path / "s1", tmp_path / "s2"
    d1.mkdir()
    d2.mkdir()
    sums = w.stage(str(d1), True)
    assert (d1 / "sitecustomize.py").read_bytes() == b"# yee sitecustomize template\n"
    assert (d1 / "verify_cap.py").exists() and (d1 / "YEE_VERSION").exists()
    assert sums["sitecustomize.py"] == yi.sha256_bytes(b"# yee sitecustomize template\n")
    w.stage(str(d2), False)
    assert not (d2 / "verify_cap.py").exists() and (d2 / "yee" / "__init__.py").exists()


def test_pick_env_and_find_recipes(tmp_path):
    r = tmp_path / "recipe"
    r.mkdir()
    (r / "start.sh").write_text("EXTRA_CONTAINER_ENV handling\n")
    (r / ".env").write_text("A=1\n")
    assert yi.pick_env(str(r))[1] == "start.sh"
    (r / ".env.tp4").write_text("A=1\n")
    (r / "start-tp4.sh").write_text("#!/bin/bash\n")
    env, launcher, default = yi.pick_env(str(r))
    assert env.endswith(".env.tp4") and launcher == "start-tp4.sh" and default
    _, _, default2 = yi.pick_env(str(r), str(tmp_path / "copy.tp4"))
    assert default2 is False
    other = tmp_path / "not_recipe"
    other.mkdir()
    (other / "start.sh").write_text("echo hi\n")
    (other / ".env").write_text("A=1\n")
    found = yi.find_recipes(str(tmp_path / "nowhere"), extra_roots=[str(r), str(other)])
    assert os.path.realpath(str(r)) in found and os.path.realpath(str(other)) not in found


class RecRunner(yi.Runner):
    def __init__(self):
        self.calls = []

    def run(self, argv, input=None, cwd=None, env=None, timeout=None):
        self.calls.append(list(argv))
        return 0, b"ok\n", b""


def test_pip_mode_commands(tmp_path):
    whl = make_wheel(str(tmp_path))
    r = RecRunner()
    assert yi.main(["--mode", "pip", "--wheel", whl, "--yes", "--python", "py"], runner=r) == 0
    assert r.calls[0] == ["py", "-m", "pip", "install", "--upgrade", os.path.abspath(whl)]
    assert r.calls[1] == ["py", "-m", "yee", "install-hook"]
    r2 = RecRunner()
    assert yi.main(["--mode", "pip", "--wheel", whl, "--check", "--python", "py"], runner=r2) == 0
    assert r2.calls == []


# ------------------------------------------------------------------ integration (bash)
FAKE_DOCKER = r"""#!/bin/bash
case "$1 $2" in
  "image inspect")
    if [[ "$*" == *json* ]]; then
      echo 'sha256:abc ["PATH=/usr/bin","PYTHONPATH=/opt/dsv41/adapter:/opt/b12x"]'
    else
      echo 'sha256:abc'
    fi ;;
  "run --rm") echo "${FAKE_VCAP_SHA:-0000}  /opt/dsv41/adapter/verify_cap.py" ;;
  "logs --tail") cat "${FAKE_DOCKER_LOGS:-/dev/null}" 2>/dev/null ;;
  *) echo "fake docker: $*" >&2; exit 1 ;;
esac
"""

FAKE_SERVER = r"""
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health": return self._send({"ok": True})
        if self.path == "/v1/models": return self._send({"data": [{"id": "fake-model"}]})
        self.send_response(404); self.end_headers()
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); self.rfile.read(n)
        if self.path == "/v1/chat/completions": return self._send({"choices": [{"message": {"content": "42"}}]})
        self.send_response(404); self.end_headers()
s = HTTPServer(("127.0.0.1", int(sys.argv[1])), H)
open(sys.argv[2], "w").write("ready")
s.serve_forever()
"""

FAKE_LAUNCHER = r"""#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
set -a; source "$ROOT/.env.tp4"; set +a
case "$1" in
  stop)
    if [ -f "$ROOT/srv.pid" ]; then kill "$(cat "$ROOT/srv.pid")" 2>/dev/null || true; rm -f "$ROOT/srv.pid"; fi
    sleep 0.3; echo stopped ;;
  serve)
    if [ -f "$ROOT/FAIL_IF_YEE" ] && grep -q '>>> yee' "$ROOT/.env.tp4"; then echo "boom"; exit 1; fi
    rm -f "$ROOT/ready"
    nohup python3 "$ROOT/fake_server.py" "$PORT" "$ROOT/ready" >/dev/null 2>&1 &
    echo $! > "$ROOT/srv.pid"
    for i in $(seq 100); do [ -f "$ROOT/ready" ] && { echo "Ready: API on port $PORT"; exit 0; }; sleep 0.1; done
    echo "server did not start"; exit 1 ;;
esac
"""


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class FakeSSH(yi.Runner):
    """ssh 换成在本机用各 worker 自己的 HOME 跑；docker 由 PATH 里的假脚本处理。"""

    def __init__(self, homes):
        self.homes = homes

    def run(self, argv, input=None, cwd=None, env=None, timeout=None):
        if argv and argv[0] == "ssh":
            host = argv[-2].split("@")[-1]
            e = dict(os.environ, HOME=self.homes[host])
            return super().run(["bash", "-c", argv[-1]], input=input, env=e, timeout=timeout)
        return super().run(argv, input=input, cwd=cwd, env=env, timeout=timeout)


def write_exec(path, text):
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def world(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_exec(str(bin_dir / "docker"), FAKE_DOCKER)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    head = tmp_path / "headhome"
    head.mkdir()
    monkeypatch.setenv("HOME", str(head))
    monkeypatch.setenv("YEE_INSTALL_NO_SYSTEMD", "1")
    monkeypatch.setenv("YEE_INSTALL_POLL_S", "0.3")
    homes = {}
    for h in ("w1", "w2"):
        d = tmp_path / ("home_" + h)
        d.mkdir()
        homes[h] = str(d)
    logs = tmp_path / "docker.log"
    logs.write_text("[yee] 4.3.0 defaults: RECALL_ACCEL=1\n[recall] patched x\n[recall] rank check on: 4 ranks\n"
                    "[recall] steps=10 rows=10\n[recall] steps=99 rows=120\n")
    monkeypatch.setenv("FAKE_DOCKER_LOGS", str(logs))
    monkeypatch.setenv("FAKE_VCAP_SHA", yi.VCAP_ORIG_SHA256)
    whl = make_wheel(str(tmp_path))
    recipes = []

    def recipe(name="recipe"):
        r = tmp_path / name
        r.mkdir()
        (r / "start.sh").write_text("# fake Mia start.sh: EXTRA_CONTAINER_ENV is split into -e K=V\n")
        write_exec(str(r / "start-tp4.sh"), FAKE_LAUNCHER)
        (r / "fake_server.py").write_text(FAKE_SERVER)
        (r / ".env.tp4").write_text(textwrap.dedent("""\
            WORKER_IPS="10.9.0.2 10.9.0.3"
            WORKER_HOSTS="w1 w2"
            WORKER_USER=root
            IMAGE=fake:img
            PORT=%d
            EXTRA_CONTAINER_ENV="A=1 YEE=0 RECALL_ACCEL_GIB=0.5"
            EXTRA_CONTAINER_ENV="$EXTRA_CONTAINER_ENV PYTHONPATH=%s"
            """ % (free_port(), ORIG)))
        recipes.append(r)
        return r

    yield {"tmp": tmp_path, "head": head, "homes": homes, "whl": whl, "recipe": recipe, "logs": logs,
           "runner": FakeSSH(homes)}
    for r in recipes:
        subprocess.run(["bash", str(r / "start-tp4.sh"), "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sourced(env_file, key):
    out = subprocess.run(["bash", "-c", 'set -a; source "$1"; set +a; printf %s "$EXTRA_CONTAINER_ENV"', "x", env_file],
                         stdout=subprocess.PIPE).stdout.decode()
    return yi.split_extra(out).get(key)


@need_bash
def test_check_is_read_only(world, capsys):
    r = world["recipe"]()
    before = (r / ".env.tp4").read_bytes()
    rc = yi.main(["--recipe", str(r), "--wheel", world["whl"], "--check"], runner=world["runner"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "安装后容器 PYTHONPATH：/root/.cache/yee43:/opt/dsv41/adapter:/opt/b12x" in out
    assert "放，每台镜像里的 verify_cap.py 都与补丁所基于的原件一致" in out
    assert "RECALL_ACCEL_GIB=0.5" in out and "YEE=0" in out
    assert (r / ".env.tp4").read_bytes() == before
    assert not (world["head"] / ".cache").exists()


@need_bash
def test_install_all_hosts_then_uninstall(world):
    r = world["recipe"]()
    env = str(r / ".env.tp4")
    assert yi.main(["--recipe", str(r), "--wheel", world["whl"], "--yes"], runner=world["runner"]) == 0
    for home in [str(world["head"])] + list(world["homes"].values()):
        base = os.path.join(home, ".cache", "yee43")
        assert os.path.isfile(os.path.join(base, "sitecustomize.py"))
        assert os.path.isfile(os.path.join(base, "verify_cap.py"))
        assert os.path.isfile(os.path.join(base, "yee", "__init__.py"))
    assert sourced(env, "PYTHONPATH") == "/root/.cache/yee43:/opt/dsv41/adapter:/opt/b12x"
    assert sourced(env, "YEE") == "1" and sourced(env, "A") == "1"
    ctl = r / "state-tp4" / "vcap_ctl.json"
    assert ctl.read_text().strip() == '{"mode": "adapt"}'
    # 再装一次：仍只有一个标记块，保留一份旧目录
    assert yi.main(["--recipe", str(r), "--wheel", world["whl"], "--yes"], runner=world["runner"]) == 0
    text = open(env).read()
    assert text.count(">>> yee 4.3.0") == 1
    olds = [n for n in os.listdir(os.path.join(str(world["head"]), ".cache")) if n.startswith("yee43.old-")]
    assert len(olds) == 1
    assert yi.main(["--recipe", str(r), "--wheel", world["whl"], "--yes", "--uninstall", "--purge"],
                   runner=world["runner"]) == 0
    assert ">>> yee" not in open(env).read()
    assert sourced(env, "PYTHONPATH") == ORIG and sourced(env, "YEE") == "0"
    assert not ctl.exists() and not (r / "state-tp4" / "vcap_ctl.json.yee-install").exists()
    assert not os.path.exists(os.path.join(str(world["head"]), ".cache", "yee43"))
    assert not os.path.exists(os.path.join(world["homes"]["w1"], ".cache", "yee43"))


@need_bash
def test_no_vcap_when_image_differs(world, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_VCAP_SHA", "1" * 64)
    r = world["recipe"]()
    assert yi.main(["--recipe", str(r), "--wheel", world["whl"], "--yes"], runner=world["runner"]) == 0
    assert "不放，head 镜像里的 verify_cap.py 与补丁原件不同" in capsys.readouterr().out
    assert not os.path.exists(os.path.join(str(world["head"]), ".cache", "yee43", "verify_cap.py"))
    assert not (r / "state-tp4" / "vcap_ctl.json").exists()


@need_bash
def test_override_mode_never_writes_state(world):
    r = world["recipe"]()
    copy = world["tmp"] / "copy.env.tp4"
    shutil.copyfile(str(r / ".env.tp4"), str(copy))
    before = (r / ".env.tp4").read_bytes()
    root = world["tmp"] / "override"
    assert yi.main(["--recipe", str(r), "--env-file", str(copy), "--cache-root-override", str(root),
                    "--wheel", world["whl"], "--yes"], runner=world["runner"]) == 0
    assert (root / "yee43" / "sitecustomize.py").exists()
    assert (r / ".env.tp4").read_bytes() == before and ">>> yee" in copy.read_text()
    assert not (r / "state-tp4").exists()


@need_bash
def test_restart_success(world, capsys):
    r = world["recipe"]()
    rc = yi.main(["--recipe", str(r), "--wheel", world["whl"], "--yes", "--restart", "--timeout-min", "2"],
                 runner=world["runner"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "重启成功" in out and "模型 fake-model 冒烟通过" in out
    assert "日志：[recall] rank check on: 4 ranks" in out and "日志：[recall] steps=99 rows=120" in out
    assert "Yee 已生效" in out


@need_bash
def test_restart_inactive_reports_code_2(world, capsys):
    world["logs"].write_text("[yee] 4.3.0 defaults: RECALL_ACCEL=1\n"
                             "[recall] rank check off: missing ranks [1, 2, 3]; copies stay off on every rank\n")
    r = world["recipe"]()
    rc = yi.main(["--recipe", str(r), "--wheel", world["whl"], "--yes", "--restart", "--timeout-min", "2"],
                 runner=world["runner"])
    out = capsys.readouterr().out
    assert rc == 2, out
    assert "Yee 自己关掉了" in out


@need_bash
def test_restart_failure_rolls_back(world, capsys):
    r = world["recipe"]()
    (r / "FAIL_IF_YEE").write_text("1")
    env = str(r / ".env.tp4")
    rc = yi.main(["--recipe", str(r), "--wheel", world["whl"], "--yes", "--restart", "--timeout-min", "2"],
                 runner=world["runner"])
    out = capsys.readouterr().out
    assert rc == 3, out
    assert "重启失败" in out and "回退成功" in out
    assert ">>> yee" not in open(env).read()
    assert sourced(env, "PYTHONPATH") == ORIG


@need_bash
def test_restart_refuses_non_default_env(world, capsys):
    r = world["recipe"]()
    copy = world["tmp"] / "copy.env.tp4"
    shutil.copyfile(str(r / ".env.tp4"), str(copy))
    rc = yi.main(["--recipe", str(r), "--env-file", str(copy), "--wheel", world["whl"], "--yes", "--restart"],
                 runner=world["runner"])
    assert rc == 1
    assert "--restart 只支持方案自己的 env 文件" in capsys.readouterr().out
