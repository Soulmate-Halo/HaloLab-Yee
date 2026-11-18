#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Yee 一键安装（只用标准库，Python 3.8+）。

两种模式。

Mia 模式：在 MiaAI dsv41-4x-spark / dsv41-3x-spark 方案的 head 宿主机上运行：

    bash install.sh --check              只检查，不写任何东西
    bash install.sh --restart            每台机器都装、重启服务、健康检查 + 冒烟，失败自动回退
    bash install.sh --uninstall --restart

generic 模式：任何部署（Mia 方案以外的本机进程、docker 容器、多台机器、K8s）：

    bash install.sh --check              只诊断，不装
    bash install.sh                      给发现的 Python 引擎装上，然后打印诊断报告
    bash install.sh --hosts h1,h2        连别的机器一起看、一起装
    bash install.sh --restart            装完重启 docker 容器，再诊断一次
    bash install.sh --uninstall          卸载

generic 模式用 wheel 里的 yee.integrations.discover / plan 做发现与诊断，装进每个目标解释器的都是同一个 wheel。
Yee 的开关全部默认打开，不需要设任何别的环境变量。--pip 保留老路径：只装进当前 Python。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import importlib
import io
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

VCAP_ORIG_SHA256 = "949c3fb72d72b48e37eaffc10e101d403e023e3b2700907fd23b1a971225d84d"  # Mia 原版 verify_cap.py
ADAPTER_VERIFY_CAP = "/opt/dsv41/adapter/verify_cap.py"
CONTAINER_CACHE = "/root/.cache"
TEMPLATE = "yee/integrations/sitecustomize_template.py"
VCAP_FILES = ("yee/patches/mia-dsv41-vcap/files/verify_cap.py", "yee/patches/mia-dsv41-vcap/files/vcap_policy.py")
BEGIN_RE = re.compile(r"^# >>> yee \S+ \(install\.sh\) >>>\s*$")
END_RE = re.compile(r"^# <<< yee \S+ <<<\s*$")
OLD_DIR_RE = re.compile(r"^/root/\.cache/(?:xy_)?yee[^/:]*/?$")
YEE_KEY_RE = re.compile(r"^(?:YEE|RECALL_|SGLANG_D_RECALL|VLLM_D_RECALL|DSV41_RECALL|DSV41_VCAP)")
SECRET_RE = re.compile(r"(?:KEY|TOKEN|PASS|SECRET|PWD)", re.I)
FAIL_MARKERS = ("rank check off", "install failed")
LOG_MARKERS = ("[yee]", "[recall]", "[yee-eb]", "[yee-kad]", "[recall-accel]")


def say(msg):
    print("[yee-install] %s" % msg, flush=True)


class Abort(Exception):
    def __init__(self, msg, code=1):
        super().__init__(msg)
        self.code = code


def txt(b):
    if b is None:
        return ""
    if isinstance(b, str):
        return b
    return b.decode("utf-8", "replace")


def json_tail(text):
    """JSON printed after other lines: once the hook is installed, every interpreter start prints [yee] lines first."""
    text = text or ""
    lines = text.splitlines()
    for k, ln in enumerate(lines):
        if ln.lstrip().startswith(("{", "[")):
            try:
                return json.loads("\n".join(lines[k:]))
            except ValueError:
                continue
    return json.loads(text or "{}")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp():
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


class Runner:
    """命令执行层；测试里换成假的。"""

    def run(self, argv, input=None, cwd=None, env=None, timeout=None):
        try:
            # 没有输入时给 /dev/null：ssh 默认会读 stdin，脚本经管道交给 bash 执行时会把后面的命令吃掉
            p = subprocess.run(argv, input=input, stdin=None if input is not None else subprocess.DEVNULL, cwd=cwd,
                               env=env, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as e:
            return 127, b"", str(e).encode()
        except subprocess.TimeoutExpired as e:
            return 124, e.stdout or b"", (e.stderr or b"") + b"\n(timeout)"
        return p.returncode, p.stdout, p.stderr

    def spawn(self, argv, cwd=None, env=None):
        subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)

    def which(self, name):
        return shutil.which(name)

    def http(self, method, url, body=None, headers=None, timeout=10):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method, headers=dict(headers or {}))
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:  # noqa: BLE001
            return 0, str(e).encode()


# ------------------------------------------------------------------ pure helpers (unit-tested)
def split_extra(value):
    """EXTRA_CONTAINER_ENV 按 start.sh 的规则拆键值：只读第一行、按空白切、同名后写的生效（位置不变）。"""
    out = {}
    first = (value or "").split("\n", 1)[0]
    for kv in first.split():
        if "=" not in kv or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", kv):
            continue
        k, v = kv.split("=", 1)
        out[k] = v
    return out


def new_pythonpath(current, dir_name):
    parts = [p for p in (current or "").split(":") if p and not OLD_DIR_RE.match(p)]
    mine = "%s/%s" % (CONTAINER_CACHE, dir_name)
    parts = [p for p in parts if p.rstrip("/") != mine]
    return ":".join([mine] + parts)


def strip_blocks(text):
    out, inside = [], False
    for line in text.splitlines(True):
        s = line.rstrip("\r\n")
        if not inside and BEGIN_RE.match(s):
            inside = True
            continue
        if inside:
            if END_RE.match(s):
                inside = False
            continue
        out.append(line)
    body = "".join(out)
    while body.endswith("\n\n"):
        body = body[:-1]
    return body


def make_block(version, pythonpath):
    return ("# >>> yee %s (install.sh) >>>\n"
            "EXTRA_CONTAINER_ENV=\"$EXTRA_CONTAINER_ENV PYTHONPATH=%s YEE=1\"\n"
            "# <<< yee %s <<<\n") % (version, pythonpath, version)


def with_block(text, version, pythonpath):
    body = strip_blocks(text)
    if body and not body.endswith("\n"):
        body += "\n"
    return body + "\n" + make_block(version, pythonpath)


def worker_hosts(env):
    def lst(s):
        return s.replace(",", " ").split()

    if env.get("WORKER_IPS"):
        ips = lst(env["WORKER_IPS"])
    else:
        ips = [env.get("WORKER1_IP") or "10.0.0.2", env.get("WORKER2_IP") or "10.0.0.3"]
    if env.get("WORKER_HOSTS"):
        hosts = lst(env["WORKER_HOSTS"])
    else:
        hosts = [env.get("WORKER%d_HOST" % (i + 1)) or ip for i, ip in enumerate(ips)]
    return hosts


def parse_env0(raw):
    env = {}
    for item in txt(raw).split("\0"):
        if "=" in item:
            k, v = item.split("=", 1)
            env[k] = v
    return env


def mask(k, v):
    return "<已隐藏>" if SECRET_RE.search(k) else v


def dir_name_for(version):
    m = re.match(r"^(\d+)\.(\d+)", version)
    return "yee%s%s" % (m.group(1), m.group(2)) if m else "yee"


def find_recipes(script_dir, extra_roots=None):
    roots = [os.getcwd(), script_dir, os.path.dirname(script_dir)]
    for pat in ("/var/tmp/mia/*", os.path.join(os.path.expanduser("~"), "*"), "/opt/*"):
        roots.extend(sorted(glob.glob(pat)))
    roots.extend(extra_roots or [])
    seen, out = set(), []
    for r in roots:
        try:
            r = os.path.realpath(r)
        except OSError:
            continue
        if r in seen or not os.path.isdir(r):
            continue
        seen.add(r)
        st = os.path.join(r, "start.sh")
        if not os.path.isfile(st):
            continue
        if not (os.path.isfile(os.path.join(r, ".env.tp4")) or os.path.isfile(os.path.join(r, ".env"))):
            continue
        try:
            with open(st, encoding="utf-8", errors="replace") as f:
                if "EXTRA_CONTAINER_ENV" not in f.read():
                    continue
        except OSError:
            continue
        out.append(r)
    return out


def pick_env(recipe, explicit=None):
    """返回 (env 文件, 启动脚本名, 是否方案默认的 env 文件)。"""
    tp4 = os.path.join(recipe, ".env.tp4")
    plain = os.path.join(recipe, ".env")
    has_tp4_launcher = os.path.isfile(os.path.join(recipe, "start-tp4.sh"))
    if explicit:
        path = os.path.abspath(explicit)
        launcher = "start-tp4.sh" if (path.endswith(".tp4") and has_tp4_launcher) else "start.sh"
        default = os.path.realpath(path) == os.path.realpath(tp4 if launcher == "start-tp4.sh" else plain)
        return path, launcher, default
    if os.path.isfile(tp4) and has_tp4_launcher:
        return tp4, "start-tp4.sh", True
    if os.path.isfile(plain):
        return plain, "start.sh", True
    raise Abort("方案目录 %s 里没有 .env.tp4（配 start-tp4.sh）也没有 .env" % recipe)


class Wheel:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.isfile(self.path):
            raise Abort("找不到 wheel：%s" % path)
        with zipfile.ZipFile(self.path) as z:
            names = z.namelist()
            self.names = list(names)
            self.version = ""
            for n in names:
                if n.endswith(".dist-info/METADATA") and n.startswith("yee-"):
                    for line in txt(z.read(n)).splitlines():
                        if line.startswith("Version:"):
                            self.version = line.split(":", 1)[1].strip()
                            break
            if not self.version:
                m = re.match(r"^yee-([0-9][^-]*)-", os.path.basename(self.path))
                self.version = m.group(1) if m else ""
            if not self.version:
                raise Abort("读不出 wheel 的版本号：%s" % self.path)
            if TEMPLATE not in names:
                raise Abort("wheel 里没有 %s" % TEMPLATE)
            self.template = z.read(TEMPLATE)
            self.vcap = {os.path.basename(f): z.read(f) for f in VCAP_FILES if f in names}
        self.sha256 = sha256_file(self.path)

    def stage(self, dest, vcap):
        with zipfile.ZipFile(self.path) as z:
            for n in z.namelist():
                target = os.path.realpath(os.path.join(dest, n))
                if not target.startswith(os.path.realpath(dest) + os.sep):
                    raise Abort("wheel 里有越界路径：%s" % n)
            z.extractall(dest)
        with open(os.path.join(dest, "sitecustomize.py"), "wb") as f:
            f.write(self.template)
        if vcap:
            for name, data in self.vcap.items():
                with open(os.path.join(dest, name), "wb") as f:
                    f.write(data)
        with open(os.path.join(dest, "YEE_VERSION"), "w", encoding="utf-8") as f:
            f.write("yee %s installed by install.sh at %s, wheel sha256 %s, vcap %s\n"
                    % (self.version, _dt.datetime.now().isoformat(timespec="seconds"), self.sha256,
                       "on" if vcap else "off"))
        return {"sitecustomize.py": sha256_bytes(self.template),
                "yee/__init__.py": sha256_file(os.path.join(dest, "yee", "__init__.py"))}


def tar_dir(src):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name in sorted(os.listdir(src)):
            t.add(os.path.join(src, name), arcname=name)
    return buf.getvalue()


# ------------------------------------------------------------------ installer
class Installer:
    PY_ENGINES = ("sglang", "vllm", "trtllm", "lmdeploy", "mlx_lm")
    PATCH_ENGINES = ("llama.cpp", "strata", "maya", "mlx-serve", "cuteafd", "tensorfold")

    def __init__(self, args, runner=None, modules=None):
        self.a = args
        self.r = runner or Runner()
        self.modules = modules
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.notes = []

    # -------------------------------------------------- common
    def find_wheel(self):
        if self.a.wheel:
            return Wheel(self.a.wheel)
        found = sorted(glob.glob(os.path.join(self.script_dir, "yee-*.whl")))
        if len(found) != 1:
            raise Abort("脚本目录里应当正好有 1 个 yee-*.whl，实际 %d 个；用 --wheel 指定" % len(found))
        return Wheel(found[0])

    def confirm(self, what):
        if self.a.yes or not sys.stdin.isatty():
            return True
        try:
            ans = input("[yee-install] %s [y/N] " % what)
        except EOFError:
            return False
        return ans.strip().lower() in ("y", "yes")

    def main(self):
        self.wheel = self.find_wheel()
        say("wheel：%s（版本 %s，sha256 %s）" % (os.path.basename(self.wheel.path), self.wheel.version,
                                                 self.wheel.sha256))
        mode = self.a.mode
        if mode == "auto":
            if self.a.recipe:
                mode = "mia"
            else:
                found = find_recipes(self.script_dir)
                if len(found) > 1:
                    raise Abort("找到多个方案目录，用 --recipe 选一个：%s" % "，".join(found))
                mode = "mia" if found else "generic"
                if found:
                    self.a.recipe = found[0]
        say("模式：%s" % {"mia": "Mia 方案（多机容器）",
                          "generic": "generic（任何部署：本机进程 / docker / 多机 / K8s）",
                          "pip": "pip（当前 Python）"}.get(mode, mode))
        if mode == "pip":
            return self.pip_mode()
        if mode == "generic":
            return self.generic_mode()
        return self.mia_mode()

    # -------------------------------------------------- pip mode
    def pip_mode(self):
        py = self.a.python or sys.executable
        cmds = [[py, "-m", "pip", "install", "--upgrade", self.wheel.path], [py, "-m", "yee", "install-hook"]]
        if self.a.uninstall:
            cmds = [[py, "-m", "yee", "uninstall-hook"], [py, "-m", "pip", "uninstall", "-y", "yee"]]
        for c in cmds:
            say(("将执行：" if self.a.check else "执行：") + " ".join(shlex.quote(x) for x in c))
        if self.a.check:
            return 0
        for c in cmds:
            rc, out, err = self.r.run(c, timeout=1800)
            if rc != 0:
                raise Abort("命令失败（退出码 %d）：%s\n%s" % (rc, " ".join(c), txt(err)[-2000:]))
        if not self.a.uninstall:
            rc, out, err = self.r.run([py, "-m", "yee", "config"], timeout=120)
            if rc == 0:
                print(txt(out))
        say("完成。之后用 %s 启动的推理服务会自动加载 Yee；关闭用 YEE=0。" % py)
        return 0

    # -------------------------------------------------- generic mode
    @property
    def runner(self):
        return self.r

    def load_modules(self):
        """从 wheel 里 import yee.integrations.discover / plan；测试可以传 modules=(discover, plan) 顶掉。"""
        if self.modules:
            return self.modules[0], self.modules[1]
        path = self.wheel.path
        saved_path = list(sys.path)
        for name in [n for n in list(sys.modules) if n == "yee" or n.startswith("yee.")]:
            del sys.modules[name]
        sys.path.insert(0, path)
        try:
            discover = importlib.import_module("yee.integrations.discover")
            plan = importlib.import_module("yee.integrations.plan")
        except Exception as e:  # noqa: BLE001
            names = getattr(self.wheel, "names", [])
            has = [n for n in names
                   if n.startswith("yee/integrations/discover") or n.startswith("yee/integrations/plan")]
            for name in [n for n in list(sys.modules) if n == "yee" or n.startswith("yee.")]:
                del sys.modules[name]
            raise Abort("wheel %s 里没有可用的 yee.integrations.discover / plan（%s%s）；请用完整的 4.3 wheel 重试"
                        % (os.path.basename(path), e,
                           "" if has else "；这个 wheel 里根本没有 yee/integrations/"))
        finally:
            sys.path[:] = saved_path
        return discover, plan

    def collect_instances(self, discover):
        a = self.a
        hosts = []
        for h in (a.hosts or []):
            for one in str(h).replace(";", ",").split(","):
                one = one.strip()
                if one:
                    hosts.append(one)
        try:
            found = discover.collect(cmds=list(a.cmd or []), urls=list(a.url or []), hosts=hosts,
                                     configs=list(a.config or []), use_local=True, use_docker=not a.no_docker,
                                     use_k8s=bool(a.k8s), runner=self.runner)
        except Exception as e:  # noqa: BLE001
            raise Abort("发现部署目标失败：%s" % e)
        return list(found or [])

    def is_local_host(self, host):
        if not host:
            return True
        h = host.split("@")[-1].strip().lower()
        if not h:
            return True
        me = socket.gethostname()
        names = {"local", "localhost", "127.0.0.1", "::1", me.lower(), me.split(".")[0].lower()}
        return h in names or h.split(".")[0] in names

    def remote_argv(self, host, argv):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
                " ".join(shlex.quote(a) for a in argv)]

    def exec_argv(self, host, argv):
        if self.is_local_host(host):
            return list(argv)
        return self.remote_argv(host, argv)

    def docker_python(self, host, container):
        cmd = ["docker", "exec", container, "sh", "-c", "command -v python3 || command -v python"]
        rc, out, err = self.r.run(self.exec_argv(host, cmd), timeout=120)
        if rc != 0:
            return None
        lines = [l.strip() for l in txt(out).splitlines() if l.strip()]
        return lines[0] if lines else None

    def ssh_python(self, host):
        cmd = ["sh", "-c", "command -v python3 || command -v python"]
        rc, out, err = self.r.run(self.remote_argv(host, cmd), timeout=120)
        if rc != 0:
            return None
        lines = [l.strip() for l in txt(out).splitlines() if l.strip()]
        return lines[0] if lines else None

    @staticmethod
    def argv_python(inst):
        """argv[0] of the server when it is an absolute python path (a venv's bin/python stays a venv path)."""
        argv = inst.get("argv") or []
        argv0 = argv[0] if argv else ""
        if argv0.startswith("/") and os.path.basename(argv0).startswith("python"):
            return argv0
        return None

    def local_python(self, inst, pid):
        # argv[0] first: /proc/<pid>/exe resolves a venv's python to the base interpreter, whose pip would install
        # into the wrong site-packages; then the venv / conda prefix of the process; then the exe link.
        py = self.argv_python(inst)
        if py:
            return py
        if pid:
            try:
                with open("/proc/%s/environ" % pid, "rb") as f:
                    penv = dict(kv.split(b"=", 1) for kv in f.read().split(b"\0") if b"=" in kv)
                for k in (b"VIRTUAL_ENV", b"CONDA_PREFIX"):
                    pre = penv.get(k, b"").decode("utf-8", "replace").strip()
                    for name in ("python3", "python"):
                        cand = os.path.join(pre, "bin", name)
                        if pre and os.path.exists(cand):
                            return cand
            except (OSError, ValueError):
                pass
            try:
                link = os.readlink("/proc/%s/exe" % pid)
                if link:
                    return link
            except OSError:
                pass
        argv = inst.get("argv") or []
        argv0 = argv[0] if argv else ""
        if os.path.basename(argv0).startswith("python"):
            return argv0
        return None

    def classify(self, instances):
        """把实例分成三类目标：本机进程 / docker 容器 / ssh 主机；补丁引擎只进报告。"""
        targets, skipped = [], []
        for inst in instances:
            engine = (inst.get("engine") or "").strip().lower()
            where = inst.get("where") or {}
            container = (where.get("container") or "").strip() or None
            host = (where.get("host") or "").strip() or None
            pid = where.get("pid")
            if engine in self.PATCH_ENGINES:
                skipped.append("补丁引擎 %s：不自动装，报告里给补丁与重编命令" % engine)
                continue
            if engine not in self.PY_ENGINES:
                skipped.append("引擎 %s 不在自动挂钩范围（报告里会写支持情况）" % (engine or "未知"))
                continue
            if container:
                py = self.argv_python(inst) or self.docker_python(host, container)
                if not py:
                    skipped.append("容器 %s 里找不到 python3/python" % container)
                    continue
                targets.append({"kind": "docker", "host": None if self.is_local_host(host) else host,
                                "container": container, "py": py, "key": "docker:%s:%s:%s" % (host, container, py),
                                "label": "容器 %s" % container})
                continue
            if host and not self.is_local_host(host):
                py = self.ssh_python(host)
                if not py:
                    skipped.append("%s 上找不到 python3/python" % host)
                    continue
                targets.append({"kind": "ssh", "host": host, "container": None, "py": py,
                                "key": "ssh:%s:%s" % (host, py), "label": "机器 %s" % host})
                continue
            py = self.local_python(inst, pid)
            if not py:
                skipped.append("本机进程 %s 找不到解释器" % (pid or "?"))
                continue
            targets.append({"kind": "local", "host": None, "container": None, "py": py, "key": "local:%s" % py,
                            "label": "本机进程 %s" % (pid or "?")})
        return targets, skipped

    def target_argv(self, t, argv):
        """在目标上执行一条命令：容器走 docker exec，远程机器走 ssh，本机直接跑。"""
        if t["kind"] == "docker":
            return self.exec_argv(t["host"], ["docker", "exec", t["container"]] + list(argv))
        return self.exec_argv(t["host"], list(argv))

    def install_target(self, t):
        name = os.path.basename(self.wheel.path)
        remote_path = "/tmp/" + name
        host = t["host"]
        if t["kind"] == "docker":
            rc, out, err = self.r.run(self.exec_argv(host, ["docker", "cp", self.wheel.path,
                                                             "%s:%s" % (t["container"], remote_path)]),
                                       timeout=900)
            if rc != 0:
                say("docker cp 到 %s 失败：%s" % (t["container"], txt(err).strip()[-300:]))
                return False
            inst_path = remote_path
        elif t["kind"] == "ssh":
            rc, out, err = self.r.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", self.wheel.path,
                                       "%s:%s" % (host, remote_path)], timeout=900)
            if rc != 0:
                say("scp 到 %s 失败：%s" % (host, txt(err).strip()[-300:]))
                return False
            inst_path = remote_path
        else:
            inst_path = self.wheel.path
        cmds = [[t["py"], "-m", "pip", "install", "--no-deps", "--force-reinstall", inst_path],
                [t["py"], "-m", "yee", "install-hook"],
                [t["py"], "-m", "yee", "config", "--json"]]
        for i, c in enumerate(cmds):
            rc, out, err = self.r.run(self.target_argv(t, c), timeout=1800)
            if rc != 0:
                say("命令失败（退出码 %d）：%s %s" % (rc, " ".join(c), txt(err).strip()[-300:]))
                return False
            if i == len(cmds) - 1:
                try:
                    json_tail(txt(out))
                except ValueError:
                    say("yee config --json 的返回解析不了：%s" % txt(out).strip()[-300:])
                    return False
        return True

    def uninstall_target(self, t):
        ok = True
        for c in ([t["py"], "-m", "yee", "uninstall-hook"], [t["py"], "-m", "pip", "uninstall", "-y", "yee"]):
            rc, out, err = self.r.run(self.target_argv(t, c), timeout=900)
            if rc != 0:
                say("卸载命令失败（退出码 %d）：%s" % (rc, " ".join(c)))
                ok = False
        return ok

    def say_container_persist(self, t):
        name = os.path.basename(self.wheel.path)
        say("容器 %s 重建以后会丢，持久做法二选一：" % t["container"])
        say("  1) 镜像里装：构建镜像时 pip install %s && python -m yee install-hook" % name)
        say("  2) 挂载 + 环境变量：-v /opt/yee:/opt/yee -e PYTHONPATH=/opt/yee")

    def restart_generic(self, targets):
        seen, containers = set(), []
        for t in targets:
            if t["kind"] != "docker":
                continue
            key = (t["host"], t["container"])
            if key in seen:
                continue
            seen.add(key)
            containers.append(t)
        if containers:
            if not self.confirm("现在重启这些容器：%s？" % "、".join(t["container"] for t in containers)):
                say("已取消重启")
                return
            for t in containers:
                rc, out, err = self.r.run(self.exec_argv(t["host"], ["docker", "restart", t["container"]]),
                                           timeout=900)
                say("docker restart %s：退出码 %d" % (t["container"], rc))
        if self.a.restart_cmd:
            argv = shlex.split(self.a.restart_cmd)
            if argv:
                rc, out, err = self.r.run(argv, timeout=1800)
                say("--restart-cmd 执行完毕（退出码 %d）：%s" % (rc, self.a.restart_cmd))
                if txt(out).strip():
                    say(txt(out).strip()[-500:])
        others = [t for t in targets if t["kind"] != "docker"]
        if others:
            say("其余目标用你原来的方式重启该服务，然后运行 python -m yee doctor：%s"
                % "、".join(t["label"] for t in others))

    def generic_mode(self):
        discover, plan_mod = self.load_modules()
        instances = self.collect_instances(discover)
        say("发现 %d 个实例" % len(instances))
        targets, failures = [], []
        if not self.a.check:
            targets, skipped = self.classify(instances)
            for why in skipped:
                say("跳过：" + why)
            done = set()
            for t in targets:
                if t["key"] in done:
                    continue
                done.add(t["key"])
                say("目标：%s（解释器 %s）" % (t["label"], t["py"]))
                if self.a.uninstall:
                    ok = self.uninstall_target(t)
                else:
                    ok = self.install_target(t)
                    if ok and t["kind"] == "docker":
                        self.say_container_persist(t)
                if not ok:
                    failures.append(t["label"])
        plan = plan_mod.analyze(instances)
        print(plan_mod.render(plan))
        out_path = self.a.plan_out or "yee-plan.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
            say("诊断结果已写 %s" % os.path.abspath(out_path))
        except OSError as e:
            say("写 %s 失败：%s" % (out_path, e))
        if self.a.restart and not self.a.check:
            self.restart_generic(targets)
            again = self.collect_instances(discover)
            plan = plan_mod.analyze(again)
            summary = plan.get("summary") or {}
            say("重启后再看：可加速 %s · 要改参数 %s · 要打补丁 %s · 不支持 %s · 无需处理 %s"
                % (summary.get("accelerated", 0), summary.get("needs_change", 0), summary.get("needs_patch", 0),
                   summary.get("unsupported", 0), summary.get("not_needed", 0)))
        if failures:
            say("有 %d 个目标没装成功：%s" % (len(failures), "、".join(failures)))
            return 1
        try:
            return int(plan.get("exit", 0))
        except (TypeError, ValueError):
            return 0

    # -------------------------------------------------- mia mode: discovery
    def source_env(self):
        script = 'set -a; source "$1" >/dev/null 2>&1; rc=$?; set +a; env -0; exit $rc'
        rc, out, err = self.r.run(["bash", "-c", script, "yee-install", self.env_file], cwd=self.recipe,
                                  env=dict(os.environ), timeout=60)
        if not out:
            raise Abort("bash 读取 %s 失败：%s" % (self.env_file, txt(err)[-500:]))
        if rc != 0:
            self.notes.append("source %s 的退出码是 %d（start.sh 同样会读到这些值）" % (self.env_file, rc))
        return parse_env0(out)

    def ssh_argv(self, host):
        a = ["ssh", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=15"]
        if self.identity and os.path.isfile(self.identity):
            a[1:1] = ["-i", self.identity]
        return a + [("%s@%s" % (self.user, host)) if self.user else host]

    def on(self, host, script, input=None, timeout=600):
        """host=None 表示本机（head）。"""
        if host is None:
            return self.r.run(["bash", "-c", script], input=input, timeout=timeout)
        return self.r.run(self.ssh_argv(host) + ["bash -c %s" % shlex.quote(script)], input=input, timeout=timeout)

    def discover(self):
        self.recipe = os.path.realpath(self.a.recipe)
        if not os.path.isfile(os.path.join(self.recipe, "start.sh")):
            raise Abort("%s 不是 Mia 方案目录（没有 start.sh）" % self.recipe)
        self.env_file, self.launcher, self.env_is_default = pick_env(self.recipe, self.a.env_file)
        env = self.source_env()
        self.env = env
        self.hosts = worker_hosts(env)
        self.user = env["WORKER_USER"] if "WORKER_USER" in env else "zurih"
        ident = env.get("SSH_IDENTITY") or os.path.join(os.path.expanduser("~"), ".ssh", "id_ed25519_shared")
        self.identity = ident if os.path.isabs(ident) else os.path.join(self.recipe, ident)
        self.image = env.get("IMAGE") or "dsv41-3x-spark:local"
        self.port = env.get("PORT") or "8888"
        self.head_ctn = env.get("HEAD_CTN") or "dsv41-head"
        sd = env.get("STATE_DIR") or ("state-tp4" if self.launcher == "start-tp4.sh" else "state")
        self.state_dir = sd if os.path.isabs(sd) else os.path.join(self.recipe, sd)
        self.extra = split_extra(env.get("EXTRA_CONTAINER_ENV", ""))
        self.dir_name = self.a.dir_name or dir_name_for(self.wheel.version)
        say("方案目录：%s；env 文件：%s；启动脚本：%s" % (self.recipe, self.env_file, self.launcher))
        say("镜像：%s；端口：%s；head 容器：%s；worker：%s（用户 %s）"
            % (self.image, self.port, self.head_ctn, " ".join(self.hosts) or "无", self.user or "ssh 配置决定"))

        ok = True
        self.cache_root = {}
        if self.a.cache_root_override:
            self.cache_root[None] = self.a.cache_root_override.rstrip("/")
        else:
            self.cache_root[None] = os.path.join(os.path.expanduser("~"), ".cache")
        for h in self.hosts:
            rc, out, err = self.on(h, 'printf %s "$HOME"', timeout=60)
            home = txt(out).strip()
            if rc != 0 or not home.startswith("/"):
                say("worker %s 连不上：%s" % (h, (txt(err).strip() or "退出码 %d" % rc)[-300:]))
                ok = False
                continue
            self.cache_root[h] = self.a.cache_root_override.rstrip("/") if self.a.cache_root_override \
                else home.rstrip("/") + "/.cache"
            say("worker %s 连通，安装目录 %s/%s" % (h, self.cache_root[h], self.dir_name))
        say("head 安装目录 %s/%s（容器内 %s/%s）" % (self.cache_root[None], self.dir_name, CONTAINER_CACHE,
                                                   self.dir_name))

        # 镜像与 PYTHONPATH
        rc, out, err = self.r.run(["docker", "image", "inspect", "--format", "{{.Id}} {{json .Config.Env}}",
                                   self.image], timeout=60)
        self.image_pp = None
        self.image_id = None
        if rc == 0:
            head = txt(out).strip()
            self.image_id, _, envjson = head.partition(" ")
            try:
                for kv in json.loads(envjson or "[]"):
                    if kv.startswith("PYTHONPATH="):
                        self.image_pp = kv.split("=", 1)[1]
            except ValueError:
                pass
            say("镜像 PYTHONPATH：%s" % (self.image_pp if self.image_pp is not None else "（镜像没设）"))
        else:
            say("head 上读不到镜像 %s：%s" % (self.image, txt(err).strip()[-300:]))
            ok = False
        self.same_image = True
        for h in self.hosts:
            if h not in self.cache_root:
                continue
            rc, out, err = self.on(h, "docker image inspect --format '{{.Id}}' %s" % shlex.quote(self.image), timeout=60)
            wid = txt(out).strip()
            if rc != 0:
                say("worker %s 上没有镜像 %s" % (h, self.image))
                ok = False
            elif self.image_id and wid != self.image_id:
                self.same_image = False
                say("worker %s 的镜像 ID 与 head 不同（%s ≠ %s；各台本机构建时属正常）" % (h, wid[:19], self.image_id[:19]))

        cur = self.extra.get("PYTHONPATH")
        if cur is None:
            cur = self.a.base_pythonpath if self.a.base_pythonpath is not None else self.image_pp
        if cur is None:
            cur = ""
            self.notes.append("镜像没有 PYTHONPATH，EXTRA_CONTAINER_ENV 里也没有；新 PYTHONPATH 只含 Yee 目录"
                              "（需要时用 --base-pythonpath 指定）")
        self.cur_pp = cur
        self.new_pp = new_pythonpath(cur, self.dir_name)
        say("当前容器 PYTHONPATH：%s" % (cur or "（空）"))
        say("安装后容器 PYTHONPATH：%s" % self.new_pp)

        # vcap
        self.vcap = False
        if self.a.no_vcap:
            self.vcap_reason = "--no-vcap"
        elif not self.wheel.vcap:
            self.vcap_reason = "wheel 里没有 vcap 文件"
        else:
            # 每台都核对镜像里的原件（各台通常各自构建镜像，镜像 ID 不同但内容相同）
            probe = "docker run --rm --network none --entrypoint sha256sum %s %s" % (
                shlex.quote(self.image), ADAPTER_VERIFY_CAP)
            self.vcap, self.vcap_reason = True, "每台镜像里的 verify_cap.py 都与补丁所基于的原件一致"
            for h in [None] + [x for x in self.hosts if x in self.cache_root]:
                rc, out, err = self.on(h, probe, timeout=300)
                got = txt(out).split()[0] if rc == 0 and txt(out).split() else ""
                name = "head" if h is None else h
                if got == VCAP_ORIG_SHA256:
                    continue
                self.vcap = False
                if got:
                    self.vcap_reason = "%s 镜像里的 verify_cap.py 与补丁原件不同（%s…），不放 vcap" % (name, got[:12])
                else:
                    self.vcap_reason = "读不到 %s 镜像里的 %s（%s）" % (name, ADAPTER_VERIFY_CAP, txt(err).strip()[-200:])
                break
        say("逐请求 conf（vcap）：%s，%s" % ("放" if self.vcap else "不放", self.vcap_reason))

        explicit = [(k, v) for k, v in self.extra.items() if YEE_KEY_RE.match(k) and k != "YEE"]
        if explicit:
            say("EXTRA_CONTAINER_ENV 里显式设置的 Yee 相关项会覆盖 4.3 的默认值：%s"
                % " ".join("%s=%s" % (k, mask(k, v)) for k, v in explicit))
        if self.extra.get("YEE") not in (None, "", "1"):
            say("EXTRA_CONTAINER_ENV 里有 YEE=%s，标记块里的 YEE=1 会覆盖它" % self.extra.get("YEE"))
        return ok

    # -------------------------------------------------- mia mode: actions
    def install_files(self):
        tmp = tempfile.mkdtemp(prefix="yee-install-")
        try:
            want = self.wheel.stage(tmp, self.vcap)
            blob = tar_dir(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        ts = stamp()
        for h in [None] + [x for x in self.hosts]:
            base = "%s/%s" % (self.cache_root[h], self.dir_name)
            script = ('set -e; B=%s; rm -rf "$B.new"; mkdir -p "$B.new"; tar -xzf - -C "$B.new"; '
                      'for o in "$B".old-*; do if [ -e "$o" ]; then rm -rf "$o"; fi; done; '
                      'if [ -e "$B" ]; then mv "$B" "$B.old-%s"; fi; mv "$B.new" "$B"; '
                      'sha256sum "$B/sitecustomize.py" "$B/yee/__init__.py"') % (shlex.quote(base), ts)
            rc, out, err = self.on(h, script, input=blob, timeout=600)
            name = "head" if h is None else h
            if rc != 0:
                raise Abort("%s 安装失败（退出码 %d）：%s" % (name, rc, txt(err).strip()[-500:]))
            got = {}
            for line in txt(out).splitlines():
                parts = line.split()
                if len(parts) == 2:
                    got["yee/__init__.py" if parts[1].endswith("yee/__init__.py") else "sitecustomize.py"] = parts[0]
            if got != want:
                raise Abort("%s 上文件校验不一致：%s ≠ %s" % (name, got, want))
            say("%s 已装到 %s（sitecustomize.py %s…）" % (name, base, want["sitecustomize.py"][:12]))

    def purge_files(self):
        for h in [None] + list(self.hosts):
            if h not in self.cache_root:
                continue
            base = "%s/%s" % (self.cache_root[h], self.dir_name)
            rc, out, err = self.on(h, 'B=%s; rm -rf "$B" "$B.new" "$B".old-*' % shlex.quote(base), timeout=300)
            say("%s：%s %s" % ("head" if h is None else h, "已删除" if rc == 0 else "删除失败", base))

    def write_env(self, install):
        with open(self.env_file, encoding="utf-8") as f:
            old = f.read()
        if install:
            new = with_block(old, self.wheel.version, self.new_pp)
        else:
            new = strip_blocks(old)
            if new and not new.endswith("\n"):
                new += "\n"
        if new == old:
            say("env 文件不用改")
            return None
        backup = "%s.bak-yee-%s" % (self.env_file, stamp())
        shutil.copyfile(self.env_file, backup)
        with open(self.env_file, "w", encoding="utf-8") as f:
            f.write(new)
        env = self.source_env()
        extra = split_extra(env.get("EXTRA_CONTAINER_ENV", ""))
        good = (extra.get("PYTHONPATH") == self.new_pp and extra.get("YEE") == "1") if install else \
            not any(BEGIN_RE.match(l) for l in new.splitlines())
        if not good:
            shutil.copyfile(backup, self.env_file)
            raise Abort("写入后核对失败，已从备份恢复 env 文件（PYTHONPATH=%s YEE=%s）"
                        % (extra.get("PYTHONPATH"), extra.get("YEE")))
        say("env 文件已%s（备份 %s）" % ("追加 Yee 标记块" if install else "删除 Yee 标记块", backup))
        return backup

    # -------------------------------------------------- restart
    def vcap_ctl_wanted(self):
        """vcap 缺省打开：写 head 的 state 目录下的控制文件（容器内 /state/vcap_ctl.json）。测试模式不写。"""
        return self.vcap and self.env_is_default and not self.a.cache_root_override

    def write_vcap_ctl(self):
        ctl = os.path.join(self.state_dir, "vcap_ctl.json")
        mark = ctl + ".yee-install"
        if os.path.exists(ctl):
            say("vcap 控制文件已存在（%s），保留原样" % ctl)
            return
        body = json.dumps({"mode": "adapt"}) + "\n"
        os.makedirs(self.state_dir, exist_ok=True)
        with open(ctl, "w", encoding="utf-8") as f:
            f.write(body)
        with open(mark, "w", encoding="utf-8") as f:
            f.write(body)
        say("已写 vcap 控制文件 %s（{\"mode\": \"adapt\"}，rank0 每 0.5 秒读一次，不用重启）" % ctl)

    def remove_vcap_ctl(self):
        ctl = os.path.join(self.state_dir, "vcap_ctl.json")
        mark = ctl + ".yee-install"
        if not os.path.isfile(mark):
            return
        try:
            with open(mark, encoding="utf-8") as f:
                mine = f.read()
            cur = open(ctl, encoding="utf-8").read() if os.path.isfile(ctl) else None
        except OSError:
            return
        if cur == mine:
            os.remove(ctl)
            say("已删除安装时写的 vcap 控制文件 %s" % ctl)
        else:
            say("vcap 控制文件 %s 在安装后被改过，保留原样" % ctl)
        os.remove(mark)

    def api_headers(self):
        key = self.env.get("API_KEY", "")
        if not key:
            p = os.path.join(self.state_dir, "api-key")
            if os.path.isfile(p):
                with open(p, encoding="utf-8", errors="replace") as f:
                    key = f.read().strip()
        return {"Authorization": "Bearer %s" % key} if key else {}

    def healthy(self):
        code, _ = self.r.http("GET", "http://127.0.0.1:%s/health" % self.port, headers=self.api_headers(), timeout=5)
        return code == 200

    def smoke(self):
        h = self.api_headers()
        code, body = self.r.http("GET", "http://127.0.0.1:%s/v1/models" % self.port, headers=h, timeout=20)
        if code != 200:
            return False, "/v1/models 返回 %s" % code
        try:
            model = json.loads(txt(body))["data"][0]["id"]
        except (ValueError, KeyError, IndexError, TypeError):
            return False, "/v1/models 的返回读不出模型名"
        req = {"model": model, "messages": [{"role": "user", "content": "What is 19 + 23?"}], "max_tokens": 16,
               "temperature": 0, "chat_template_kwargs": {"thinking": False}}
        code, body = self.r.http("POST", "http://127.0.0.1:%s/v1/chat/completions" % self.port, body=req,
                                 headers=h, timeout=300)
        if code != 200:
            return False, "chat 请求返回 %s：%s" % (code, txt(body)[:200])
        try:
            json.loads(txt(body))["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError):
            return False, "chat 返回里没有 choices"
        return True, "模型 %s 冒烟通过" % model

    def stop(self):
        rc, out, err = self.r.run(["bash", os.path.join(self.recipe, self.launcher), "stop"], cwd=self.recipe,
                                  timeout=900)
        say("stop 退出码 %d" % rc)

    def serve(self):
        ts = stamp()
        log = os.path.join(self.recipe, "yee-install-%s.log" % ts)
        n = 1
        while os.path.exists(log) or os.path.exists(log + ".rc"):
            n += 1
            log = os.path.join(self.recipe, "yee-install-%s-%d.log" % (ts, n))
        rcf = log + ".rc"
        inner = "cd %s && ./%s serve > %s 2>&1; echo $? > %s" % (
            shlex.quote(self.recipe), shlex.quote(self.launcher), shlex.quote(log), shlex.quote(rcf))
        started = False
        if (self.r.which("systemd-run") and hasattr(os, "geteuid") and os.geteuid() == 0
                and not os.environ.get("YEE_INSTALL_NO_SYSTEMD")):
            rc, out, err = self.r.run(["systemd-run", "--unit", "yee-serve-%s" % os.path.basename(log)[len("yee-install-"):-4],
                                       "--working-directory=%s" % self.recipe,
                                       "-E", "HOME=%s" % os.path.expanduser("~"), "-E", "PATH=%s" % os.environ.get(
                                           "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
                                       "/bin/bash", "-c", inner], timeout=60)
            started = rc == 0
        if not started:
            self.r.spawn(["setsid", "bash", "-c", inner] if self.r.which("setsid") else ["bash", "-c", inner],
                         cwd=self.recipe, env=dict(os.environ))
        say("已在后台启动 serve，日志 %s" % log)
        return log, rcf

    def wait_ready(self, log, rcf):
        deadline = time.time() + float(self.a.timeout_min) * 60
        t0 = time.time()
        last = 0
        done_at = None
        while time.time() < deadline:
            rc = None
            if os.path.isfile(rcf):
                try:
                    with open(rcf) as f:
                        rc = int(f.read().strip() or "0")
                except (OSError, ValueError):
                    rc = None
            if rc is not None and rc != 0:
                return False, "serve 退出码 %d，见 %s" % (rc, log)
            if rc == 0:
                if self.healthy():
                    return True, "serve 已完成、/health 200（%.1f 分钟）" % ((time.time() - t0) / 60)
                done_at = done_at or time.time()
                if time.time() - done_at > 180:
                    return False, "serve 已退出但 /health 3 分钟内没有 200"
            if time.time() - last >= 60:
                last = time.time()
                say("等待服务就绪…已过 %.0f 分钟" % ((time.time() - t0) / 60))
            time.sleep(float(os.environ.get("YEE_INSTALL_POLL_S", "10")))
        return False, "等待超时（%s 分钟），见 %s" % (self.a.timeout_min, log)

    def check_logs(self):
        if not self.r.which("docker"):
            return "unknown", ["本机没有 docker 命令，跳过日志检查"]
        rc, out, err = self.r.run(["docker", "logs", "--tail", "20000", self.head_ctn], timeout=120)
        lines = (txt(out) + "\n" + txt(err)).splitlines()
        picked, seen, bad, steps = [], set(), [], None
        tracebacks = 0
        for line in lines:
            s = line.strip()
            if "Traceback" in s:
                tracebacks += 1
            if any(m in s for m in FAIL_MARKERS):
                bad.append(s)
            if not any(m in s for m in LOG_MARKERS):
                continue
            if s.startswith("[recall] steps="):
                steps = s
                continue
            key = s[:60]
            if key not in seen and len(picked) < 25:
                seen.add(key)
                picked.append(s)
        if steps:
            picked.append(steps)
        if tracebacks:
            picked.append("（head 容器日志里有 %d 处 Traceback，请看 docker logs %s）" % (tracebacks, self.head_ctn))
        if bad:
            return "inactive", picked + ["未生效：" + b for b in bad[:5]]
        if any(s.startswith("[recall] patched") or "rank check on" in s or "[yee] " in s for s in picked):
            return "active", picked
        return "unknown", picked + ["日志里没有看到 Yee 的启动行"]

    def restart_cycle(self):
        if not self.env_is_default:
            raise Abort("--restart 只支持方案自己的 env 文件（%s 读的是方案目录下的默认文件）" % self.launcher)
        self.stop()
        log, rcf = self.serve()
        ok, why = self.wait_ready(log, rcf)
        if ok:
            ok2, why2 = self.smoke()
            why = why + "；" + why2
            ok = ok2
        return ok, why

    # -------------------------------------------------- mia mode: flow
    def mia_mode(self):
        reachable = self.discover()
        for n in self.notes:
            say("提示：" + n)
        if self.a.check:
            if not self.a.uninstall:
                say("将在 env 文件末尾写入：\n" + make_block(self.wheel.version, self.new_pp).rstrip())
                if self.vcap_ctl_wanted():
                    say("将写 vcap 控制文件 %s（已存在则保留）" % os.path.join(self.state_dir, "vcap_ctl.json"))
            say("检查%s" % ("通过，可以安装" if reachable else "未通过，先解决上面的问题"))
            return 0 if reachable else 1
        if not reachable:
            raise Abort("有主机连不上或缺镜像，没有改动任何东西")
        if self.a.restart and not self.env_is_default:
            raise Abort("--restart 只支持方案自己的 env 文件（%s 读的是方案目录下的默认文件），没有改动任何东西"
                        % self.launcher)
        if self.a.uninstall:
            if not self.confirm("从 %s 删除 Yee 标记块？" % self.env_file):
                raise Abort("已取消")
            backup = self.write_env(False)
            self.remove_vcap_ctl()
            if self.a.purge:
                self.purge_files()
        else:
            self.install_files()
            if not self.confirm("把 Yee 标记块写进 %s？" % self.env_file):
                raise Abort("已取消（文件已装好，env 未改）")
            backup = self.write_env(True)
            if self.vcap_ctl_wanted():
                self.write_vcap_ctl()
        status = "not-restarted"
        if self.a.restart:
            if not self.confirm("现在重启服务（会中断服务约 10 分钟）？"):
                raise Abort("已取消重启")
            ok, why = self.restart_cycle()
            say(("重启成功：" if ok else "重启失败：") + why)
            if not ok:
                if backup:
                    shutil.copyfile(backup, self.env_file)
                    say("已从 %s 恢复 env 文件，按原配置重启" % backup)
                    ok2, why2 = self.restart_cycle()
                    say(("回退成功：" if ok2 else "回退后仍失败：") + why2)
                raise Abort("重启失败，已回退", code=3)
            if not self.a.uninstall:
                status, lines = self.check_logs()
                for l in lines:
                    say("日志：" + l)
            else:
                status = "uninstalled"
        self.summary(status)
        return 2 if status == "inactive" else 0

    def summary(self, status):
        say("—— 汇总 ——")
        if self.a.uninstall:
            say("已卸载：env 文件里没有 Yee 标记块%s" % ("，各台安装目录已删除" if self.a.purge else ""))
        else:
            say("Yee %s 已装到 %d 台机器：head%s" % (self.wheel.version, 1 + len(self.hosts),
                                                     "".join("、" + h for h in self.hosts)))
            say("容器 PYTHONPATH：%s；vcap：%s" % (self.new_pp, "放" if self.vcap else "不放"))
        if status == "not-restarted":
            say("还没重启。生效需要重启：cd %s && ./%s stop && ./%s serve（或 bash install.sh %s--restart）"
                % (self.recipe, self.launcher, self.launcher, "--uninstall " if self.a.uninstall else ""))
        elif status == "active":
            say("服务已重启，Yee 已生效。看开关：docker exec %s python3 -m yee config" % self.head_ctn)
        elif status == "inactive":
            say("服务已重启，但 Yee 自己关掉了（见上面的日志行），服务按原样运行")
        elif status == "unknown":
            say("服务已重启；没能从日志确认 Yee 状态，用 docker logs %s | grep -E '\\[yee|\\[recall' 查看" % self.head_ctn)
        if not self.a.uninstall:
            say("回退：bash install.sh --uninstall --restart")


def build_parser():
    p = argparse.ArgumentParser(prog="install.sh", description="Yee 一键安装（4.3）：Mia 方案或任何部署都能装")
    p.add_argument("--recipe", help="Mia 方案目录（含 start.sh 与 .env.tp4 / .env）")
    p.add_argument("--env-file", help="env 文件（缺省：方案目录下的 .env.tp4 或 .env）")
    p.add_argument("--wheel", help="yee wheel（缺省：脚本目录里唯一的 yee-*.whl）")
    p.add_argument("--check", action="store_true", help="只检查/诊断，不写任何东西")
    p.add_argument("--restart", action="store_true", help="装完重启服务（generic 模式重启 docker 容器），再诊断一次")
    p.add_argument("--uninstall", action="store_true", help="卸载（删掉 env 文件里的 Yee 标记块 / 卸载各目标）")
    p.add_argument("--purge", action="store_true", help="与 --uninstall 一起用：再删掉各台机器的安装目录")
    p.add_argument("--no-vcap", action="store_true", help="不放逐请求 conf（mia-dsv41-vcap）的两个文件")
    p.add_argument("--dir-name", help="安装目录名（缺省 yee+主次版本号，如 yee43）")
    p.add_argument("--cache-root-override", help="测试用：各台机器都装到这个路径下，而不是 $HOME/.cache")
    p.add_argument("--base-pythonpath", help="镜像 PYTHONPATH 读不到时用这个值")
    p.add_argument("--yes", action="store_true", help="不询问")
    p.add_argument("--mode", choices=("auto", "mia", "generic", "pip"), default="auto",
                   help="auto：找到 Mia 方案走 Mia 模式，否则走 generic 模式；pip：只装进当前 Python")
    p.add_argument("--python", help="pip 模式用的 Python（缺省当前解释器）")
    p.add_argument("--timeout-min", default="40", help="重启后等待就绪的上限（分钟，缺省 40）")
    p.add_argument("--pip", action="store_true", help="强制走老路径：只装进当前 Python（generic 模式不装）")
    p.add_argument("--cmd", action="append", help="generic 模式：按这条启动命令解析实例（可多次）")
    p.add_argument("--url", action="append", help="generic 模式：探测这个服务地址（可多次）")
    p.add_argument("--hosts", action="append", help="generic 模式：逗号分隔的机器列表（如 h1,h2），ssh 过去看/装")
    p.add_argument("--config", action="append", help="generic 模式：Maya JSON 启动配置（可多次）")
    p.add_argument("--no-docker", action="store_true", help="generic 模式：不看 docker 容器")
    p.add_argument("--k8s", action="store_true", help="generic 模式：读 kubectl（只看，不改集群）")
    p.add_argument("--restart-cmd", help="generic 模式：--restart 时在本机执行的重启命令")
    p.add_argument("--plan-out", help="generic 模式：诊断 JSON 的输出路径（缺省 ./yee-plan.json）")
    return p


def main(argv=None, runner=None, modules=None):
    args = build_parser().parse_args(argv)
    if args.pip and args.mode in ("auto", "generic"):
        args.mode = "pip"
    if args.mode == "mia" and not args.recipe:
        found = find_recipes(os.path.dirname(os.path.abspath(__file__)))
        if len(found) != 1:
            say("错误：--mode mia 需要 --recipe（自动找到 %d 个方案目录）" % len(found))
            return 1
        args.recipe = found[0]
    try:
        return Installer(args, runner, modules=modules).main()
    except Abort as e:
        say("错误：%s" % e)
        return e.code
    except KeyboardInterrupt:
        say("已中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
