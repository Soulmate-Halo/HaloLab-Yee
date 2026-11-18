"""找到跑着的推理服务（Yee 4.3.0 doctor 的发现层）。

本机进程 / docker 容器 / K8s Pod / ssh 远端 / HTTP 探测 / Maya JSON 配置 → 实例 dict。
只用标准库；任何异常都不抛到调用方，命令不存在（没有 docker/kubectl）就跳过。
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import urllib.error
import urllib.request

try:
    from . import launch as _launch
except Exception:  # pragma: no cover - 兜底
    _launch = None


# --------------------------------------------------------------------------- runner
class SubprocessRunner:
    """默认 runner：真的跑命令。命令不存在返回 rc=127，不抛。"""

    def run(self, argv, input=None, timeout=None):
        try:
            p = subprocess.run(list(argv), input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=timeout)
            return p.returncode, p.stdout, p.stderr
        except FileNotFoundError as e:
            return 127, b"", str(e).encode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            return 124, b"", b"timeout"
        except Exception as e:  # noqa: BLE001
            return 1, b"", str(e).encode("utf-8", "replace")


def _runner(runner):
    return runner if runner is not None else SubprocessRunner()


def _run(runner, argv, input=None, timeout=None):
    """跑一条命令 → (rc, out_text, err_text)；出异常就当失败。"""
    try:
        rc, out, err = runner.run(list(argv), input=input, timeout=timeout)
    except TypeError:
        try:
            rc, out, err = runner.run(list(argv))
        except Exception:  # noqa: BLE001
            return 1, "", ""
    except Exception:  # noqa: BLE001
        return 1, "", ""
    if isinstance(out, bytes):
        out = out.decode("utf-8", "replace")
    if isinstance(err, bytes):
        err = err.decode("utf-8", "replace")
    try:
        rc = int(rc)
    except Exception:
        rc = 1
    return rc, out or "", err or ""


def _text(data):
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return str(data or "")


# --------------------------------------------------------------------------- 本机
def _proc_list():
    try:
        return [p for p in os.listdir("/proc") if p.isdigit()]
    except Exception:
        return []


def _read_proc(pid, name):
    try:
        with open("/proc/%s/%s" % (pid, name), "rb") as f:
            return f.read()
    except Exception:
        return b""


def _ppid(pid):
    """父进程 pid（读不到返回 None）。"""
    try:
        raw = _read_proc(pid, "status").decode("utf-8", "replace")
        for ln in raw.splitlines():
            if ln.startswith("PPid:"):
                return int(ln.split(":", 1)[1].strip())
    except Exception:
        pass
    return None


def local(include_env=True):
    """本机（Linux /proc）上的推理服务进程。非 Linux 返回 []。"""
    out = []
    try:
        if not os.path.isdir("/proc"):
            return out
    except Exception:
        return out
    seen_argv = {}
    rows = []
    mine = {os.getpid()}  # this process and its ancestors (yee doctor / install.sh carry --cmd text in their argv)
    try:
        p = os.getppid()
        for _ in range(8):
            if not p or p in mine:
                break
            mine.add(p)
            p = _ppid(p)
    except Exception:
        pass
    for pid in _proc_list():
        if int(pid) in mine:
            continue
        raw = _read_proc(pid, "cmdline")
        if not raw:
            continue
        argv = [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]
        if not argv:
            continue
        if any(a.endswith("yee_install.py") for a in argv) or \
                any(a == "-m" and i + 1 < len(argv) and argv[i + 1] in ("yee", "recall_accel")
                    for i, a in enumerate(argv)):
            continue  # Yee's own tools
        env = {}
        if include_env:
            rawenv = _read_proc(pid, "environ")
            if rawenv:
                for item in rawenv.split(b"\0"):
                    if b"=" in item:
                        k, v = item.split(b"=", 1)
                        env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
        rows.append((int(pid), argv, env))
    for pid, argv, env in rows:
        try:
            inst = _launch.parse_launch(argv, env) if _launch is not None else None
        except Exception:
            inst = None
        if not inst:
            continue
        key = " ".join(argv)
        if key in seen_argv:
            continue  # 同一服务的子进程（argv 相同）：只留第一条（父进程）
        seen_argv[key] = pid
        inst["where"]["pid"] = pid
        inst["where"]["host"] = _hostname()
        inst["where"]["source"] = "local"
        cg = _read_proc(pid, "cgroup").decode("utf-8", "replace")
        if any(k in cg for k in ("docker", "containerd", "kubepods")):
            inst["notes"].append("在容器内")
        ppid = _ppid(pid)
        if ppid is not None and ppid in seen_argv.values():
            continue  # 父进程已经在列表里
        out.append(inst)
    return out


def _hostname():
    try:
        return socket.gethostname()
    except Exception:
        return ""


# --------------------------------------------------------------------------- docker
def _docker_ps(runner):
    rc, out, _err = _run(runner, ["docker", "ps", "--format", "{{json .}}"])
    if rc != 0:
        return []
    rows = []
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        name = obj.get("Names") or obj.get("ID") or ""
        if name:
            rows.append((str(name), obj))
    return rows


def _docker_top(runner, name):
    rc, out, _err = _run(runner, ["docker", "top", name, "-eo", "pid,args"])
    if rc != 0:
        return []
    rows = []
    for ln in out.splitlines()[1:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split(None, 1)
        if len(parts) != 2:
            continue
        pid, args = parts[0], parts[1]
        if not pid.isdigit():
            continue
        rows.append((int(pid), args))
    return rows


def _docker_env(runner, name):
    """docker inspect → {"env": {...}, "notes": [...]}。"""
    env = {}
    notes = []
    rc, out, _err = _run(runner, ["docker", "inspect", name])
    if rc != 0:
        return env, notes
    try:
        data = json.loads(out)
    except Exception:
        return env, notes
    if not isinstance(data, list) or not data:
        return env, notes
    obj = data[0] if isinstance(data[0], dict) else {}
    for item in (obj.get("Config") or {}).get("Env") or []:
        if "=" in str(item):
            k, v = str(item).split("=", 1)
            env[k] = v
    image = str((obj.get("Config") or {}).get("Image") or obj.get("Image") or "")
    if image:
        notes.append("镜像 %s" % image)
    for m in obj.get("Mounts") or []:
        if not isinstance(m, dict):
            continue
        src = m.get("Source")
        dst = m.get("Destination")
        if src and dst:
            notes.append("挂载 %s → %s" % (src, dst))
    return env, notes


def docker(runner=None):
    """docker 容器里的推理服务。"""
    out = []
    r = _runner(runner)
    try:
        containers = _docker_ps(r)
    except Exception:
        return out
    for name, _obj in containers:
        try:
            procs = _docker_top(r, name)
        except Exception:
            procs = []
        cenv, cnotes = _docker_env(r, name)
        for pid, args in procs:
            argv = _split_cmdline(args)
            if not argv:
                continue
            inst = None
            try:
                inst = _launch.parse_launch(argv, cenv) if _launch is not None else None
            except Exception:
                inst = None
            if not inst:
                continue
            inst["where"]["container"] = name
            inst["where"]["pid"] = pid
            inst["where"]["host"] = _hostname()
            inst["where"]["source"] = "docker"
            for n in cnotes:
                if n not in inst["notes"]:
                    inst["notes"].append(n)
            out.append(inst)
    return out


def _split_cmdline(text):
    try:
        import shlex
        return shlex.split(text)
    except Exception:
        return [p for p in str(text or "").split(" ") if p]


# --------------------------------------------------------------------------- kubernetes
def kubernetes(runner=None):
    """kubectl get pods -A -o json 里的容器命令。"""
    out = []
    r = _runner(runner)
    rc, text, _err = _run(r, ["kubectl", "get", "pods", "-A", "-o", "json"], timeout=30)
    if rc != 0:
        return out
    try:
        data = json.loads(text)
    except Exception:
        return out
    for pod in (data or {}).get("items") or []:
        if not isinstance(pod, dict):
            continue
        meta = pod.get("metadata") or {}
        ns = str(meta.get("namespace") or "default")
        pname = str(meta.get("name") or "")
        spec = pod.get("spec") or {}
        for ctr in spec.get("containers") or []:
            if not isinstance(ctr, dict):
                continue
            argv = [str(a) for a in (ctr.get("command") or [])] + [str(a) for a in (ctr.get("args") or [])]
            if not argv:
                continue
            env = {}
            for e in ctr.get("env") or []:
                if not isinstance(e, dict):
                    continue
                name = e.get("name")
                if not name:
                    continue
                env[str(name)] = str(e.get("value") if e.get("value") is not None else "")
            inst = None
            try:
                inst = _launch.parse_launch(argv, env) if _launch is not None else None
            except Exception:
                inst = None
            if not inst:
                continue
            inst["where"]["container"] = "%s/%s/%s" % (ns, pname, ctr.get("name") or "container")
            inst["where"]["host"] = _hostname()
            inst["where"]["source"] = "kubernetes"
            out.append(inst)
    return out


# --------------------------------------------------------------------------- ssh
_REMOTE_SCAN = (
    "for p in /proc/[0-9]*/cmdline; do "
    "printf '%s\\0' \"$(tr '\\0' ' ' < $p 2>/dev/null)\" 2>/dev/null; printf '\\n'; done; "
    "command -v docker >/dev/null 2>&1 && docker ps -q | while read c; do "
    "docker top $c -eo pid,args 2>/dev/null | tail -n +2; done"
)


def _ssh_argv(host, remote):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", str(host), str(remote)]


def _json_tail(text):
    """JSON after other lines: with the hook installed every interpreter start prints [yee] lines first."""
    lines = (text or "").splitlines()
    for k, ln in enumerate(lines):
        if ln.lstrip().startswith(("{", "[")):
            try:
                return json.loads("\n".join(lines[k:]))
            except ValueError:
                continue
    return json.loads(text or "")


def ssh(host, runner=None):
    """远端机器上的推理服务：先问 yee doctor，没有就扫 /proc 与 docker。"""
    out = []
    r = _runner(runner)
    rc, text, _err = _run(r, _ssh_argv(host, "python3 -m yee doctor --json --local"), timeout=60)
    if rc in (0, 1, 3):  # doctor's exit is the plan's exit (0 / 1 / 3); the JSON is there in all three
        try:
            plan = _json_tail(text)
        except Exception:
            plan = None
        if isinstance(plan, dict):
            for row in plan.get("instances") or []:
                if not isinstance(row, dict):
                    continue
                inst = row.get("instance")
                if not isinstance(inst, dict):
                    continue
                inst.setdefault("where", {})
                inst["where"]["host"] = host
                inst["where"]["source"] = "ssh"
                out.append(inst)
            if out:
                return out
    rc2, text2, _err2 = _run(r, _ssh_argv(host, _REMOTE_SCAN), timeout=60)
    if rc2 != 0:
        return out
    for ln in text2.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        argv = _split_cmdline(ln)
        if not argv:
            continue
        inst = None
        try:
            inst = _launch.parse_launch(argv, None) if _launch is not None else None
        except Exception:
            inst = None
        if not inst:
            continue
        inst["where"]["host"] = host
        inst["where"]["source"] = "ssh"
        out.append(inst)
    return out


# --------------------------------------------------------------------------- HTTP 探测
def _http_get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def _http_json(url, timeout=5):
    raw = _http_get(url, timeout=timeout)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return None


def metrics_load(text):
    """Prometheus 文本 → {"running","waiting","accept_len","spec_draft","spec_accepted"}。"""
    out = {"running": None, "waiting": None, "accept_len": None, "spec_draft": None, "spec_accepted": None}
    counters = {}
    try:
        for ln in str(text or "").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.rsplit(None, 1)
            if len(parts) != 2:
                continue
            name, value = parts[0].strip(), parts[1].strip()
            try:
                num = float(value)
            except Exception:
                continue
            counters.setdefault(name, []).append(num)
    except Exception:
        return out
    for key, names in (("running", ("sglang:num_running_reqs", "vllm:num_requests_running")),
                       ("waiting", ("sglang:num_queue_reqs", "vllm:num_requests_waiting")),
                       ("accept_len", ("sglang:spec_accept_length",))):
        for n in names:
            if n in counters:
                out[key] = counters[n][-1]
                break
    for key, names in (("spec_draft", ("vllm:spec_decode_num_draft_tokens_total",)),
                       ("spec_accepted", ("vllm:spec_decode_num_accepted_tokens_total",))):
        for n in names:
            if n in counters:
                out[key] = sum(counters[n])
                break
    return out


def url(u, timeout=5):
    """探测一个 HTTP 服务 → 实例 dict（认不出引擎时 engine="unknown"）。"""
    base = str(u or "").rstrip("/")
    if not base:
        return None
    inst = None
    info = _http_json(base + "/get_server_info", timeout=timeout)
    if isinstance(info, dict):
        try:
            inst = _launch.from_server_info("sglang", info) if _launch is not None else None
        except Exception:
            inst = None
    if inst is None:
        props = _http_json(base + "/props", timeout=timeout)
        if isinstance(props, dict):
            inst = {"engine": "llama.cpp", "model": props.get("model_path") or props.get("default_generation_settings", {}).get("model")
                    if isinstance(props.get("default_generation_settings"), dict) else props.get("model_path")}
            inst["notes"] = ["来自 /props"]
    if inst is None:
        models = _http_json(base + "/v1/models", timeout=timeout)
        if isinstance(models, dict):
            data = models.get("data")
            model = None
            if isinstance(data, list) and data and isinstance(data[0], dict):
                model = data[0].get("id")
            inst = {"engine": "unknown", "model": model, "notes": ["来自 /v1/models"]}
    if inst is None:
        inst = {"engine": "unknown", "model": None, "notes": []}
    inst.setdefault("engine", "unknown")
    inst.setdefault("model", None)
    inst.setdefault("notes", [])
    inst.setdefault("where", {})
    inst["where"]["source"] = "url"
    inst["where"]["host"] = base
    metrics = _http_get(base + "/metrics", timeout=timeout)
    if metrics is not None:
        load = metrics_load(metrics.decode("utf-8", "replace"))
        inst["load"] = load
    return inst


# --------------------------------------------------------------------------- Maya 配置
def maya_configs(paths):
    """Maya JSON 配置 → 实例列表。"""
    out = []
    try:
        items = list(paths or [])
    except Exception:
        items = []
    for p in items:
        try:
            inst = _launch.from_maya_config(p) if _launch is not None else None
        except Exception:
            inst = None
        if not inst:
            continue
        inst.setdefault("where", {})
        inst["where"]["source"] = "config"
        out.append(inst)
    return out


# --------------------------------------------------------------------------- 汇总
def _key(inst):
    w = inst.get("where") or {}
    base = (str(w.get("host") or ""), str(w.get("container") or ""), str(w.get("pid") or ""))
    if w.get("pid"):
        return base
    # --cmd / --config / --url instances carry no pid: the launch command, port and model tell them apart
    return base + (str(w.get("source") or ""), " ".join(str(a) for a in inst.get("argv") or []),
                   str(inst.get("port") or ""), str(inst.get("model") or ""))


def collect(cmds=(), urls=(), hosts=(), configs=(), use_local=True, use_docker=True, use_k8s=False, runner=None):
    """把各处发现的实例收成一份列表（去重 + 补 model_info）。"""
    out = []
    r = _runner(runner)
    if use_local:
        try:
            out.extend(local())
        except Exception:
            pass
    if use_docker:
        try:
            out.extend(docker(runner))
        except Exception:
            pass
    if use_k8s:
        try:
            out.extend(kubernetes(runner))
        except Exception:
            pass
    for c in list(cmds or []):
        try:
            inst = _launch.parse_launch(c, os.environ) if _launch is not None else None
        except Exception:
            inst = None
        if not inst:
            continue
        inst.setdefault("where", {})
        inst["where"]["source"] = "cmd"
        out.append(inst)
    for u in list(urls or []):
        try:
            inst = url(u)
        except Exception:
            inst = None
        if inst:
            out.append(inst)
    for h in list(hosts or []):
        try:
            out.extend(ssh(h, runner))
        except Exception:
            pass
    for c in list(configs or []):
        out.extend(maya_configs([c]))
    seen = set()
    final = []
    for inst in out:
        try:
            k = _key(inst)
        except Exception:
            k = None
        if k in seen:
            continue
        seen.add(k)
        try:
            model = inst.get("model")
            if model and not inst.get("model_info"):
                info = _launch.model_info(model) if _launch is not None else {}
                if info:
                    inst["model_info"] = info
        except Exception:
            pass
        final.append(inst)
    return final


__all__ = ["local", "docker", "kubernetes", "ssh", "url", "maya_configs", "metrics_load", "collect",
           "SubprocessRunner"]
