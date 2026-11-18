"""Command line of Yee: ``yee <subcommand>`` (``recall-accel`` is the same program; or ``python -m yee``).

  auto [--cmdline CMD] [--apply]   detect engine + speculative head, check the symbols, print what to set;
                                   --apply also writes the .pth hook into this interpreter
  detect [--cmdline CMD]           the detection part of auto only (JSON)
  doctor [--cmd CMD]... [--url URL]... [--hosts H1,H2] [--config FILE]...
          [--no-local] [--no-docker] [--k8s] [--json] [--local] [--symbols] [--adapter NAME]
                                   4.3: find the running inference services (local / docker / k8s / ssh / URL /
                                   launch command), say whether Yee accelerates them and what to change; the exit
                                   code is the plan's (0 ok, 1 changes needed, 3 nothing found / unsupported).
                                   --adapter keeps the 4.2 symbol check only.
  install-hook | uninstall-hook    write / remove yee.pth in this interpreter's site-packages
                                   (uninstall also removes the 2.1 recall_accel.pth)
  status | pause on|off | clear    the record directory's stats.json, the "paused" / "clear.request" files
  bench --url URL --model M        same greedy request twice: tok/s cold vs warm (did copies land?)
  selftest                         run the bundled tests (no pytest needed)
  tensorfold ...                   TensorFold: detect / args / status / clear / export / import
  patch list|show|path|apply       source patches for TensorFold / cuteafd / mlx-serve
  doc [topic]                      the built-in manual (Chinese); no topic lists the topics
  bases [--json]                   base x speculative head support table
  config                           every switch's effective value and its source (env / default / off)
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import site
import sys
import sysconfig
import time
import urllib.request

from . import __version__
from .core import ENV_NEW, ENV_OLD, _env

PTH_NAME = "yee.pth"
PTH_LINE = "import yee.autoload"
OLD_PTH_NAMES = ("recall_accel.pth",)  # written by recall-accel 2.x
EXIT_UNSUPPORTED = 3


def _site_packages() -> str:
    cands = []
    pl = sysconfig.get_paths().get("purelib")
    if pl:
        cands.append(pl)
    try:
        cands.extend(site.getsitepackages())
    except Exception:  # noqa: BLE001
        pass
    try:
        cands.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001
        pass
    uniq = list(dict.fromkeys(p for p in cands if p))
    for p in uniq:
        if os.path.isdir(os.path.join(p, "yee")):
            return p
    for p in uniq:
        if os.path.isdir(p):
            return p
    return pl or os.path.join(sys.prefix, "Lib", "site-packages")


def _pth_path() -> str:
    return os.path.join(_site_packages(), PTH_NAME)


def _record_dir(override: str = "") -> str:
    if override:
        return os.path.abspath(os.path.expanduser(override))
    d = _env(ENV_NEW["dir"], ENV_OLD["dir"], "")
    if d:
        return os.path.abspath(os.path.expanduser(d))
    tok = _env(ENV_NEW["tokenizer_dir"], ENV_OLD["tokenizer_dir"], "/models/DeepSeek-V4.1-Flash")
    digest = "unknown"
    try:
        from .integrations.sglang_dspark import _digest_from_dir

        dg, _, found = _digest_from_dir(tok)
        if found:
            digest = dg
    except Exception:  # noqa: BLE001
        pass
    return os.path.join(os.path.expanduser("~/.cache/recall_accel"), digest[:16])


def _versions() -> dict:
    out = {"yee": __version__, "python": sys.version.split()[0]}
    for name in ("numpy", "torch", "sglang", "vllm"):
        try:
            m = importlib.import_module(name)
            out[name] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


# ----------------------------------------------------------------------------------------------- detect / auto
def _detect(cmdline: str = "") -> dict:
    from .integrations import registry, tensorfold

    info = {"versions": None, "source": None, "servers": [], "choices": []}
    if cmdline:
        servers = [registry.parse_cmdline(cmdline)]
        info["source"] = "--cmdline"
    else:
        servers = registry.detect_running()
        info["source"] = "process table" if servers else None
    imp = registry.engines_importable()
    if not servers:
        for eng in ("sglang", "vllm"):
            if imp.get(eng):
                servers.append({"engine": eng, "algorithm": "NONE", "topk": None,
                                "note": "engine importable, no running server / --cmdline: head unknown"})
        tf = tensorfold.find_binary()
        if tf:
            servers.append({"engine": "tensorfold", "algorithm": "NONE", "topk": None, "path": tf})
        info["source"] = info["source"] or "importable engines"
    for s in servers:
        s.pop("argv", None)
        c = registry.choose(s.get("engine"), s.get("algorithm"), s.get("topk"))
        c["importable_here"] = bool(imp.get(s.get("engine") or "", False))
        info["choices"].append(c)
    info["servers"] = servers
    return info


def _env_lines(engine: str) -> list:
    lines = ["RECALL_ACCEL=1", "RECALL_ACCEL_BACKEND=memory   # disk to keep the record across restarts",
             "RECALL_ACCEL_GIB=0.25", "RECALL_ACCEL_TOKENIZER_DIR=<the model directory>"]
    if engine == "vllm":
        lines.append("VLLM_USE_V2_MODEL_RUNNER=1   # the hook lives in vllm/v1/worker/gpu")
    return lines


def cmd_detect(args) -> int:
    print(json.dumps(_detect(args.cmdline), indent=1, ensure_ascii=False, default=str))
    return 0


def cmd_auto(args) -> int:
    from .integrations import registry, tensorfold

    info = _detect(args.cmdline)
    print("recall-accel %s auto (detected from %s)" % (__version__, info["source"] or "nothing"), flush=True)
    if not info["choices"]:
        print("no SGLang / vLLM / TensorFold found: nothing to do", flush=True)
        return EXIT_UNSUPPORTED
    rc = EXIT_UNSUPPORTED
    for c in info["choices"]:
        eng, alg, ad = c["engine"], c["algorithm"], c["adapter"]
        print("- engine=%s head=%s -> adapter=%s status=%s" % (eng, alg, ad, c["status"]), flush=True)
        print("  %s" % (c["reason"],), flush=True)
        if eng == "tensorfold":
            det = tensorfold.detect()
            print("  tensorfold: %s" % (det["reason"],), flush=True)
            if det["recall_supported"]:
                for k, v in sorted(tensorfold.launch_env().items()):
                    print("  export %s=%s   # Yee patch 0001 (yee patch apply tensorfold): priced proposers "
                          "fall back to Recall" % (k, v), flush=True)
                print("  start with: tensorfold serve ... %s" % " ".join(tensorfold.launch_args(detected=det)),
                      flush=True)
                rc = 0
            continue
        if ad is None or c["status"] in ("unsupported", "no-op"):
            print("  -> not accelerated; the server runs unchanged", flush=True)
            continue
        ok = True
        if c["importable_here"]:
            rows = registry.check_symbols(ad)
            for good, text in rows:
                tag = "OK  " if good else ("info" if good is None else "FAIL")
                print("    %s %s" % (tag, text), flush=True)
            ok = all(g is not False for g, _ in rows)
            print("  symbols: %s" % ("PASS" if ok else "FAIL (hooks would stay inert: not installing)"), flush=True)
        else:
            print("  symbols: not checked (%s is not importable in this interpreter; run auto inside the "
                  "server's image)" % eng, flush=True)
        if not ok:
            continue
        print("  set for the server process:", flush=True)
        for ln in _env_lines(eng):
            print("    export %s" % ln, flush=True)
        if args.apply:
            if c["importable_here"]:
                cmd_install_hook(args)
            else:
                print("  --apply skipped: install the wheel into the server's interpreter and run auto there",
                      flush=True)
        else:
            print("  then: yee install-hook (or rerun with --apply), restart the server", flush=True)
        rc = 0
    return rc


def _doctor_symbols(args) -> int:
    """The 4.2 symbol check: every symbol / signature / source marker the adapters need."""
    from .integrations import registry

    v = _versions()
    for k, val in v.items():
        print("%-13s %s" % (k, val if val else "not importable"), flush=True)
    names = [args.adapter] if args.adapter else []
    if not names:
        if v.get("sglang"):
            names.extend(registry.SGLANG_ALL)
        if v.get("vllm"):
            names.append("vllm_spec")
    if not names or (args.adapter and not (v.get("sglang") or v.get("vllm"))):
        print("doctor: neither sglang nor vllm importable here (TensorFold: yee tensorfold detect)")
        return EXIT_UNSUPPORTED
    bad = 0
    for n in names:
        rows = registry.check_symbols(n)
        good = all(g is not False for g, _ in rows)
        bad += 0 if good else 1
        print("[%s] %s" % (n, "PASS" if good else "FAIL"), flush=True)
        for g, text in rows:
            print("    %s %s" % ("OK  " if g else ("info" if g is None else "FAIL"), text), flush=True)
    hook = os.path.exists(_pth_path())
    print("hook: %s (%s)" % ("installed" if hook else "not installed", _pth_path()), flush=True)
    print("RECALL_ACCEL on: %s" % _env(ENV_NEW["on"], ENV_OLD["on"], "0"), flush=True)
    print("doctor %s" % ("PASS" if not bad else "FAIL (%d adapter(s))" % bad), flush=True)
    return 0 if not bad else 1


def _doctor_engines_here() -> bool:
    """True when sglang or vllm can be imported here (then the symbol check is worth appending)."""
    try:
        import importlib.util

        for name in ("sglang", "vllm"):
            try:
                if importlib.util.find_spec(name) is not None:
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False
    except Exception:  # noqa: BLE001
        return False


def _doctor_hosts(value) -> list:
    out = []
    for one in str(value or "").replace(";", ",").split(","):
        one = one.strip()
        if one:
            out.append(one)
    return out


def cmd_doctor(args) -> int:
    """4.3: find the deployment, say what to change; exit code = plan["exit"]."""
    if getattr(args, "adapter", ""):
        return _doctor_symbols(args)
    try:
        from .integrations import discover, plan
    except Exception as e:  # noqa: BLE001
        print("doctor: 发现模块不可用：%r" % (e,), flush=True)
        return EXIT_UNSUPPORTED
    local_only = bool(getattr(args, "local", False))
    hosts = [] if local_only else _doctor_hosts(getattr(args, "hosts", ""))
    urls = [] if local_only else list(getattr(args, "url", None) or [])
    use_local = True if local_only else not getattr(args, "no_local", False)
    use_docker = not getattr(args, "no_docker", False)
    use_k8s = bool(getattr(args, "k8s", False)) and not local_only
    try:
        instances = discover.collect(cmds=list(getattr(args, "cmd", None) or []), urls=urls, hosts=hosts,
                                     configs=list(getattr(args, "config", None) or []), use_local=use_local,
                                     use_docker=use_docker, use_k8s=use_k8s)
    except Exception as e:  # noqa: BLE001
        print("doctor: 发现部署目标失败：%r" % (e,), flush=True)
        return EXIT_UNSUPPORTED
    try:
        p = plan.analyze(list(instances or []))
    except Exception as e:  # noqa: BLE001
        print("doctor: 生成部署诊断失败：%r" % (e,), flush=True)
        return EXIT_UNSUPPORTED
    if getattr(args, "json", False):
        print(json.dumps(p, ensure_ascii=False, indent=1, default=str), flush=True)
    else:
        print(plan.render(p), flush=True)
        if getattr(args, "symbols", False) or _doctor_engines_here():
            print("符号检查", flush=True)
            rc = _doctor_symbols(args)
            if rc and not p.get("exit"):
                return rc
    try:
        return int(p.get("exit") or 0)
    except Exception:  # noqa: BLE001
        return EXIT_UNSUPPORTED


# ----------------------------------------------------------------------------------------------- hook / record
def cmd_install_hook(args) -> int:
    p = _pth_path()
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(PTH_LINE + "\n")
    except OSError as e:
        print("install-hook FAILED: %s: %s" % (p, e), flush=True)
        return 1
    print("wrote %s (%s)" % (p, PTH_LINE), flush=True)
    return 0


def cmd_uninstall_hook(args) -> int:
    p = _pth_path()
    if os.path.exists(p):
        os.remove(p)
        print("removed %s" % p, flush=True)
    else:
        print("no hook at %s" % p, flush=True)
    for name in OLD_PTH_NAMES:
        old = os.path.join(os.path.dirname(p), name)
        if os.path.exists(old):
            os.remove(old)
            print("removed %s (2.x hook)" % old, flush=True)
    return 0


def cmd_status(args) -> int:
    d = _record_dir(args.dir)
    p = os.path.join(d, "stats.json")
    print("record dir: %s" % d, flush=True)
    if not os.path.exists(p):
        print("no stats.json yet", flush=True)
        return 0
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    for k in sorted(data):
        print("  %-18s %s" % (k, data[k]), flush=True)
    return 0


def cmd_pause(args) -> int:
    d = _record_dir(args.dir)
    os.makedirs(d, mode=0o700, exist_ok=True)
    p = os.path.join(d, "paused")
    if args.state == "on":
        open(p, "w").close()
        print("paused: wrote %s" % p, flush=True)
    elif os.path.exists(p):
        os.remove(p)
        print("paused: removed %s" % p, flush=True)
    return 0


def cmd_clear(args) -> int:
    d = _record_dir(args.dir)
    os.makedirs(d, mode=0o700, exist_ok=True)
    p = os.path.join(d, "clear.request")
    open(p, "w").close()
    print("clear requested: wrote %s" % p, flush=True)
    return 0


# ----------------------------------------------------------------------------------------------- bench / selftest
def _complete(url, model, prompt, max_tokens, timeout):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        js = json.loads(r.read())
    dt = time.perf_counter() - t
    n = int((js.get("usage") or {}).get("completion_tokens") or 0)
    text = ((js.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return n, dt, text


def cmd_bench(args) -> int:
    prompt = args.prompt or ("Write a detailed, numbered explanation of how a hash map handles collisions, "
                             "with a short Python example for each strategy.")
    try:
        rows = [_complete(args.url, args.model, prompt, args.max_tokens, args.timeout) for _ in range(args.rounds)]
    except Exception as e:  # noqa: BLE001
        print("bench: request failed: %r" % (e,), flush=True)
        return 1
    for i, (n, dt, _) in enumerate(rows):
        print("round %d: %d tokens in %.2fs = %.1f tok/s" % (i + 1, n, dt, n / dt if dt else 0.0), flush=True)
    same = len({r[2] for r in rows}) == 1
    print("replies identical: %s" % same, flush=True)
    if len(rows) >= 2 and rows[0][1] > 0 and rows[-1][1] > 0:
        a = rows[0][0] / rows[0][1]
        b = rows[-1][0] / rows[-1][1]
        print("last vs first: x%.2f (copies land on repeated content; >1.1 means Recall is working)"
              % (b / a if a else 0.0), flush=True)
    return 0 if same else 1


def cmd_selftest(args) -> int:
    from . import selftests

    return selftests.run_all(verbose=args.verbose)


# ----------------------------------------------------------------------------------------------- doc / bases
def _docs_root():
    """yee/docs as an importlib.resources Traversable (Python >= 3.9), else as a plain directory path."""
    try:
        from importlib import resources

        return resources.files(__package__).joinpath("docs")
    except (ImportError, AttributeError):
        import pathlib

        return pathlib.Path(os.path.dirname(os.path.abspath(__file__)), "docs")


def doc_topics() -> dict:
    """topic -> file name of every yee/docs/*.md (read through importlib.resources: works from a wheel)."""
    out = {}
    for item in _docs_root().iterdir():
        if item.name.endswith(".md"):
            out[item.name[:-3].lower()] = item.name
    return dict(sorted(out.items()))


def doc_text(topic: str) -> str:
    t = topic.strip().lower()
    if t.endswith(".md"):
        t = t[:-3]
    name = doc_topics().get(t)
    if not name:
        raise KeyError(topic)
    return _docs_root().joinpath(name).read_text(encoding="utf-8")


def cmd_doc(args) -> int:
    topics = doc_topics()
    if not args.topic:
        print("Yee %s built-in docs: yee doc <topic>" % __version__, flush=True)
        for t, name in topics.items():
            first = ""
            try:
                first = next((ln.lstrip("# ").strip() for ln in doc_text(t).splitlines() if ln.strip()), "")
            except Exception:  # noqa: BLE001
                pass
            print("  %-16s %s" % (t, first), flush=True)
        return 0
    try:
        sys.stdout.write(doc_text(args.topic))
        sys.stdout.flush()
    except KeyError:
        print("no such topic %r; topics: %s" % (args.topic, ", ".join(topics)), flush=True)
        return 2
    return 0


STATIC_BASES = (
    {"base": "tensorfold", "head": "RECALL (native)", "adapter": "tensorfold", "status": "native",
     "nongreedy": True, "shared": True, "tree": "none", "tree_note": "TensorFold's own Recall path; Yee 2.3.0 does not change it",
     "note": "TensorFold >= 1.0.12 has Recall built in: tensorfold serve MODEL --recall (yee tensorfold args); "
             "sampled requests use the same verify path; the record is shared by every user of the machine"},
    {"base": "tensorfold", "head": "priced proposers (DSv4.1 lookup plan, Llama drafter, Qwen4 lanes)",
     "adapter": "patch 0001", "status": "patch", "nongreedy": True, "shared": True,
     "tree": "none", "tree_note": "patch 0001 asks Recall after an empty priced proposer: one chain, unchanged",
     "note": "upstream never asks Recall when a priced proposer is empty: yee patch apply tensorfold --src DIR, "
             "rebuild, then TF_RECALL_AFTER_PRICED=1"},
    {"base": "tensorfold", "head": "Qwen4 / Kimi Metal / Drama / Flash Next on Mac", "adapter": None,
     "status": "unsupported", "nongreedy": None, "shared": None, "tree": "none", "tree_note": "no Recall",
     "note": "these engines reject --recall (native_host / qwen4_exp refuse it before start)"},
    {"base": "cuteafd", "head": "DeepSeek V4.1 native serve", "adapter": "patch v2", "status": "patch",
     "nongreedy": False, "shared": False, "tree": "none", "tree_note": "greedy chain patch, unchanged",
     "note": "yee patch apply cuteafd --src DIR --variant v2 (v1 kept); greedy only; record in process memory"},
    {"base": "mlx-serve", "head": "DSpark (M5 xypd)", "adapter": "source-drop", "status": "patch",
     "nongreedy": True, "shared": True, "tree": "native",
     "tree_note": "r598: recalled copies and the stage draft verified as branch states of one batched verify "
                  "(dspark.zig dsparkTreeStep); not in the source drop yet (r597 files), the build lives on M5",
     "note": "yee patch apply mlx-serve --src DIR replaces 4 files in src/engines/mlx/spec, rebuild; "
             "greedy and sampled requests; penalty requests do not speculate"},
    {"base": "strata", "head": "MTP + prompt lookup (strata 0.1.59)", "adapter": "patch strata-recall",
     "status": "patch", "nongreedy": True, "shared": True, "tree": "none",
     "tree_note": "a recall copy replaces one window's chain drafts; no branches",
     "note": "yee patch apply strata-recall --src DIR, rebuild the CUDA strata target, then RECALL_ACCEL=1 (or "
             "--recall); solo windows and the serial --batch-mtp slots; not STRATA_BATCH_DF2 / STRATA_PIPED_MTP"},
)


def bases_rows() -> list:
    from .integrations import registry

    rows = []
    for engine in ("sglang", "vllm"):
        for alg, (adapter, status, note) in registry.ADAPTERS[engine].items():
            mode, why = registry.tree_mode(engine, alg)
            rows.append({"base": engine, "head": alg, "adapter": adapter, "status": status,
                         "nongreedy": adapter is not None, "shared": adapter is not None, "note": note,
                         "tree": mode if adapter is not None else "none", "tree_note": why})
    rows.extend(dict(r) for r in STATIC_BASES)
    return rows


def cmd_bases(args) -> int:
    rows = bases_rows()
    if args.json:
        print(json.dumps(rows, indent=1, ensure_ascii=False), flush=True)
        return 0
    print("%-10s %-22s %-14s %-11s %-7s %s" % ("base", "head", "adapter", "status", "tree", "note"), flush=True)
    for r in rows:
        print("%-10s %-22s %-14s %-11s %-7s %s" % (r["base"], r["head"], r["adapter"] or "-", r["status"],
                                                   r.get("tree", "none"), r["note"]), flush=True)
    return 0


def cmd_config(args) -> int:
    """4.2: what each switch resolves to in this environment, after the startup defaults."""
    from .integrations import defaults

    rows = defaults.effective()
    if args.json:
        print(json.dumps([{"name": n, "value": v, "source": s} for n, v, s in rows], indent=1), flush=True)
        return 0
    w = max(len(n) for n, _, _ in rows)
    print("yee %s  (source: env = set in the environment, default = 4.2 default, off = YEE=0)" % __version__,
          flush=True)
    for n, v, s in rows:
        print("%-*s  %-10s %s" % (w, n, v, s), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import patches

    prog = os.path.basename(sys.argv[0] or "") if sys.argv else ""
    prog = "recall-accel" if prog.lower().startswith("recall-accel") else "yee"
    ap = argparse.ArgumentParser(prog=prog, description="Yee: Recall copy drafts for SGLang / vLLM / TensorFold / "
                                                        "cuteafd / mlx-serve (docs: yee doc)")
    ap.add_argument("--version", action="version", version="yee %s" % __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("auto", cmd_auto), ("detect", cmd_detect)):
        sp = sub.add_parser(name)
        sp.add_argument("--cmdline", default="", help="the server's launch command (default: running processes)")
        sp.add_argument("--apply", action="store_true", help="auto: also write the .pth hook")
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("doctor", help="4.3: find the deployment, say what to change (exit code = the plan's)")
    sp.add_argument("--cmd", action="append", default=[], metavar="CMD",
                    help="the server's launch command (repeatable)")
    sp.add_argument("--url", action="append", default=[], metavar="URL",
                    help="probe a running server over HTTP (repeatable)")
    sp.add_argument("--hosts", default="", metavar="H1,H2",
                    help="comma separated remote hosts to look at over ssh")
    sp.add_argument("--config", action="append", default=[], metavar="FILE",
                    help="a Maya launch config (json), repeatable")
    sp.add_argument("--no-local", action="store_true", help="do not look at the local processes")
    sp.add_argument("--no-docker", action="store_true", help="do not look at the docker containers")
    sp.add_argument("--k8s", action="store_true", help="also look at the Kubernetes pods")
    sp.add_argument("--json", action="store_true", help="print the plan as JSON (kit and ssh parse it)")
    sp.add_argument("--local", action="store_true",
                    help="only the local processes and the local docker (ignore --hosts / --url / --k8s)")
    sp.add_argument("--symbols", action="store_true", help="also run the 4.2 symbol check")
    sp.add_argument("--adapter", default="", help="only the 4.2 symbol check of this adapter")
    sp.set_defaults(fn=cmd_doctor)
    sub.add_parser("install-hook").set_defaults(fn=cmd_install_hook)
    sub.add_parser("uninstall-hook").set_defaults(fn=cmd_uninstall_hook)
    sp = sub.add_parser("status")
    sp.add_argument("--dir", default="")
    sp.set_defaults(fn=cmd_status)
    sp = sub.add_parser("pause")
    sp.add_argument("state", choices=["on", "off"])
    sp.add_argument("--dir", default="")
    sp.set_defaults(fn=cmd_pause)
    sp = sub.add_parser("clear")
    sp.add_argument("--dir", default="")
    sp.set_defaults(fn=cmd_clear)
    sp = sub.add_parser("bench")
    sp.add_argument("--url", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--prompt", default="")
    sp.add_argument("--max-tokens", type=int, default=512)
    sp.add_argument("--rounds", type=int, default=2)
    sp.add_argument("--timeout", type=float, default=600.0)
    sp.set_defaults(fn=cmd_bench)
    sp = sub.add_parser("selftest")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(fn=cmd_selftest)
    sub.add_parser("tensorfold", add_help=False)
    patches.add_parser(sub)
    sp = sub.add_parser("doc", help="built-in manual")
    sp.add_argument("topic", nargs="?", default="")
    sp.set_defaults(fn=cmd_doc)
    sp = sub.add_parser("bases", help="base x speculative head support table")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_bases)
    sp = sub.add_parser("config", help="effective value and source of every switch (4.2 defaults)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_config)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "tensorfold":
        from .integrations import tensorfold

        return tensorfold.main(argv[1:])
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
