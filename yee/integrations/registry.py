"""Which adapter serves which engine + speculative head, how to detect it, and how to install it.

Matrix (``ADAPTERS``): engine, algorithm -> adapter module, status, note.
  status "verified"      ran on real hardware with output checks (SGLang DSPARK on DGX Spark, Mia image)
  status "container"     symbols + signatures checked in the real image, CPU simulation passes, no GPU serving run
  status "unsupported"   detected but left alone (reason given); the server runs exactly as without the package

Install (``install_auto``): every adapter only patches the modules of its own head, and its hook only runs when
that head's code path runs, so in "auto" all adapters of the importable engines are installed at once and the
one the server actually builds takes part. ``RECALL_ACCEL_ADAPTER`` names one adapter instead (or ``off``).
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import shlex
import sys

# engine -> {ALGORITHM: (adapter, status, note)}
ADAPTERS = {
    "sglang": {
        "DSPARK": ("sglang_dspark", "verified", "DSparkWorkerV2 proposer; verified on ysy37-40 (Mia image)"),
        "DFLASH": ("sglang_dflash", "container", "DFlashWorkerV2 block drafts"),
        "EAGLE": ("sglang_eagle", "container", "build_eagle_verify_input, topk=1 chains"),
        "EAGLE3": ("sglang_eagle", "container", "same path as EAGLE"),
        "NEXTN": ("sglang_eagle", "container", "NEXTN is an EAGLE alias (spec_registry.py:194)"),
        "MTP": ("sglang_eagle", "container", "MTP heads run the EAGLE workers"),
        "STANDALONE": ("sglang_eagle", "container", "StandaloneWorkerV2(EAGLEWorkerV2)"),
        "UNO": ("sglang_eagle", "container", "uno_worker_v2.py:550 builds through build_eagle_verify_input"),
        "NGRAM": ("sglang_ngram", "container", "NGRAMWorker._prepare_draft_tokens, chain per request"),
        "FROZEN_KV_MTP": (None, "unsupported", "FrozenKVMTP draft() builds its own verify input (not hooked)"),
    },
    "vllm": {
        "DSPARK": ("vllm_spec", "container", "DSparkSpeculator (inherits DFlashSpeculator.propose)"),
        "DFLASH": ("vllm_spec", "container", "DFlashSpeculator / DFlash2Speculator propose"),
        "EAGLE": ("vllm_spec", "container", "EagleSpeculator propose (V2 runner)"),
        "EAGLE3": ("vllm_spec", "container", "EagleSpeculator propose (V2 runner)"),
        "MTP": ("vllm_spec", "container", "MTPSpeculator / MultiModuleMTPSpeculator / Gemma4Speculator"),
        "NGRAM": (None, "unsupported", "V2 init_speculator has no ngram speculator"),
        "SUFFIX": (None, "unsupported", "no V2 speculator"),
    },
}
ALIASES = {"NEXTN": "NEXTN", "EAGLE-3": "EAGLE3", "DEEPSEEK_MTP": "MTP", "QWEN3_NEXT_MTP": "MTP",
           "MIMO_MTP": "MTP", "ERNIE_MTP": "MTP", "GLM4_MOE_MTP": "MTP", "LONGCAT_FLASH_MTP": "MTP",
           "DFLASH2": "DFLASH", "NGRAM_GPU": "NGRAM"}
SGLANG_ALL = ("sglang_dspark", "sglang_dflash", "sglang_eagle", "sglang_ngram")


def norm_algorithm(name) -> str:
    if not name:
        return "NONE"
    up = str(name).strip().upper().replace("-", "_")
    if up.endswith("_MTP") and up not in ADAPTERS["sglang"]:
        up = "MTP"
    return ALIASES.get(up, up)


def _flag(argv, *names):
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def parse_cmdline(cmd) -> dict:
    """Engine + head from a launch command line (string or argv list)."""
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd or [])
    joined = " ".join(argv)
    out = {"engine": None, "algorithm": "NONE", "topk": None, "argv": argv}
    if "sglang" in joined and ("launch_server" in joined or "sglang serve" in joined or " serve " in joined
                               or "sglang.launch" in joined):
        out["engine"] = "sglang"
        out["algorithm"] = norm_algorithm(_flag(argv, "--speculative-algorithm", "--speculative-algo"))
        tk = _flag(argv, "--speculative-eagle-topk")
        out["topk"] = int(tk) if tk and tk.lstrip("-").isdigit() else None
    elif "vllm" in joined:
        out["engine"] = "vllm"
        sc = _flag(argv, "--speculative-config", "--speculative_config")
        method = None
        if sc:
            try:
                method = (json.loads(sc) or {}).get("method")
            except ValueError:
                method = None
        out["algorithm"] = norm_algorithm(method)
    elif "tensorfold" in joined:
        out["engine"] = "tensorfold"
        out["algorithm"] = "RECALL" if "--recall" in argv else "NONE"
    return out


def choose(engine, algorithm, topk=None) -> dict:
    """The adapter for (engine, algorithm) with its status and reason."""
    alg = norm_algorithm(algorithm)
    row = {"engine": engine, "algorithm": alg, "adapter": None, "status": "unsupported", "reason": ""}
    if engine == "tensorfold":
        row.update(adapter="tensorfold", status="native",
                   reason="TensorFold has Recall built in (>= 1.0.12): use `yee tensorfold args`")
        return row
    table = ADAPTERS.get(engine or "")
    if table is None:
        row["reason"] = "unknown engine %r: nothing is patched" % (engine,)
        return row
    if alg == "NONE":
        if engine == "sglang":
            row["reason"] = ("no speculative head: start with --speculative-algorithm NGRAM to get copies "
                             "without a draft model, or add a draft head")
        else:
            row["reason"] = "no speculative head (--speculative-config): nothing to accelerate"
        return row
    hit = table.get(alg)
    if hit is None:
        row["reason"] = "head %s is not in the %s matrix: nothing is patched" % (alg, engine)
        return row
    adapter, status, note = hit
    row.update(adapter=adapter, status=status, reason=note)
    if adapter == "sglang_eagle" and topk not in (None, 1):
        row.update(status="no-op", reason="EAGLE topk=%s is a tree; copies run on topk=1 chains only" % topk)
    return row


def detect_running() -> list:
    """Server command lines found in the process table (Linux /proc)."""
    found = []
    if not os.path.isdir("/proc"):
        return found
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
        except OSError:
            continue
        info = parse_cmdline(argv)
        if info["engine"] and any(k in " ".join(argv) for k in ("launch_server", "serve", "api_server")):
            info["pid"] = int(pid)
            found.append(info)
    return found


def engines_importable() -> dict:
    out = {}
    for name in ("sglang", "vllm"):
        try:
            out[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            out[name] = False
    return out


def adapter_module(name):
    return importlib.import_module("%s.%s" % (__package__, name))


def _static(mpath):
    """(ast tree, source text) of a module without importing it (its parent packages are imported)."""
    import ast

    spec = importlib.util.find_spec(mpath)
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        raise ImportError("no source for %s" % mpath)
    with open(spec.origin, "r", encoding="utf-8") as f:
        src = f.read()
    return ast.parse(src), src


def _static_lookup(tree, dotted):
    import ast

    node = tree
    for part in dotted.split("."):
        found = None
        for ch in getattr(node, "body", []):
            if isinstance(ch, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ch.name == part:
                found = ch
            elif isinstance(ch, (ast.Assign, ast.AnnAssign)) and node is tree:
                tg = ch.targets if isinstance(ch, ast.Assign) else [ch.target]
                if any(isinstance(t, ast.Name) and t.id == part for t in tg):
                    found = ch
            elif isinstance(ch, (ast.ImportFrom, ast.Import)) and node is tree:
                if any((al.asname or al.name).split(".")[-1] == part for al in ch.names):
                    found = ch
        if found is None:
            return None
        node = found
    return node


def _static_params(fn):
    a = fn.args
    return tuple(x.arg for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs))


def _static_rows(mod, mpath, err):
    """REQUIRED_SYMBOLS / SIGNATURES / SOURCE_MARKERS of one module checked from its source (import failed)."""
    import ast

    rows = []
    try:
        tree, src = _static(mpath)
    except Exception as e:  # noqa: BLE001
        return [(False, "module %s (%r; no source either: %r)" % (mpath, err, e))]
    why = "static, import needs a GPU here: %s" % (str(err).strip().splitlines() or ["?"])[0][:80]
    for mp, cname, attrs in getattr(mod, "REQUIRED_SYMBOLS", ()):
        if mp != mpath:
            continue
        holder = _static_lookup(tree, cname) if cname else tree
        if holder is None:
            rows.append((False, "%s.%s (%s)" % (mpath, cname, why)))
            continue
        bad = [x for x in attrs if _static_lookup(holder if cname else tree, x) is None
               and not (cname and isinstance(holder, ast.ClassDef) and holder.bases)]
        rows.append((not bad, "%s%s: %s (%s)" % (mpath, "." + cname if cname else "",
                                                ", ".join(attrs) if not bad else "missing " + ", ".join(bad), why)))
    for mp, dotted, params in getattr(mod, "SIGNATURES", ()):
        if mp != mpath:
            continue
        fn = _static_lookup(tree, dotted)
        names = _static_params(fn) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) else ()
        miss = [p for p in params if p not in names]
        rows.append((bool(names) and not miss, "signature %s.%s%s (%s)" % (mpath, dotted, "" if not miss else
                                                                         " missing " + ", ".join(miss), why)))
    for mp, cname, markers in getattr(mod, "SOURCE_MARKERS", ()):
        if mp != mpath:
            continue
        node = _static_lookup(tree, cname)
        seg = ast.get_source_segment(src, node) if node is not None else ""
        miss = [k for k in markers if k not in (seg or "")]
        rows.append((bool(seg) and not miss, "source %s.%s%s (%s)" % (mpath, cname, "" if not miss else
                                                                     " missing %r" % miss, why)))
    return rows


def check_symbols(adapter: str) -> list:
    """[(ok, text)] for the adapter's REQUIRED_SYMBOLS, SIGNATURES and SOURCE_MARKERS in this interpreter.

    A module whose import fails because this interpreter has no GPU (sgl_kernel / libcuda) is checked from its
    source instead; ok=None rows are information only."""
    rows = []
    mod = adapter_module(adapter)
    static_done = set()

    def gpu_import_error(e):
        t = repr(e)
        return isinstance(e, (ImportError, OSError)) and any(k in t for k in ("libcuda", "sgl_kernel", "common_ops",
                                                                            "CUDA"))

    for mpath, cname, attrs in getattr(mod, "REQUIRED_SYMBOLS", ()):
        try:
            m = importlib.import_module(mpath)
        except Exception as e:  # noqa: BLE001
            if gpu_import_error(e):
                if mpath not in static_done:
                    static_done.add(mpath)
                    rows.extend(_static_rows(mod, mpath, e))
            else:
                rows.append((False, "module %s (%r)" % (mpath, e)))
            continue
        obj = getattr(m, cname, None) if cname else m
        if obj is None:
            rows.append((False, "%s.%s" % (mpath, cname)))
            continue
        bad = [a for a in attrs if not hasattr(obj, a)]
        label = "%s%s" % (mpath, "." + cname if cname else "")
        rows.append((not bad, "%s: %s" % (label, ", ".join(attrs) if not bad else "missing " + ", ".join(bad))))
    for mpath, cname, attrs in getattr(mod, "OPTIONAL_SYMBOLS", ()):
        try:
            m = importlib.import_module(mpath)
            obj = getattr(m, cname, None) if cname else m
            ok = obj is not None and all(hasattr(obj, a) for a in attrs)
        except Exception:  # noqa: BLE001
            ok = False
        rows.append((None, "%s%s: %s (only needed for that head)" % (mpath, "." + cname if cname else "",
                                                                     "present" if ok else "absent")))
    for mpath, dotted, params in getattr(mod, "SIGNATURES", ()):
        if mpath in static_done:
            continue
        try:
            obj = importlib.import_module(mpath)
            for part in dotted.split("."):
                obj = getattr(obj, part)
            obj = getattr(obj, "__wrapped__", obj) if getattr(obj, "__recall_wrapped__", False) else obj
            names = tuple(inspect.signature(obj).parameters)
            miss = [p for p in params if p not in names]
            rows.append((not miss, "signature %s.%s%s" % (mpath, dotted,
                                                          "" if not miss else " missing " + ", ".join(miss))))
        except Exception as e:  # noqa: BLE001
            rows.append((False, "signature %s.%s (%r)" % (mpath, dotted, e)))
    for mpath, cname, markers in getattr(mod, "SOURCE_MARKERS", ()):
        if mpath in static_done:
            continue
        try:
            src = inspect.getsource(getattr(importlib.import_module(mpath), cname))
            miss = [k for k in markers if k not in src]
            rows.append((not miss, "source %s.%s%s" % (mpath, cname, "" if not miss else " missing %r" % miss)))
        except Exception as e:  # noqa: BLE001
            rows.append((False, "source %s.%s (%r)" % (mpath, cname, e)))
    return rows


def _merge_targets(names) -> dict:
    merged = {}
    for n in names:
        for mod, fn in adapter_module(n).TARGETS.items():
            prev = merged.get(mod)
            if prev is None:
                merged[mod] = fn
            elif prev is not fn:
                merged[mod] = (lambda a, b: (lambda m: (a(m), b(m))))(prev, fn)
    return merged


def install_auto(adapter: str = "") -> list:
    """Install the hooks this interpreter can use. Returns the adapter names installed. Never raises."""
    choice = (adapter or os.environ.get("RECALL_ACCEL_ADAPTER") or "auto").strip().lower()
    if choice in ("off", "none", "0"):
        print("[recall-accel] RECALL_ACCEL_ADAPTER=off: no hooks", flush=True)
        return []
    eng = engines_importable()
    names = []
    if choice == "auto":
        if eng.get("sglang"):
            names.extend(SGLANG_ALL)
        if eng.get("vllm"):
            names.append("vllm_spec")
    else:
        names = [choice if choice.startswith(("sglang_", "vllm_")) else "sglang_" + choice]
    done = []
    try:
        sg = [n for n in names if n.startswith("sglang_")]
        if sg:
            from . import sglang_dspark

            sglang_dspark.install(_merge_targets(sg))
            done.extend(sg)
        if "vllm_spec" in names:
            from . import vllm_spec

            vllm_spec.install()
            done.append("vllm_spec")
    except Exception as e:  # noqa: BLE001  never take the server down
        print("[recall-accel] install failed: %r" % (e,), flush=True)
    if not done:
        print("[recall-accel] no SGLang / vLLM importable here: no hooks (pid %d)" % os.getpid(), flush=True)
    return done


__all__ = ["ADAPTERS", "choose", "parse_cmdline", "detect_running", "engines_importable", "check_symbols",
           "install_auto", "norm_algorithm"]
_ = sys
