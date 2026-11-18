"""启动命令 → 实例描述（Yee 4.3.0 doctor 的第一步）。

只用标准库；任何异常都不抛到调用方：认不出引擎返回 None，读不到的东西返回空 dict。
实例 dict 的字段照 SPEC 4.3.0 §1.1，所有键都在，未知用 None / 默认值。
"""
from __future__ import annotations

import json
import os
import shlex

try:  # registry 只用到 norm_algorithm，导入失败就用本地归一
    from . import registry as _registry
except Exception:  # pragma: no cover - 兜底
    _registry = None


ENGINES = ("sglang", "vllm", "llama.cpp", "trtllm", "lmdeploy", "mlx_lm", "mlx-serve", "tgi", "ollama",
           "tensorfold", "strata", "maya", "cuteafd", "ktransformers", "unknown")

# env 里只保留这些前缀（大小写不敏感）
ENV_PREFIXES = ("YEE", "RECALL_ACCEL", "SGLANG_D_RECALL", "VLLM_D_RECALL", "DSV41_", "MAYA_", "STRATA_",
                "VLLM_USE_V2_MODEL_RUNNER", "CUDA_VISIBLE_DEVICES", "EP_SIZE", "PYTHONPATH")
# yee.env 只保留这些前缀
YEE_ENV_PREFIXES = ("YEE", "RECALL", "DSV41", "VLLM_D_RECALL", "MAYA", "STRATA")
# 名字里出现这些词的一律丢掉（KEY/TOKEN/SECRET/PASSWORD）
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD")

# 布尔开关（不带值），用来找位置参数时跳过
BOOL_FLAGS = ("--headless", "--enable-expert-parallel", "--enforce-eager", "--disable-cuda-graph",
              "--enable-dp-attention", "--speculative-adaptive", "--recall", "--recall-read-only",
              "--shared-expert-arena", "--no-enable-prefix-caching")


# --------------------------------------------------------------------------- 小工具
def _norm(name):
    """投机头算法名归一（registry.norm_algorithm 的兜底版）。"""
    if _registry is not None:
        try:
            return _registry.norm_algorithm(name)
        except Exception:
            pass
    up = str(name or "").strip().upper().replace("-", "_")
    return up or "NONE"


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _loose_dict(s):
    """``{method:mtp,num_speculative_tokens:3}`` (the shell ate the quotes of a JSON object) -> dict, else None.
    Flat objects only."""
    try:
        s = str(s).strip()
        if not (s.startswith("{") and s.endswith("}")):
            return None
        out = {}
        for part in s[1:-1].split(","):
            if not part.strip():
                continue
            if ":" not in part:
                return None
            k, v = part.split(":", 1)
            k, v = k.strip().strip("'\""), v.strip().strip("'\"")
            if not k or "{" in v or "}" in k:
                return None
            if v.lower() in ("true", "false"):
                out[k] = v.lower() == "true"
            elif _int(v) is not None:
                out[k] = _int(v)
            else:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
        return out or None
    except Exception:  # noqa: BLE001
        return None


def _bool(v):
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on", "enable", "enabled"):
        return True
    if s in ("0", "false", "no", "off", "disable", "disabled"):
        return False
    return bool(v)


def _flag(argv, *names):
    """取 --a b 或 --a=b 的值；没有就 None。"""
    try:
        for i, a in enumerate(argv):
            for n in names:
                if a == n:
                    return argv[i + 1] if i + 1 < len(argv) else None
                if a.startswith(n + "="):
                    return a.split("=", 1)[1]
    except Exception:
        return None
    return None


def _has(argv, *names):
    """开关在不在（--a / --a=v 都算）。"""
    try:
        for a in argv:
            for n in names:
                if a == n or a.startswith(n + "="):
                    return True
    except Exception:
        return False
    return False


def _base(arg):
    s = str(arg or "").replace("\\", "/")
    return s.rsplit("/", 1)[-1]


def _positional(argv, start=0):
    """start 之后第一个位置参数（跳过开关和开关的值）。"""
    try:
        i = int(start)
        while i < len(argv):
            a = argv[i]
            if a.startswith("-"):
                if "=" not in a and a not in BOOL_FLAGS:
                    i += 2
                    continue
                i += 1
                continue
            return a
        return None
    except Exception:
        return None


def _after(argv, marker, exact=True):
    """marker 之后的下标；找不到返回 0。"""
    try:
        for i, a in enumerate(argv):
            if (exact and a == marker) or (not exact and marker in a):
                return i + 1
    except Exception:
        pass
    return 0


def _split_list(text):
    if not text:
        return []
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


# --------------------------------------------------------------------------- 实例模板
def _blank(engine="unknown"):
    return {
        "engine": engine,
        "flavor": None,
        "role": "server",
        "version": None,
        "model": None,
        "port": None,
        "head": {"algorithm": "NONE", "draft_model": None, "num_steps": None, "topk": None,
                 "num_draft_tokens": None, "adaptive": False, "disable_by_batch_size": None, "raw": {}},
        "parallel": {"tp": 1, "pp": 1, "dp": 1, "ep": 1, "nnodes": 1, "node_rank": None,
                     "dp_attention": False, "backend": None},
        "concurrency": {"max_running": None, "cuda_graph_max_bs": None, "cuda_graph": None},
        "router": None,
        "yee": {"env": {}, "installed": None, "version": None},
        "model_info": {},
        "load": None,
        "where": {"host": "", "pid": None, "container": None, "source": "cmd"},
        "argv": [],
        "env": {},
        "notes": [],
    }


def redact(env):
    """只保留 Yee 相关变量，名字含 KEY/TOKEN/SECRET/PASSWORD 的一律去掉。"""
    out = {}
    try:
        for k, v in dict(env or {}).items():
            name = str(k)
            up = name.upper()
            if not any(up.startswith(p) for p in ENV_PREFIXES):
                continue
            if any(w in up for w in SECRET_WORDS):
                continue
            out[name] = v
    except Exception:
        return {}
    return out


def _yee_env(env):
    out = {}
    try:
        for k, v in redact(env).items():
            if any(str(k).upper().startswith(p) for p in YEE_ENV_PREFIXES):
                out[k] = v
    except Exception:
        return {}
    return out


def _apply_env(inst, env):
    """env → 实例的 env / yee.env / yee.installed。"""
    try:
        raw = dict(env or {})
        inst["env"] = redact(raw)
        inst["yee"]["env"] = _yee_env(raw)
        if raw.get("RECALL_ACCEL") is not None or raw.get("YEE") is not None:
            inst["yee"]["installed"] = True
        ver = raw.get("YEE_VERSION") or raw.get("RECALL_ACCEL_VERSION")
        if ver:
            inst["yee"]["version"] = str(ver)
    except Exception:
        pass
    return inst


def _is_mia(argv, env):
    """Mia 镜像识别：DSV41_* 环境变量 / PYTHONPATH 里的 dsv41 adapter / dspark block size。"""
    try:
        for k in dict(env or {}):
            if str(k).upper().startswith("DSV41_"):
                return True
        pyp = str((env or {}).get("PYTHONPATH") or "")
        if "/opt/dsv41/adapter" in pyp:
            return True
        if _has(argv, "--speculative-dspark-block-size"):
            return True
    except Exception:
        pass
    return False


# --------------------------------------------------------------------------- 各引擎
def _flag_multi(argv, name):
    """取开关后面的所有值（--worker-urls a b 或 --worker-urls=a,b）。"""
    out = []
    try:
        i = 0
        while i < len(argv):
            a = argv[i]
            if a == name:
                j = i + 1
                while j < len(argv) and not str(argv[j]).startswith("-"):
                    out.append(str(argv[j]))
                    j += 1
                i = j
                continue
            if a.startswith(name + "="):
                out.extend(_split_list(a.split("=", 1)[1]))
            i += 1
    except Exception:
        pass
    return out


def _positional_after(argv, marker):
    """marker 之后的第一个位置参数。"""
    try:
        i = argv.index(marker)
    except ValueError:
        return None
    j = i + 1
    while j < len(argv):
        a = argv[j]
        if a.startswith("-"):
            if "=" not in a and a not in BOOL_FLAGS:
                j += 2
                continue
            j += 1
            continue
        return a
    return None


def _first_positional(argv):
    """第一个位置参数（跳过 argv[0]）。"""
    try:
        if not argv:
            return None
        j = 1
        while j < len(argv):
            a = argv[j]
            if a.startswith("-"):
                if "=" not in a and a not in BOOL_FLAGS:
                    j += 2
                    continue
                j += 1
                continue
            return a
        return None
    except Exception:
        return None


def _parse_sglang(argv, env):
    inst = _blank("sglang")
    inst["argv"] = list(argv)
    router = any("sglang_router" in a for a in argv)
    if router and any("launch_router" in a for a in argv):
        inst["role"] = "router"
        inst["flavor"] = "router"
    inst["model"] = _flag(argv, "--model-path", "--model")
    for key, names in (("tp", ("--tp-size", "--tp")), ("dp", ("--dp-size", "--dp")),
                       ("pp", ("--pp-size", "--pp")), ("ep", ("--ep-size", "--ep"))):
        v = _int(_flag(argv, *names))
        inst["parallel"][key] = v if v else 1
    inst["parallel"]["nnodes"] = _int(_flag(argv, "--nnodes")) or 1
    inst["parallel"]["node_rank"] = _int(_flag(argv, "--node-rank"))
    inst["parallel"]["dp_attention"] = _has(argv, "--enable-dp-attention")
    alg = _flag(argv, "--speculative-algorithm")
    inst["head"]["algorithm"] = _norm(alg)
    inst["head"]["draft_model"] = _flag(argv, "--speculative-draft-model-path")
    inst["head"]["num_steps"] = _int(_flag(argv, "--speculative-num-steps"))
    inst["head"]["topk"] = _int(_flag(argv, "--speculative-eagle-topk"))
    inst["head"]["num_draft_tokens"] = _int(_flag(argv, "--speculative-num-draft-tokens"))
    inst["head"]["adaptive"] = _has(argv, "--speculative-adaptive")
    inst["head"]["raw"] = {"speculative_algorithm": alg}
    block = _flag(argv, "--speculative-dspark-block-size")
    if block is not None:
        inst["head"]["raw"]["dspark_block_size"] = block
    cg = _int(_flag(argv, "--cuda-graph-max-bs"))
    inst["concurrency"]["cuda_graph_max_bs"] = cg
    if _has(argv, "--disable-cuda-graph"):
        inst["concurrency"]["cuda_graph"] = False
    elif cg is not None:
        inst["concurrency"]["cuda_graph"] = True
    inst["concurrency"]["max_running"] = _int(_flag(argv, "--max-running-requests"))
    inst["port"] = _int(_flag(argv, "--port"))
    mode = _flag(argv, "--disaggregation-mode")
    if mode:
        low = str(mode).lower()
        if low.startswith("decode"):
            inst["role"] = "decode"
        elif low.startswith("prefill"):
            inst["role"] = "prefill"
        inst["head"]["raw"]["disaggregation_mode"] = mode
    if router:
        inst["router"] = {"policy": _flag(argv, "--policy"),
                           "workers": [w for w in _flag_multi(argv, "--worker-urls") if w]}
    if _is_mia(argv, env):
        inst["flavor"] = "mia-dsv41"
    return inst


def _parse_vllm(argv, env):
    inst = _blank("vllm")
    inst["argv"] = list(argv)
    inst["model"] = _flag(argv, "--model") or _positional_after(argv, "serve")
    for key, names in (("tp", ("-tp", "--tensor-parallel-size")), ("pp", ("-pp", "--pipeline-parallel-size")),
                       ("dp", ("-dp", "--data-parallel-size"))):
        v = _int(_flag(argv, *names))
        inst["parallel"][key] = v if v else 1
    if _has(argv, "--enable-expert-parallel"):
        inst["notes"].append("已开专家并行（EP 规模随 TP×DP）")
    inst["parallel"]["backend"] = _flag(argv, "--distributed-executor-backend")
    sc = _flag(argv, "--speculative-config", "--speculative_config")
    if sc:
        obj = None
        try:
            obj = json.loads(sc)
        except Exception:
            obj = None
        if not isinstance(obj, dict):
            obj = _loose_dict(sc)
            if isinstance(obj, dict):
                inst["notes"].append("--speculative-config 的引号被 shell 去掉了，已按 key:value 解析")
        if isinstance(obj, dict):
            inst["head"]["algorithm"] = _norm(obj.get("method"))
            inst["head"]["draft_model"] = obj.get("model")
            inst["head"]["num_draft_tokens"] = _int(obj.get("num_speculative_tokens"))
            inst["head"]["disable_by_batch_size"] = _int(obj.get("disable_by_batch_size"))
            inst["head"]["raw"] = {"speculative_config": obj}
        else:
            inst["head"]["algorithm"] = "NONE"
            inst["head"]["raw"] = {"speculative_config": sc}
            inst["notes"].append("vLLM --speculative-config 不是合法 JSON，已按原文保留")
    inst["concurrency"]["max_running"] = _int(_flag(argv, "--max-num-seqs"))
    if _has(argv, "--enforce-eager"):
        inst["concurrency"]["cuda_graph"] = False
    inst["port"] = _int(_flag(argv, "--port"))
    if _has(argv, "--headless"):
        inst["role"] = "worker"
    kv = _flag(argv, "--kv-transfer-config")
    if kv:
        try:
            kobj = json.loads(kv)
        except Exception:
            kobj = None
        if isinstance(kobj, dict):
            role = str(kobj.get("kv_role") or "")
            if "consumer" in role:
                inst["role"] = "decode"
            elif "producer" in role:
                inst["role"] = "prefill"
            inst["head"]["raw"]["kv_transfer_config"] = kobj
    return inst


def _parse_llama(argv, env):
    inst = _blank("llama.cpp")
    inst["argv"] = list(argv)
    inst["model"] = _flag(argv, "-m", "--model")
    inst["head"]["draft_model"] = _flag(argv, "-md", "--model-draft")
    draft_max = _int(_flag(argv, "--draft-max", "--draft", "--draft-n"))
    inst["head"]["num_steps"] = draft_max
    dmin = _int(_flag(argv, "--draft-min"))
    spec_type = _flag(argv, "--spec-type")
    spec_n = _int(_flag(argv, "--spec-draft-n-max"))
    if spec_type:
        inst["head"]["raw"]["spec_type"] = spec_type
        if str(spec_type).lower().replace("_", "-") in ("draft-mtp", "mtp"):
            inst["head"]["algorithm"] = "MTP"
    if spec_n is not None:
        inst["head"]["num_draft_tokens"] = spec_n
        if inst["head"]["algorithm"] == "NONE":
            inst["head"]["algorithm"] = "MTP"
    if inst["head"]["draft_model"] and inst["head"]["algorithm"] == "NONE":
        inst["head"]["algorithm"] = "DRAFT_MODEL"
    if dmin is not None:
        inst["head"]["raw"]["draft_min"] = dmin
    inst["concurrency"]["max_running"] = _int(_flag(argv, "-np", "--parallel"))
    inst["port"] = _int(_flag(argv, "--port", "--host-port"))
    if _has(argv, "--no-webui"):
        pass
    if not inst["model"]:
        inst["model"] = _first_positional(argv)
    return inst


def _yaml_speculative(text):
    """粗解析 yaml 里的 speculative_config（没有 yaml 库时的兜底）。"""
    out = {}
    try:
        lines = str(text or "").splitlines()
        base = None
        for ln in lines:
            if ln.strip().startswith("speculative_config:"):
                base = len(ln) - len(ln.lstrip())
                continue
            if base is None:
                continue
            if not ln.strip() or ln.strip().startswith("#"):
                continue
            indent = len(ln) - len(ln.lstrip())
            if indent <= base:
                break
            body = ln.strip()
            if ":" not in body:
                continue
            k, v = body.split(":", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k == "decoding_type":
                out["decoding_type"] = v
            elif k == "max_draft_len":
                out["max_draft_len"] = v
            elif k == "draft_len_schedule":
                out["draft_len_schedule"] = v
            elif k in ("model", "draft_model"):
                out[k] = v
    except Exception:
        return {}
    return out


def _parse_trtllm(argv, env):
    inst = _blank("trtllm")
    inst["argv"] = list(argv)
    inst["model"] = _flag(argv, "--model", "--model_dir") or _first_positional(argv)
    for key, names in (("tp", ("--tp_size", "--tp-size")), ("pp", ("--pp_size", "--pp-size")),
                       ("ep", ("--ep_size", "--ep-size"))):
        v = _int(_flag(argv, *names))
        inst["parallel"][key] = v if v else 1
    inst["concurrency"]["max_running"] = _int(_flag(argv, "--max_batch_size"))
    extra = _flag(argv, "--extra_llm_api_options")
    if extra:
        inst["head"]["raw"]["extra_llm_api_options"] = extra
        spec = _yaml_speculative(extra)
        if spec:
            inst["head"]["raw"]["speculative_config"] = spec
            inst["head"]["algorithm"] = _norm(spec.get("decoding_type"))
            inst["head"]["num_draft_tokens"] = _int(spec.get("max_draft_len"))
            if spec.get("model"):
                inst["head"]["draft_model"] = spec["model"]
    return inst


def _parse_lmdeploy(argv, env):
    inst = _blank("lmdeploy")
    inst["argv"] = list(argv)
    inst["model"] = _flag(argv, "--model-name", "--model") or _positional_after(argv, "api_server")
    v = _int(_flag(argv, "--tp", "--tp-size"))
    inst["parallel"]["tp"] = v if v else 1
    alg = _flag(argv, "--speculative-algorithm")
    inst["head"]["algorithm"] = _norm(alg)
    inst["head"]["draft_model"] = _flag(argv, "--speculative-draft-model")
    inst["head"]["num_draft_tokens"] = _int(_flag(argv, "--speculative-num-draft-tokens"))
    return inst


def _parse_mlx_lm(argv, env):
    inst = _blank("mlx_lm")
    inst["argv"] = list(argv)
    inst["model"] = _flag(argv, "--model")
    inst["head"]["draft_model"] = _flag(argv, "--draft-model")
    inst["head"]["num_draft_tokens"] = _int(_flag(argv, "--num-draft-tokens"))
    if inst["head"]["draft_model"] or inst["head"]["num_draft_tokens"] is not None:
        inst["head"]["algorithm"] = "DRAFT_MODEL"
    return inst


def _parse_mlx_serve(argv, env):
    inst = _blank("mlx-serve")
    inst["argv"] = list(argv)
    inst["model"] = _first_positional(argv)
    inst["port"] = _int(_flag(argv, "--port"))
    try:
        raw = dict(env or {})
        if raw.get("MLX_SERVE_D_RECALL") not in (None, "", "0", "off"):
            inst["notes"].append("MLX_SERVE_D_RECALL 已开：召回补丁已生效")
    except Exception:
        pass
    return inst


def _parse_tgi(argv, env):
    inst = _blank("tgi")
    inst["argv"] = list(argv)
    inst["model"] = _flag(argv, "--model-id", "--model_name", "--model")
    inst["notes"].append("num_shard=%s" % (_flag(argv, "--num-shard") or "-"))
    spec = _int(_flag(argv, "--speculate"))
    inst["head"]["num_draft_tokens"] = spec
    if spec:
        inst["head"]["algorithm"] = "SPECULATE"
    inst["concurrency"]["max_running"] = _int(_flag(argv, "--max-concurrent-requests"))
    return inst


def _parse_ollama(argv, env):
    inst = _blank("ollama")
    inst["argv"] = list(argv)
    inst["port"] = _int(_flag(argv, "--port"))
    return inst


def _parse_tensorfold(argv, env):
    inst = _blank("tensorfold")
    inst["argv"] = list(argv)
    inst["model"] = _positional_after(argv, "serve") or _first_positional(argv)
    if _has(argv, "--recall", "--recall-read-only", "--recall-gib", "--recall-dir"):
        inst["head"]["algorithm"] = "RECALL"
    inst["head"]["raw"]["recall_dir"] = _flag(argv, "--recall-dir")
    return inst


def _parse_strata(argv, env):
    inst = _blank("strata")
    inst["argv"] = list(argv)
    spec = _int(_flag(argv, "--spec"))
    inst["head"]["num_draft_tokens"] = spec
    if spec is not None:
        inst["head"]["raw"]["spec"] = spec
        if spec > 0:
            inst["head"]["algorithm"] = "MTP"
        else:
            inst["notes"].append("strata --spec 0：MTP 草稿已关闭")
    batch = _int(_flag(argv, "--batch"))
    inst["concurrency"]["max_running"] = batch
    if _has(argv, "--recall"):
        inst["notes"].append("已开 --recall")
    inst["head"]["raw"]["recall_dir"] = _flag(argv, "--recall-dir")
    return inst


def _parse_maya(argv, env):
    inst = _blank("maya")
    inst["argv"] = list(argv)
    inst["model"] = _flag(argv, "--glm-pack", "--model")
    par = _int(_flag(argv, "--parallel"))
    inst["concurrency"]["max_running"] = par
    try:
        raw = dict(env or {})
        mtp = raw.get("MAYA_GLM_MULTI_MTP")
        if mtp is not None:
            inst["head"]["algorithm"] = "MTP"
            inst["head"]["num_draft_tokens"] = _int(mtp)
            inst["head"]["raw"]["MAYA_GLM_MULTI_MTP"] = mtp
        if raw.get("MAYA_GLM_MULTI_MTP_MAX") is not None:
            inst["head"]["raw"]["MAYA_GLM_MULTI_MTP_MAX"] = raw.get("MAYA_GLM_MULTI_MTP_MAX")
        if raw.get("RECALL_ACCEL") not in (None, "", "0", "off"):
            inst["notes"].append("RECALL_ACCEL 已开")
    except Exception:
        pass
    if "yee" in _base(argv[0] if argv else ""):
        inst["notes"].append("Maya 二进制名含 yee：已带召回补丁")
    return inst


def _parse_cuteafd(argv, env):
    inst = _blank("cuteafd")
    inst["argv"] = list(argv)
    inst["model"] = _flag(argv, "--model", "--model-path") or _first_positional(argv)
    inst["head"]["raw"]["config"] = _flag(argv, "--config")
    try:
        raw = dict(env or {})
        if raw.get("CUTEAFD_RECALL_DIR"):
            inst["notes"].append("CUTEAFD_RECALL_DIR 已设：召回目录就绪（开关是目录里的 enabled 文件）")
    except Exception:
        pass
    return inst


def _parse_ktransformers(argv, env):
    inst = _blank("ktransformers")
    inst["argv"] = list(argv)
    inst["model"] = _flag(argv, "--model_name", "--model")
    return inst


# --------------------------------------------------------------------------- 识别
def _is_sglang(argv):
    joined = " ".join(argv)
    if "sglang_router" in joined:
        return True
    if "sglang" in joined and ("launch_server" in joined or "sglang.launch" in joined
                               or "sglang serve" in joined or " serve " in joined):
        return True
    return False


def _is_vllm(argv):
    joined = " ".join(argv)
    return "vllm" in joined and ("serve" in joined or "api_server" in joined or "vllm.entrypoints" in joined)


def _is_llama(argv):
    base = _base(argv[0] if argv else "")
    if "llama-server" in base or "llama-server" in " ".join(argv):
        return True
    if base in ("server", "llama-server") or base.endswith("/server"):
        if _has(argv, "-m", "--model"):
            return True
        return False
    return False


def _is_trtllm(argv):
    base = _base(argv[0] if argv else "")
    if "trtllm-serve" in base or "trtllm_serve" in base:
        return True
    return any(a.endswith("trtllm-serve") for a in argv)


def _is_lmdeploy(argv):
    joined = " ".join(argv)
    return "lmdeploy" in joined and ("api_server" in joined or "serve" in joined)


def _is_mlx_lm(argv):
    joined = " ".join(argv)
    return "mlx_lm.server" in joined or "mlx_lm server" in joined or "mlx_lm.generate" in joined


def _is_mlx_serve(argv):
    base = _base(argv[0] if argv else "")
    return "mlx-serve" in base or "mlx_serve" in base


def _is_tgi(argv):
    base = _base(argv[0] if argv else "")
    return "text-generation-launcher" in base


def _is_ollama(argv):
    joined = " ".join(argv)
    if "ollama serve" in joined or "ollama runner" in joined:
        return True
    base = _base(argv[0] if argv else "")
    return base in ("ollama", "ollama.exe")


def _is_tensorfold(argv):
    joined = " ".join(argv)
    return "tensorfold" in joined and "serve" in joined


def _is_strata(argv):
    base = _base(argv[0] if argv else "")
    if base == "strata" or base.endswith("/strata"):
        return True
    if "strata" in base and (_has(argv, "--spec", "--shared-expert-arena")):
        return True
    return False


def _is_maya(argv):
    base = _base(argv[0] if argv else "")
    if "maya" not in base:
        return False
    return _has(argv, "--glm-pack", "--parallel") or "yee" in base


def _is_cuteafd(argv):
    base = _base(argv[0] if argv else "")
    return "cuteafd" in base or "cuteafd" in " ".join(argv)


def _is_ktransformers(argv):
    joined = " ".join(argv)
    return "ktransformers.server" in joined or _base(argv[0] if argv else "") in ("ktransformers", "ktransformers.exe")


# 顺序敏感：maya / strata / cuteafd 看 basename，sglang / vllm 看命令行
_ENGINE_CHECKS = (
    ("sglang", _is_sglang, _parse_sglang),
    ("vllm", _is_vllm, _parse_vllm),
    ("llama.cpp", _is_llama, _parse_llama),
    ("trtllm", _is_trtllm, _parse_trtllm),
    ("lmdeploy", _is_lmdeploy, _parse_lmdeploy),
    ("mlx_lm", _is_mlx_lm, _parse_mlx_lm),
    ("mlx-serve", _is_mlx_serve, _parse_mlx_serve),
    ("tgi", _is_tgi, _parse_tgi),
    ("ollama", _is_ollama, _parse_ollama),
    ("tensorfold", _is_tensorfold, _parse_tensorfold),
    ("strata", _is_strata, _parse_strata),
    ("maya", _is_maya, _parse_maya),
    ("cuteafd", _is_cuteafd, _parse_cuteafd),
    ("ktransformers", _is_ktransformers, _parse_ktransformers),
)


def parse_launch(cmd, environ=None):
    """启动命令（字符串或 argv）→ 实例 dict；认不出引擎返回 None。"""
    try:
        argv = shlex.split(cmd) if isinstance(cmd, str) else [str(a) for a in (cmd or [])]
    except Exception:
        try:
            argv = [str(a) for a in (cmd or [])]
        except Exception:
            return None
    if not argv:
        return None
    env = dict(environ or {})
    for name, check, parse in _ENGINE_CHECKS:
        try:
            if not check(argv):
                continue
        except Exception:
            continue
        try:
            inst = parse(argv, env)
        except Exception:
            return None
        if not inst:
            return None
        inst["argv"] = list(argv)
        _apply_env(inst, env)
        if inst["engine"] not in ENGINES:
            inst["engine"] = "unknown"
        return inst
    return None


def from_maya_config(obj_or_path):
    """Maya 的 JSON 启动配置（dict 或路径）→ 实例。"""
    try:
        obj = obj_or_path
        if isinstance(obj, (str, bytes, os.PathLike)):
            with open(obj, "r", encoding="utf-8") as f:
                obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        exe = str(obj.get("exe") or "")
        args = [str(a) for a in (obj.get("args") or [])]
        env = {str(k): v for k, v in dict(obj.get("env") or {}).items()}
        argv = ([exe] if exe else []) + args
        inst = _blank("maya")
        inst["argv"] = list(argv)
        inst["model"] = obj.get("model_name") or obj.get("model") or _flag(args, "--glm-pack", "--model")
        inst["port"] = _int(obj.get("port"))
        par = _int(_flag(args, "--parallel"))
        if par is None:
            par = _int(obj.get("parallel"))
        inst["concurrency"]["max_running"] = par
        gpu = obj.get("gpu")
        if isinstance(gpu, list) and gpu:
            inst["parallel"]["pp"] = len(gpu)  # Maya splits the layers over the cards (--layer-split)
            inst["notes"].append("按层切到 %d 张卡" % len(gpu))
        installer = obj.get("installer") if isinstance(obj.get("installer"), dict) else {}
        draft = str(installer.get("draft") or "").lower()
        mtp = env.get("MAYA_GLM_MULTI_MTP")
        if draft == "mtp" or mtp is not None:
            inst["head"]["algorithm"] = "MTP"
            inst["head"]["num_draft_tokens"] = _int(mtp)
            inst["head"]["raw"]["installer_draft"] = draft or None
            if mtp is not None:
                inst["head"]["raw"]["MAYA_GLM_MULTI_MTP"] = mtp
        else:
            inst["notes"].append("Maya 配置里没有 MTP（installer.draft 不是 mtp，也没有 MAYA_GLM_MULTI_MTP）")
        if env.get("RECALL_ACCEL") not in (None, "", "0", "off"):
            inst["notes"].append("RECALL_ACCEL 已开")
        if "yee" in _base(exe):
            inst["notes"].append("Maya 二进制名含 yee：已带召回补丁")
        _apply_env(inst, env)
        return inst
    except Exception:
        return None


def from_server_info(engine, info):
    """SGLang /get_server_info（或同类接口）的 JSON → 实例。"""
    try:
        obj = info if isinstance(info, dict) else {}
        args = obj.get("server_args")
        if isinstance(args, dict):
            argv = []
            for k, v in args.items():
                if isinstance(v, bool):
                    if v:
                        argv.append("--%s" % str(k).replace("_", "-"))
                    continue
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    for x in v:
                        argv.extend(["--%s" % str(k).replace("_", "-"), str(x)])
                    continue
                argv.extend(["--%s" % str(k).replace("_", "-"), str(v)])
            argv = [str(a) for a in argv]
        elif isinstance(args, (list, tuple)):
            argv = [str(a) for a in args]
        elif isinstance(obj.get("args"), (list, tuple)):
            argv = [str(a) for a in obj["args"]]
        else:
            argv = []
        env = {str(k): v for k, v in dict(obj.get("env") or {}).items()}
        inst = _blank(engine or "unknown")
        inst["argv"] = argv
        if argv:
            try:
                marker = {"sglang": "sglang.launch_server", "vllm": "vllm serve"}.get(engine or "", "")
                probe = ([marker] + argv) if marker else argv
                inner = parse_launch(probe, env)
                if inner and inner.get("engine") not in (None, "unknown"):
                    inner["argv"] = list(argv)
                    inner["where"]["source"] = "url"
                    inner["env"] = redact(env) or inner["env"]
                    inner["yee"]["env"] = _yee_env(env)
                    if obj.get("port") and not inner.get("port"):
                        inner["port"] = _int(obj.get("port"))
                    return inner
            except Exception:
                pass
        inst["model"] = obj.get("model_path") or obj.get("model")
        inst["port"] = _int(obj.get("port"))
        alg = obj.get("speculative_algorithm")
        if alg:
            inst["head"]["algorithm"] = _norm(alg)
        _apply_env(inst, env)
        return inst
    except Exception:
        return None


def model_info(model_path):
    """读 <model>/config.json → {"arch","mtp_layers","model_type"}；读不到返回 {}。"""
    out = {}
    try:
        if not model_path:
            return {}
        path = os.path.join(str(model_path), "config.json")
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            return {}
        archs = cfg.get("architectures")
        if isinstance(archs, list) and archs:
            out["arch"] = str(archs[0])
        mt = cfg.get("model_type")
        if mt is not None:
            out["model_type"] = str(mt)
        for key in ("num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers", "num_mtp_modules"):
            v = cfg.get(key)
            if isinstance(v, bool):
                continue
            if isinstance(v, int):
                out["mtp_layers"] = int(v)
                break
            iv = _int(v)
            if iv is not None:
                out["mtp_layers"] = iv
                break
        return out
    except Exception:
        return {}


__all__ = ["ENGINES", "parse_launch", "from_maya_config", "from_server_info", "model_info", "redact"]
