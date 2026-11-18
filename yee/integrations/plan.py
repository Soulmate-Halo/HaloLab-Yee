"""Yee 4.3 deployment plan: instances (``launch.parse_launch`` / ``discover.collect``) -> verdict, feature states
and the exact parameter / env / patch changes that make Yee accelerate them.

``analyze(instances)`` applies the rules of ``docs/deploy.md`` (H head, P patch engines, C concurrency / multi-stream,
K cluster) and returns a JSON-able plan; ``render(plan)`` is the Chinese report of ``yee doctor``;
``startup_lines()`` are the ``[yee] 4.3.0 plan:`` lines a server prints at start; ``effect_lines()`` are the
``[yee] effect:`` lines printed while it runs. Standard library only; nothing here raises.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

VERSION = "4.3.0"
SHOWN = "YEE_PLAN_SHOWN"  # set by print_startup(); child processes inherit it and stay quiet
EXIT_OK, EXIT_ACTION, EXIT_UNSUPPORTED = 0, 1, 3

AUTO_ENGINES = ("sglang", "vllm", "trtllm", "mlx_lm")  # Python hooks, installed by the .pth / sitecustomize
RECALL_AUTO = ("sglang", "vllm")
PATCH_ENGINES = ("llama.cpp", "strata", "maya", "mlx-serve", "cuteafd")
UNSUPPORTED = ("tgi", "ollama", "ktransformers", "lmdeploy", "unknown")
EAGLE_FAMILY = ("EAGLE", "EAGLE3", "NEXTN", "MTP", "STANDALONE", "UNO")
MASTER = ("YEE", "RECALL_ACCEL", "SGLANG_D_RECALL", "VLLM_D_RECALL", "DSV41_RECALL")
DIR_VARS = ("RECALL_ACCEL_DIR", "SGLANG_D_RECALL_DIR", "VLLM_D_RECALL_DIR", "DSV41_RECALL_DIR")
BACKEND_VARS = ("RECALL_ACCEL_BACKEND", "SGLANG_D_RECALL_BACKEND", "VLLM_D_RECALL_BACKEND", "RECALL_BACKEND")
SGLANG_MAX_BS = 64
KAD_K = 12
_FALSE = ("0", "false", "no", "off")

RULES = {
    "H0": "没开投机头", "H1": "SGLang EAGLE 树形草稿（topk>1）", "H2": "SGLang FROZEN_KV_MTP",
    "H3": "vLLM 的头 Yee 不挂", "H4": "vLLM 需要 V2 model runner", "H5": "头不在支持矩阵里",
    "H6": "strata 没开 MTP", "H7": "Maya 没开 MTP", "H8": "llama.cpp 没有草稿", "H9": "引擎不支持，建议换引擎",
    "P1": "llama.cpp 补丁", "P2": "strata 补丁", "P3": "Maya 补丁", "P4": "mlx-serve / cuteafd / TensorFold",
    "C1": "并发超过 CUDA graph 批上限", "C2": "CUDA graph 批上限未知", "C3": "投机时关了 CUDA graph",
    "C4": "单批超过 Yee 的 64 行上限", "C5": "vLLM 按批大小关闭投机", "C6": "Maya 多路时 MTP 窗口关闭",
    "C7": "strata 多行批", "C8": "高并发", "C9": "SGLang KAD 前提", "C10": "TensorRT-LLM 没配草稿长度表",
    "C11": "KAD 启动 K 不是 12",
    "K1": "多机每个节点都要装", "K2": "容器里装与持久化", "K3": "同机多副本共享记录", "K4": "跨机多副本亲和路由",
    "K5": "PD 分离的 prefill 节点", "K6": "路由进程", "K7": "Yee 版本不一致", "K8": "专家预算前提",
    "E1": "投机路径没跑到", "E2": "召回贡献很小", "E3": "召回出错",
}
VERDICT_ZH = {"accelerated": "可加速", "needs-change": "要改参数才能加速", "needs-patch": "要打补丁重编译",
              "unsupported": "不支持（见换引擎建议）", "not-needed": "无需处理"}
STATE_ZH = {"on": "开", "off": "关", "needs-change": "要改", "n/a": "不适用"}
KIND_ZH = {"flag": "参数", "env": "环境变量", "patch": "补丁", "build": "构建", "install": "安装", "route": "路由",
           "config": "配置", "restart": "重启"}
SOURCE_ZH = {"proc": "本机进程", "local": "本机进程", "docker": "docker", "ssh": "ssh", "url": "URL", "k8s": "K8s",
             "config": "配置文件", "cmd": "命令行"}
FEATURES = (("recall", "召回"), ("kad", "KAD"), ("expert_budget", "专家预算"), ("tree", "树形"),
            ("shared_records", "记录共享"))
SHARING_ZH = {"single": "单副本", "same-host": "同机多副本", "cross-host": "跨机多副本"}


# ----------------------------------------------------------------------------------------------- helpers
def _d(x):
    return x if isinstance(x, dict) else {}


def _int(x):
    try:
        if x is None or isinstance(x, bool):
            return None
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


def _norm_alg(name):
    try:
        from .registry import norm_algorithm

        return norm_algorithm(name)
    except Exception:  # noqa: BLE001
        return str(name or "NONE").strip().upper() or "NONE"


def normalize(inst) -> dict:
    """A copy of ``inst`` with every key of the launch.py schema present."""
    i = copy.deepcopy(inst) if isinstance(inst, dict) else {}
    i["engine"] = str(i.get("engine") or "unknown")
    i.setdefault("flavor", None)
    i["role"] = i.get("role") or "server"
    for k in ("version", "model", "port", "router", "load"):
        i.setdefault(k, None)
    h = _d(i.get("head"))
    h["algorithm"] = _norm_alg(h.get("algorithm"))
    for k in ("draft_model", "num_steps", "topk", "num_draft_tokens", "disable_by_batch_size"):
        h.setdefault(k, None)
    h["adaptive"] = bool(h.get("adaptive"))
    h["raw"] = h.get("raw") if isinstance(h.get("raw"), dict) else {}
    i["head"] = h
    p = _d(i.get("parallel"))
    for k in ("tp", "pp", "dp", "ep", "nnodes"):
        p[k] = _int(p.get(k)) or 1
    p.setdefault("node_rank", None)
    p["dp_attention"] = bool(p.get("dp_attention"))
    p.setdefault("backend", None)
    i["parallel"] = p
    c = _d(i.get("concurrency"))
    for k in ("max_running", "cuda_graph_max_bs", "cuda_graph"):
        c.setdefault(k, None)
    i["concurrency"] = c
    y = _d(i.get("yee"))
    y["env"] = _d(y.get("env"))
    y.setdefault("installed", None)
    y.setdefault("version", None)
    i["yee"] = y
    i["model_info"] = _d(i.get("model_info"))
    w = _d(i.get("where"))
    w["host"] = w.get("host") or "localhost"
    w.setdefault("pid", None)
    w.setdefault("container", None)
    w["source"] = w.get("source") or "cmd"
    i["where"] = w
    i["argv"] = [str(a) for a in (i.get("argv") or [])]
    i["env"] = _d(i.get("env"))
    i["notes"] = list(i.get("notes") or [])
    return i


def _env(i) -> dict:
    out = dict(_d(i.get("env")))
    out.update(_d(_d(i.get("yee")).get("env")))
    return out


def _val(env, name):
    v = env.get(name)
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def _off(env, names) -> bool:
    return any((_val(env, n) or "x").lower() in _FALSE for n in names)


def _peak(i):
    c, ld = i["concurrency"], _d(i.get("load"))
    vals = [v for v in (_int(c.get("max_running")), _int(ld.get("running"))) if v]
    return max(vals) if vals else None


def _spec_on(i) -> bool:
    return i["head"]["algorithm"] not in ("NONE", "")


def _ch(rule, kind, set_, why, optional=False) -> dict:
    return {"rule": rule, "kind": kind, "set": set_, "why": why, "optional": bool(optional)}


def _feat(state, why) -> dict:
    return {"state": state, "why": why}


def _manifest(base) -> dict:
    try:
        from .. import patches

        return _d(_d(patches.load_manifest().get("bases")).get(base))
    except Exception:  # noqa: BLE001
        return {}


def _argv0(i) -> str:
    return os.path.basename(i["argv"][0].replace("\\", "/")) if i["argv"] else ""


def _patched(i) -> bool:
    if i["yee"].get("installed") is True:
        return True
    if "yee" in _argv0(i).lower():
        return True
    if i["engine"] == "strata" and "--recall" in i["argv"]:
        return True
    return any("[yee-recall]" in str(n) for n in i["notes"])


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = _find_key(v, key)
            if hit is not None:
                return hit
    return None


def _model_label(i) -> str:
    m = str(i.get("model") or "").replace("\\", "/").rstrip("/")
    if m:
        return m.rsplit("/", 1)[-1] or m
    return "%s:%s" % (i["engine"], i.get("port") or "?")


def _mtp_flags(engine, mtp):
    """[(set, why)] that turn on a speculative head for an engine without one."""
    out = []
    has = ("模型自带 %d 层 MTP/NextN，开 MTP 不需要额外草稿模型" % mtp) if mtp else "模型自带 MTP/NextN 层时用"
    if engine == "sglang":
        if mtp is None or mtp > 0:
            out.append(("--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 "
                        "--speculative-num-draft-tokens 4", has))
        if not mtp:
            out.append(("--speculative-algorithm NGRAM", "模型没有 MTP 层时用：NGRAM 不需要草稿模型，Yee 能挂"))
    elif engine == "vllm":
        if mtp is None or mtp > 0:
            out.append(("--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'", has))
        if not mtp:
            out.append(("--speculative-config '{\"method\":\"eagle3\",\"model\":\"<EAGLE3 草稿目录>\","
                        "\"num_speculative_tokens\":3}'",
                        "模型没有 MTP 层时用 EAGLE3 草稿；vLLM 原生 ngram 也能跑，但 Yee 不挂 ngram"))
    elif engine == "trtllm":
        if mtp is None or mtp > 0:
            out.append(("--extra_llm_api_options 的 yaml 里加 speculative_config: {decoding_type: MTP, "
                        "num_nextn_predict_layers: 3}", has))
        if not mtp:
            out.append(("--extra_llm_api_options 的 yaml 里配 EAGLE3 草稿（speculative_config 的 decoding_type: Eagle、"
                        "speculative_model_dir: <EAGLE3 草稿目录>、max_draft_len: 3）", "模型没有 MTP 层时用"))
    elif engine == "mlx_lm":
        out.append(("--draft-model <同系列小模型> --num-draft-tokens 3", "mlx-lm 的投机解码靠草稿模型"))
    return out


# ----------------------------------------------------------------------------------------------- per instance
def _rules_head(i, changes, notes):
    e, h = i["engine"], i["head"]
    alg, mtp = h["algorithm"], _int(i["model_info"].get("mtp_layers"))
    env = _env(i)
    if e in AUTO_ENGINES and alg == "NONE":
        for s, why in _mtp_flags(e, mtp):
            changes.append(_ch("H0", "config" if e == "trtllm" else "flag", s,
                               "没开投机头，Yee 没有可以加速的路径；" + why))
        return
    if e == "sglang":
        try:
            from .registry import ADAPTERS
        except Exception:  # noqa: BLE001
            ADAPTERS = {}
        table = _d(ADAPTERS.get("sglang"))
        topk = _int(h.get("topk"))
        if alg in EAGLE_FAMILY and topk and topk > 1:
            steps = _int(h.get("num_steps"))
            n = str(steps + 1) if steps else "<num-steps+1>"
            changes.append(_ch("H1", "flag", "--speculative-eagle-topk 1 --speculative-num-draft-tokens %s" % n,
                               "topk=%d 是树形草稿，Yee 的召回只接 topk=1 的链式草稿" % topk))
        elif alg == "FROZEN_KV_MTP":
            changes.append(_ch("H2", "flag", "--speculative-algorithm EAGLE",
                               "FROZEN_KV_MTP 自己组核对输入，Yee 挂不上；MTP 模型用 EAGLE（与 NEXTN 同路）"))
        elif table and alg not in table:
            changes.append(_ch("H5", "flag", "--speculative-algorithm EAGLE（MTP 模型）或 --speculative-algorithm NGRAM",
                               "%s 不在 Yee 的 SGLang 支持矩阵里，钩子不会参与" % alg))
    elif e == "vllm":
        if alg not in ("DSPARK", "DFLASH", "EAGLE", "EAGLE3", "MTP"):
            for s, why in _mtp_flags("vllm", mtp):
                changes.append(_ch("H3", "flag", s, "vLLM 的 %s 头 Yee 不挂（V2 runner 没有对应 speculator），"
                                                    "原生照常跑；%s" % (alg.lower(), why)))
        elif (_val(env, "VLLM_USE_V2_MODEL_RUNNER") or "").lower() not in ("1", "true"):
            changes.append(_ch("H4", "env", "VLLM_USE_V2_MODEL_RUNNER=1",
                               "Yee 的 vLLM 钩子（vllm_spec / vllm_d_recall）装在 V2 GPU model runner"
                               "（vllm/v1/worker/gpu）上，V1 runner 下不生效"))
    elif e == "strata":
        if alg == "NONE":
            changes.append(_ch("H6", "flag", "--spec 4", "strata 的召回替换 MTP 窗口里的草稿，--spec 0 时没有窗口"))
    elif e == "maya":
        if alg == "NONE":
            changes.append(_ch("H7", "env", "MAYA_GLM_MULTI_MTP=3",
                               "Maya 的召回只在 MTP 验证窗口里起作用；需要带 NextN 层的 GGUF"))
    elif e == "llama.cpp":
        if alg == "NONE":
            changes.append(_ch("H8", "flag", "--spec-type draft-mtp --spec-draft-n-max 3",
                               "没有草稿；模型 GGUF 带 MTP 层时用（见 docs/llama-mtp.md）"))
            changes.append(_ch("H8", "flag", "-md <草稿 GGUF> --draft-max 8", "没有 MTP 层时用同系列小模型做草稿"))


def _rules_unsupported(i, changes):
    alt = {
        "ollama": ("llama-server -m <同一个 gguf> -ngl 99 -np <并发数> --spec-type draft-mtp --spec-draft-n-max 3",
                   "Ollama 没有投机解码接口；同一个 GGUF 换 llama.cpp 的 llama-server，再打 llama.cpp-recall 补丁"),
        "tgi": ("vllm serve <模型> --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' 或 "
                "python -m sglang.launch_server --model-path <模型> --speculative-algorithm EAGLE",
                "TGI 的 --speculate 没有可挂钩的点；换 vLLM 或 SGLang"),
        "ktransformers": ("python -m sglang.launch_server --model-path <模型> --speculative-algorithm EAGLE",
                          "KTransformers 没有可挂钩的投机路径；换 SGLang"),
        "lmdeploy": ("vllm serve <模型> --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'",
                     "LMDeploy 的投机按常量草稿数构建，Yee 只能统计不能加速，原生照常跑；要用召回换 vLLM 或 SGLang"),
    }.get(i["engine"], ("vLLM / SGLang / llama.cpp 之一（见 docs/deploy.md 适配矩阵）",
                        "认不出这个推理引擎；用 --cmd 给出完整启动命令再诊断一次"))
    changes.append(_ch("H9", "config", alt[0], alt[1]))


def _rules_patch(i, changes, notes):
    e, env = i["engine"], _env(i)
    if e == "tensorfold":
        if "--recall" not in i["argv"] and i["head"]["algorithm"] != "RECALL":
            changes.append(_ch("P4", "flag", "--recall", "TensorFold ≥ 1.0.12 自带 Recall，启动时加 --recall 才开"
                                                         "（参数清单：yee tensorfold args）"))
        return True
    base = {"llama.cpp": ("P1", "llama.cpp-recall", "<llama.cpp 源码根>"),
            "strata": ("P2", "strata-recall", "<strata 源码根>"),
            "maya": ("P3", "maya-recall", "<Maya 源码根>"),
            "mlx-serve": ("P4", "mlx-serve", "<mlx-serve 源码根>"),
            "cuteafd": ("P4", "cuteafd", "<cuteafd 源码根>")}.get(e)
    if base is None:
        return True
    rule, name, src = base
    patched = _patched(i)
    m = _manifest(name)
    if not patched:
        changes.append(_ch(rule, "patch", "yee patch apply %s --src %s" % (name, src),
                           "%s 是编译型引擎，召回要打补丁重编译（先加 --check 试一次）" % e))
        build = "cmake --build build --target maya" if e == "maya" else (
            m.get("rebuild") or "按 yee patch show %s 里的说明重编" % name)
        changes.append(_ch(rule, "build", build, "补丁要重编译后才生效"))
        if e == "llama.cpp":
            notes.append("KAD 用补丁 llama.cpp-kad，MTP 草稿头截词表用 llama.cpp-mtp-vocab（yee patch list）")
        notes.append("若这个二进制已经打过 Yee 补丁，忽略补丁与构建两项")
    want = _d(m.get("env")) or ({"RECALL_ACCEL": "1"} if e in ("strata", "maya") else {})
    if "--recall" not in i["argv"] and _val(env, "YEE_RECALL") != "1":
        for k, v in sorted(want.items()):
            if _val(env, k) != str(v):
                changes.append(_ch(rule, "env", "%s=%s" % (k, v), "补丁里召回默认关，设了才开"))
    if e == "maya" and not any(_val(env, n) for n in DIR_VARS):
        changes.append(_ch(rule, "env", "RECALL_ACCEL_DIR=<记录目录>", "记录落盘才能跨重启保留、多副本共享", True))
    return patched


def _rules_conc(i, changes, notes):
    e, h, c, env = i["engine"], i["head"], i["concurrency"], _env(i)
    peak, spec = _peak(i), _spec_on(i)
    cgmb = _int(c.get("cuda_graph_max_bs"))
    if e == "sglang" and spec:
        if peak and cgmb and peak > cgmb:
            changes.append(_ch("C1", "flag", "--cuda-graph-max-bs %d" % min(peak, 512),
                               "峰值并发 %d 大于 CUDA graph 批上限 %d，超出的批退出 CUDA graph，投机步明显变慢"
                               % (peak, cgmb)))
        elif cgmb is None and c.get("cuda_graph") is not False:
            notes.append("C2：开投机时 SGLang 的 CUDA graph 批上限默认偏小，确认 --cuda-graph-max-bs ≥ 峰值并发")
    if spec and c.get("cuda_graph") is False:
        if e == "sglang":
            changes.append(_ch("C3", "flag", "去掉 --disable-cuda-graph", "投机步不走 CUDA graph 时每步开销大，加速被吃掉"))
        elif e == "vllm":
            changes.append(_ch("C3", "flag", "去掉 --enforce-eager", "投机步不走 CUDA graph 时每步开销大，加速被吃掉"))
    if e == "sglang" and peak and peak > SGLANG_MAX_BS:
        notes.append("C4：单批超过 %d 行时 Yee 召回整批跳过（sglang_dspark.MAX_BS），%d 行以内照常"
                     % (SGLANG_MAX_BS, SGLANG_MAX_BS))
    if e == "vllm" and spec:
        n = _int(h.get("disable_by_batch_size"))
        if n and (peak is None or peak > n):
            base = h["raw"].get("speculative_config") if isinstance(h["raw"].get("speculative_config"), dict) \
                else h["raw"]
            cfg = dict(base) if base else {"method": h["algorithm"].lower()}
            if peak:
                cfg["disable_by_batch_size"] = peak
            else:
                cfg.pop("disable_by_batch_size", None)
            changes.append(_ch("C5", "flag", "--speculative-config '%s'" % json.dumps(cfg, separators=(",", ":")),
                               "运行中请求超过 %d 时投机整体关闭，Yee 也不出草稿；GPU 有余量时提高或去掉，"
                               "先用 yee bench 对比" % n, True))
    if e == "maya":
        mx = _int(_val(env, "MAYA_GLM_MULTI_MTP_MAX"))
        mx = 4 if mx is None else mx
        par = _int(c.get("max_running"))
        if par == 1:
            notes.append("Maya --parallel 1 不走多槽调度器，召回不接")
        if mx > 0 and par and par > mx:
            changes.append(_ch("C6", "env", "MAYA_GLM_MULTI_MTP_MAX=0",
                               "--parallel %d 大于 MAYA_GLM_MULTI_MTP_MAX=%d：运行中请求超过 %d 时 MTP 窗口关闭，召回只记账"
                               "不出草稿。8×3080 实测 8 路放开后 65.1 tok/s，低于默认的 76.6 tok/s，只在 GPU 不是瓶颈"
                               "（专家全在显存）时考虑" % (par, mx, mx), True))
    if e == "strata":
        b = _int(c.get("max_running"))
        if b and b > 1:
            notes.append("C7：--batch %d 时多行批与 MTP 草稿同时生效，召回在每行窗口里起作用；--batch 不要超过显存允许的行数" % b)
    if peak and peak >= 32:
        notes.append("C8：峰值并发 %d。高并发时投机本身的收益变小，召回只在业务有重复内容（同题重问、改写、代码编辑、"
                     "RAG 摘抄）时有可见收益；新内容为主的业务要靠投机头本身和自适应草稿数（KAD）" % peak)


def _kad(i, changes, notes):
    e, h, env = i["engine"], i["head"], _env(i)
    alg = h["algorithm"]
    if _off(env, ("YEE_KAD",)):
        return _feat("off", "YEE_KAD=0")
    if alg == "NONE" and e != "tensorfold":
        return _feat("n/a", "没开投机头")
    if e == "sglang":
        if i.get("flavor") != "mia-dsv41":
            notes.append("C9：上游 SGLang 没有按批变草稿步数的机制，KAD 不生效，其余照常")
            return _feat("n/a", "上游 SGLang 没有 adaptive 运行时")
        if alg not in EAGLE_FAMILY:
            return _feat("n/a", "%s 的块长或送验宽度整批固定，KAD 原生跑" % alg)
        if not h["adaptive"]:
            changes.append(_ch("C9", "flag", "--speculative-adaptive",
                               "Mia 镜像的 adaptive 运行时，KAD 每批按 K_b = max k_i 选已捕获步数", True))
            return _feat("needs-change", "需要 --speculative-adaptive")
        return _feat("on", "Mia adaptive 运行时")
    if e == "vllm":
        if alg not in ("DSPARK", "DFLASH", "EAGLE", "EAGLE3", "MTP"):
            return _feat("n/a", "%s 头不接 KAD" % alg)
        if _off(env, ("VLLM_D_RECALL_KAD",)):
            return _feat("off", "VLLM_D_RECALL_KAD=0")
        st = _feat("on", "调度器按请求定宽")
    elif e == "trtllm":
        if _find_key(h["raw"], "draft_len_schedule") is None:
            changes.append(_ch("C10", "config", "--extra_llm_api_options 的 yaml 里给 speculative_config 加 "
                                                "draft_len_schedule（按批大小给草稿长度，格式见 TensorRT-LLM 文档）",
                               "没配时草稿长度固定，Yee 的 KAD 无从调整；配了以后高并发时自动缩短草稿"))
            return _feat("needs-change", "需要 draft_len_schedule")
        st = _feat("on", "走引擎的动态草稿长度路径")
    elif e == "mlx_lm":
        st = _feat("on", "每轮草稿数 = min(KAD 宽度, 启动 K)")
    elif e == "llama.cpp":
        return _feat("needs-change", "需要补丁 llama.cpp-kad（yee patch apply llama.cpp-kad --src <llama.cpp 源码根>）")
    elif e == "tensorfold":
        return _feat("needs-change", "需要补丁 tensorfold-kad")
    else:
        return _feat("n/a", "%s 未接入 KAD" % e)
    k = _int(h.get("num_draft_tokens")) or _int(h.get("num_steps"))
    if k and k != KAD_K:
        notes.append("C11：启动 K=%d，KAD 宽度上限变成 min(%d, %d)；要用满策略把启动 K 设为 %d" % (k, KAD_K, k, KAD_K))
    return st


def _expert_budget(i, notes):
    env = _env(i)
    if _off(env, ("YEE_EXPERT_BUDGET",)):
        return _feat("off", "YEE_EXPERT_BUDGET=0")
    if i.get("flavor") != "mia-dsv41":
        return _feat("n/a", "只在 Mia 镜像的 moe_b12x_next 确定性路径起作用")
    ep = _int(_val(env, "EP_SIZE")) or i["parallel"]["ep"]
    if ep != 1:
        notes.append("K8：专家预算要 EP_SIZE=1 才生效（现在 %d），其余加速照常" % ep)
        return _feat("n/a", "EP_SIZE=%d" % ep)
    if _val(env, "DSV41_MOE_B12X_NEXT") == "1" and _val(env, "DSV41_MOE_B12X_NEXT_DETERMINISTIC") == "1":
        return _feat("on", "moe_b12x_next 确定性路径、EP_SIZE=1")
    return _feat("n/a", "需要 DSV41_MOE_B12X_NEXT=1、DSV41_MOE_B12X_NEXT_DETERMINISTIC=1")


def _tree(i):
    if i["engine"] not in RECALL_AUTO or not _spec_on(i):
        return _feat("n/a", "")
    try:
        from .registry import tree_mode

        mode, why = tree_mode(i["engine"], i["head"]["algorithm"], i["head"].get("topk"))
    except Exception as e:  # noqa: BLE001
        return _feat("n/a", "tree_mode: %r" % (e,))
    return _feat("on" if mode in ("native", "select") else "n/a", "%s: %s" % (mode, why))


def _recall(i, changes, patched):
    e, env = i["engine"], _env(i)
    if _off(env, MASTER):
        return _feat("off", "主开关 YEE / RECALL_ACCEL = 0")
    if e in ("trtllm", "mlx_lm"):
        return _feat("n/a", "这个引擎 Yee 只做 KAD（逐请求自适应草稿数）")
    if e in UNSUPPORTED:
        return _feat("n/a", "引擎不支持")
    need = [c for c in changes if c["rule"][:1] in ("H", "P") and not c["optional"]]
    if e in RECALL_AUTO or e in PATCH_ENGINES or e == "tensorfold":
        if need:
            return _feat("needs-change", "、".join(sorted({c["rule"] for c in need})))
        if e in RECALL_AUTO:
            return _feat("on", "自动挂钩（%s）" % i["head"]["algorithm"])
        return _feat("on", "补丁已在" if patched else "引擎自带")
    return _feat("n/a", "")


def _label(i) -> str:
    w = i["where"]
    s = i["engine"]
    if i.get("flavor") == "mia-dsv41":
        s += "（Mia dsv41）"
    elif i.get("flavor"):
        s += "（%s）" % i["flavor"]
    if w.get("pid"):
        s += " pid %s" % w["pid"]
    s += " @ %s" % w["host"]
    if w.get("container"):
        s += " 容器 %s" % w["container"]
    return s


def _instance(idx, raw) -> dict:
    i = normalize(raw)
    changes, notes = [], list(i["notes"])
    role, e = i["role"], i["engine"]
    out = {"id": "i%d" % idx, "label": _label(i), "engine": e, "role": role, "verdict": "accelerated",
           "features": {}, "changes": changes, "notes": notes, "instance": i}
    if role in ("prefill", "router"):
        out["verdict"] = "not-needed"
        notes.append("K5：PD 分离时召回在 decode 节点起作用，prefill 节点不用装（装了也不影响）" if role == "prefill"
                     else "K6：路由进程本身不用装 Yee")
        out["features"] = {k: _feat("n/a", "") for k, _ in FEATURES}
        return out
    patched = False
    if e in UNSUPPORTED:
        _rules_unsupported(i, changes)
    else:
        if role != "worker":
            _rules_head(i, changes, notes)
        if e in PATCH_ENGINES or e == "tensorfold":
            patched = _rules_patch(i, changes, notes)
        if role != "worker":
            _rules_conc(i, changes, notes)
        else:
            notes.append("多机实例的从节点：启动参数跟随主节点；本节点要装同一版本 Yee（K1）")
    if e in RECALL_AUTO and i["where"].get("container") and i["yee"].get("installed") is not True:
        changes.append(_ch("K2", "install", "镜像里 pip install yee-%s-py3-none-any.whl && python -m yee install-hook；"
                                            "或挂载 -v /opt/yee:/opt/yee -e PYTHONPATH=/opt/yee" % VERSION,
                           "装进容器里的解释器；直接 pip 进运行中的容器，容器重建就没了"))
    if i.get("flavor") == "mia-dsv41" and i["head"]["algorithm"] == "DSPARK" and \
            not any(a.startswith("--speculative-dspark-block-size") for a in i["argv"]) and \
            _val(_env(i), "DSPARK_BLOCK_SIZE") is None:
        notes.append("读不到 --speculative-dspark-block-size，Yee 不补 verify cap 相关默认值")
    out["features"] = {
        "recall": _recall(i, changes, patched),
        "kad": _kad(i, changes, notes) if e not in UNSUPPORTED else _feat("n/a", "引擎不支持"),
        "expert_budget": _expert_budget(i, notes),
        "tree": _tree(i),
        "shared_records": _feat("n/a", "单副本"),
    }
    if e in UNSUPPORTED:
        out["verdict"] = "unsupported"
    elif e in PATCH_ENGINES and not patched:
        out["verdict"] = "needs-patch"
    elif any(c["kind"] in ("flag", "env", "config", "patch", "build") and not c["optional"] for c in changes):
        out["verdict"] = "needs-change"
    return out


# ----------------------------------------------------------------------------------------------- cluster
def _dir_of(i):
    env = _env(i)
    for n in DIR_VARS:
        v = _val(env, n)
        if v:
            return v
    a = i["argv"]
    for k, x in enumerate(a):
        if x == "--recall-dir" and k + 1 < len(a):
            return a[k + 1]
        if x.startswith("--recall-dir="):
            return x.split("=", 1)[1]
    return None


def _cluster(items) -> dict:
    groups, order = {}, []
    routers = [it for it in items if it["role"] == "router"]
    for it in items:
        if it["role"] == "router":
            continue
        key = _model_label(it["instance"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(it)
    cl = {"groups": [], "changes": [], "notes": []}
    seen = set()

    def add(c):
        if (c["rule"], c["set"]) not in seen:
            seen.add((c["rule"], c["set"]))
            cl["changes"].append(c)

    versions = {}
    for key in order:
        mem = groups[key]
        reps = [it for it in mem if it["role"] not in ("prefill", "worker")]
        hosts = sorted({it["instance"]["where"]["host"] for it in mem})
        rep_hosts = sorted({it["instance"]["where"]["host"] for it in reps})
        sharing = "single" if len(reps) <= 1 else ("same-host" if len(rep_hosts) <= 1 else "cross-host")
        cl["groups"].append({"model": key, "instances": [it["id"] for it in mem], "hosts": hosts,
                             "replicas": len(reps), "sharing": sharing})
        eng = reps[0]["engine"] if reps else (mem[0]["engine"] if mem else "")
        if any(it["instance"]["parallel"]["nnodes"] > 1 or it["role"] == "worker" for it in mem):
            missing = any(it["instance"]["yee"].get("installed") is False for it in mem)
            add(_ch("K1", "install", "bash install.sh --hosts %s" % (",".join(hosts) if len(hosts) > 1
                                                                       else "<所有节点，逗号分隔>"),
                    "%s多机实例的每个节点都要装同一版本 Yee；只装一台时 rank 检查会关掉召回与专家预算（日志 rank check off）"
                    % ("" if missing else "确认："), not missing))
        for it in mem:
            v = it["instance"]["yee"].get("version")
            if v:
                versions.setdefault(v, []).append(it["id"])
        share_engines = RECALL_AUTO + ("strata", "maya")
        if sharing == "same-host" and eng in share_engines:
            dirs = {_dir_of(it["instance"]) for it in reps}
            cont = any(it["instance"]["where"].get("container") for it in reps)
            mem_backend = any((_val(_env(it["instance"]), n) or "").lower() in ("memory", "cuda", "gpu")
                              for it in reps for n in BACKEND_VARS)
            if eng == "strata":
                need = len(dirs) > 1 or None in dirs
                set_ = "每个副本加 --recall-dir <同一目录>"
            else:
                need = len(dirs) > 1 or (cont and None in dirs) or mem_backend or (eng == "maya" and None in dirs)
                set_ = "所有副本设 RECALL_ACCEL_DIR=/var/lib/yee-recall%s%s" % (
                    "、RECALL_ACCEL_BACKEND=disk" if mem_backend else "",
                    "，容器都挂 -v /var/lib/yee-recall:/var/lib/yee-recall" if cont else "")
            if need:
                add(_ch("K3", "config", set_, "同机副本用同一个磁盘记录目录才共享记录（每 2 秒同步）"))
            st = _feat("needs-change", "K3") if need else _feat("on", "同机共享（同一记录目录）")
        elif sharing == "cross-host" and eng in share_engines:
            set_ = "sglang-router 用 --policy cache_aware" if eng == "sglang" else \
                "路由按会话/前缀亲和，例如 nginx：hash $http_x_session_id consistent;"
            add(_ch("K4", "route", set_, "记录只在本机共享（flock，不能放 NFS）；相关请求落到同一台机器才能复用记录", True))
            st = _feat("off", "跨机不共享，靠亲和路由（K4）")
        else:
            st = _feat("n/a", SHARING_ZH.get(sharing, sharing))
        for it in reps:
            if it["features"].get("shared_records", {}).get("state") in (None, "n/a"):
                it["features"]["shared_records"] = st
    for r in routers:
        pol = str(_d(r["instance"].get("router")).get("policy") or "").lower()
        if any(g["sharing"] == "cross-host" for g in cl["groups"]) and pol in ("", "round_robin", "random",
                                                                                 "power_of_two"):
            add(_ch("K6", "route", "--policy cache_aware", "路由策略 %s 会把相关请求分到不同机器，跨机副本复用不到记录"
                                                           % (pol or "默认"), True))
    if len(versions) > 1:
        add(_ch("K7", "install", "统一升级到 Yee %s（bash install.sh --hosts <全部节点>）" % VERSION,
                "实例装的 Yee 版本不一致：%s" % "、".join("%s=%s" % (v, ",".join(ids))
                                                     for v, ids in sorted(versions.items()))))
    return cl


# ----------------------------------------------------------------------------------------------- public
def analyze(instances, workload=None) -> dict:
    """Plan of a deployment (see module doc). Never raises."""
    try:
        items = [_instance(n + 1, raw) for n, raw in enumerate(instances or [])]
        cl = _cluster(items)
    except Exception as e:  # noqa: BLE001
        return {"version": VERSION, "instances": [], "cluster": {"groups": [], "changes": [], "notes": [
            "plan failed: %r" % (e,)]}, "summary": _summary([]), "exit": EXIT_UNSUPPORTED}
    share = _d(workload).get("recall_share")
    if isinstance(share, (int, float)) and share < 0.01:
        cl["notes"].append("E2：召回只占输出约 %.2f%%，业务里可复用的重复内容少，速度主要来自投机头" % (share * 100))
    plan = {"version": VERSION, "instances": items, "cluster": cl, "summary": _summary(items)}
    plan["exit"] = _exit(items, cl)
    return plan


def _summary(items) -> dict:
    s = {"accelerated": 0, "needs_change": 0, "needs_patch": 0, "unsupported": 0, "not_needed": 0}
    for it in items:
        s[it["verdict"].replace("-", "_")] += 1
    return s


def _exit(items, cl) -> int:
    if not items:
        return EXIT_UNSUPPORTED
    vs = [it["verdict"] for it in items]
    if all(v == "unsupported" for v in vs):
        return EXIT_UNSUPPORTED
    if any(v in ("needs-change", "needs-patch", "unsupported") for v in vs):
        return EXIT_ACTION
    if any(not c["optional"] for c in cl["changes"]):
        return EXIT_ACTION
    return EXIT_OK


def _change_line(c, indent="     ") -> str:
    tag = KIND_ZH.get(c["kind"], c["kind"]) + ("·可选·先测" if c.get("optional") else "")
    return "%s- [%s] %s   原因：%s（%s）" % (indent, tag, c["set"], c["why"], c["rule"])


def _head_line(i) -> str:
    p, h = i["parallel"], i["head"]
    bits = [_label(i)]
    if i.get("model"):
        bits.append("模型 %s" % i["model"])
    par = "TP%d" % p["tp"]
    for k, n in (("dp", "DP"), ("pp", "PP"), ("ep", "EP")):
        if p[k] > 1:
            par += " %s%d" % (n, p[k])
    if p["nnodes"] > 1:
        par += " 节点%d" % p["nnodes"]
    bits.append(par)
    bits.append("投机头 %s" % h["algorithm"] + (" topk=%s" % h["topk"] if h.get("topk") else ""))
    peak = _peak(i)
    if peak:
        bits.append("并发上限 %d" % peak)
    return " · ".join(bits)


def render(plan) -> str:
    """The Chinese report of ``yee doctor``."""
    try:
        return _render(plan)
    except Exception as e:  # noqa: BLE001
        return "Yee %s 部署诊断：报告生成失败 %r" % (VERSION, e)


def _render(plan) -> str:
    items = plan.get("instances") or []
    lines = []
    if not items:
        lines.append("Yee %s 部署诊断：没有找到推理服务。用 --cmd \"<启动命令>\"、--url http://主机:端口、--hosts、"
                     "--config 指定后再试。" % VERSION)
        lines.append("合计：可加速 0 · 要改参数 0 · 要打补丁 0 · 不支持 0 · 无需处理 0。退出码 %d" % plan.get("exit", 3))
        return "\n".join(lines)
    src = {}
    for it in items:
        s = _d(_d(it.get("instance")).get("where")).get("source", "cmd")
        src[s] = src.get(s, 0) + 1
    lines.append("Yee %s 部署诊断：找到 %d 个实例（%s）" % (VERSION, len(items), "、".join(
        "%s %d" % (SOURCE_ZH.get(k, k), v) for k, v in src.items())))
    for it in items:
        i = normalize(it.get("instance") or {})
        lines.append("[%s] %s" % (it["id"], _head_line(i)))
        lines.append("     结论：%s" % VERDICT_ZH.get(it["verdict"], it["verdict"]))
        fl = []
        for k, zh in FEATURES:
            f = _d(it["features"].get(k))
            st = f.get("state", "n/a")
            if k == "tree" and st == "on":
                fl.append("%s %s" % (zh, str(f.get("why", "")).split(":", 1)[0]))
            elif k == "shared_records" and st == "on":
                fl.append("%s 同机共享" % zh)
            else:
                fl.append("%s %s" % (zh, STATE_ZH.get(st, st)))
        lines.append("     " + " · ".join(fl))
        for c in it["changes"]:
            lines.append(_change_line(c))
        for n in it["notes"]:
            lines.append("     说明：%s" % n)
    cl = plan.get("cluster") or {}
    if cl.get("groups") or cl.get("changes") or cl.get("notes"):
        lines.append("集群")
        for g in cl.get("groups", []):
            if g["sharing"] == "single":
                desc = "1 个副本"
            elif g["sharing"] == "same-host":
                desc = "%d 个副本，同一台机器 → 可共享记录" % g["replicas"]
            else:
                desc = "%d 个副本，跨 %d 台机器 → 记录不跨机共享" % (g["replicas"], len(g["hosts"]))
            lines.append("  模型 %s：%s（%s）" % (g["model"], desc, ",".join(g["instances"])))
        for c in cl.get("changes", []):
            lines.append(_change_line(c))
        for n in cl.get("notes", []):
            lines.append("     说明：%s" % n)
    s = plan.get("summary") or _summary(items)
    lines.append("合计：可加速 %d · 要改参数 %d · 要打补丁 %d · 不支持 %d · 无需处理 %d。退出码 %d" % (
        s["accelerated"], s["needs_change"], s["needs_patch"], s["unsupported"], s["not_needed"],
        plan.get("exit", 0)))
    return "\n".join(lines)


def _notice_off(environ) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get("YEE_NOTICE", "1")).strip().lower() in _FALSE


def startup_lines(argv=None, environ=None) -> list:
    """``[yee] 4.3.0 plan:`` lines for the server process whose launch command is ``argv`` (default sys.argv)."""
    try:
        if _notice_off(environ):
            return []
        from .launch import parse_launch

        env = dict(os.environ if environ is None else environ)
        inst = parse_launch(list(sys.argv if argv is None else argv), env)
        if not inst:
            return []
        it = analyze([inst])["instances"][0]
        i = it["instance"]
        lines = ["[yee] %s plan: %s head=%s -> %s" % (VERSION, i["engine"], i["head"]["algorithm"],
                                                       VERDICT_ZH.get(it["verdict"], it["verdict"]))]
        for c in it["changes"][:6]:
            lines.append("[yee] 要改: %s  (%s%s %s)" % (c["set"], c["rule"], " 可选·先测" if c["optional"] else "",
                                                        c["why"]))
        if it["changes"]:
            lines.append("[yee] 全部建议: python -m yee doctor")
        return lines
    except Exception:  # noqa: BLE001
        return []


def _advice(instance, limit=3) -> list:
    if not instance:
        return []
    try:
        it = analyze([instance])["instances"][0]
    except Exception:  # noqa: BLE001
        return []
    return [c for c in it["changes"] if c["rule"][:1] in ("H", "C")][:limit]


def effect_lines(counts, instance, environ=None) -> list:
    """``[yee] effect:`` lines from the recall counters (steps, rows, used, landed, errors, ...)."""
    try:
        if _notice_off(environ):
            return []
        c = _d(counts)
        steps, rows = _int(c.get("steps")) or 0, _int(c.get("rows")) or 0
        landed, used, errors = _int(c.get("landed")) or 0, _int(c.get("used")) or 0, _int(c.get("errors")) or 0
        out = []
        if errors > 0:
            out.append("[yee] effect: E3 召回出错 %d 次，看前面以 [recall] 开头的报错行" % errors)
        adv = _advice(instance)
        if steps > 0 and rows == 0:
            out.append("[yee] effect: E1 钩子已装，但投机路径还没有跑到（steps=%d rows=0）。常见原因：没开投机头、"
                       "批超过上限、投机按批大小关闭" % steps)
            out.extend("[yee] effect:   改 %s  (%s %s)" % (a["set"], a["rule"], a["why"]) for a in adv)
        elif rows > 0 and landed / float(rows) < 0.02:
            out.append("[yee] effect: E2 召回贡献约 <1%%（估算：landed=%d rows=%d，每行 %.3f token）。业务里可复用的"
                       "重复内容少，召回带不来可见提速，速度主要来自投机头" % (landed, rows, landed / float(rows)))
            if adv:
                out.extend("[yee] effect:   改 %s  (%s %s)" % (a["set"], a["rule"], a["why"]) for a in adv)
            else:
                out.append("[yee] effect:   投机头配置已可用；这类业务召回收益有限")
        elif rows > 0 and not errors:
            out.append("[yee] effect: 召回在工作 used=%d landed=%d rows=%d，每行约 %.2f token" % (
                used, landed, rows, landed / float(rows)))
        return out
    except Exception:  # noqa: BLE001
        return []


def print_startup(argv=None, environ=None) -> int:
    """Print ``startup_lines()`` once per process tree (the launcher prints, its children inherit SHOWN)."""
    try:
        env = os.environ if environ is None else environ
        if env.get(SHOWN):
            return 0
        lines = startup_lines(argv, env)
        if lines:
            env[SHOWN] = "1"
            for line in lines:
                print(line, flush=True)
        return len(lines)
    except Exception:  # noqa: BLE001
        return 0


def effect_tick(state, counts, environ=None, now=None, argv=None) -> list:
    """Called after every ``[recall] steps=`` line: prints ``effect_lines()`` every YEE_EFFECT_S seconds (600),
    the first time one interval after the first call. ``state`` is the adapter's own dict."""
    try:
        env = os.environ if environ is None else environ
        now = time.monotonic() if now is None else now
        try:
            every = float(env.get("YEE_EFFECT_S") or 600)
        except ValueError:
            every = 600.0
        t = state.get("effect_t")
        if t is None:
            state["effect_t"] = now
            return []
        if now - t < every:
            return []
        state["effect_t"] = now
        if "effect_inst" not in state:
            try:
                from .launch import parse_launch

                state["effect_inst"] = parse_launch(list(sys.argv if argv is None else argv), dict(env))
            except Exception:  # noqa: BLE001
                state["effect_inst"] = None
        lines = effect_lines(counts, state.get("effect_inst"), env)
        for line in lines:
            print(line, flush=True)
        return lines
    except Exception:  # noqa: BLE001
        return []


__all__ = ["VERSION", "RULES", "analyze", "render", "startup_lines", "effect_lines", "normalize", "print_startup",
           "effect_tick", "EXIT_OK", "EXIT_ACTION", "EXIT_UNSUPPORTED"]
