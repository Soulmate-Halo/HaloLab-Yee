"""Plain-assert tests of the Yee 4.3 deployment plan (``yee.integrations.plan``): rules H / P / C / K / E, verdicts,
exit codes, cluster grouping, the Chinese report, startup and effect lines.

Run: python tests/test_plan.py (also runs under pytest and in ``yee selftest``).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "yee")):
    sys.path.insert(0, ROOT)

from yee.integrations import plan  # noqa: E402

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


def _inst(engine="sglang", alg="NONE", **kw):
    i = {"engine": engine, "flavor": kw.pop("flavor", None), "role": kw.pop("role", "server"), "version": None,
         "model": kw.pop("model", "/models/M"), "port": kw.pop("port", 30000),
         "head": {"algorithm": alg, "draft_model": None, "num_steps": kw.pop("num_steps", None),
                  "topk": kw.pop("topk", None), "num_draft_tokens": kw.pop("num_draft_tokens", None),
                  "adaptive": kw.pop("adaptive", False), "disable_by_batch_size": kw.pop("dbbs", None),
                  "raw": kw.pop("raw", {})},
         "parallel": {"tp": kw.pop("tp", 1), "pp": 1, "dp": 1, "ep": kw.pop("ep", 1), "nnodes": kw.pop("nnodes", 1),
                      "node_rank": None, "dp_attention": False, "backend": None},
         "concurrency": {"max_running": kw.pop("max_running", None), "cuda_graph_max_bs": kw.pop("cgmb", None),
                         "cuda_graph": kw.pop("cuda_graph", None)},
         "router": kw.pop("router", None),
         "yee": {"env": {}, "installed": kw.pop("installed", None), "version": kw.pop("yee_version", None)},
         "model_info": kw.pop("model_info", {}), "load": kw.pop("load", None),
         "where": {"host": kw.pop("host", "h1"), "pid": kw.pop("pid", 100), "container": kw.pop("container", None),
                   "source": kw.pop("source", "proc")},
         "argv": kw.pop("argv", []), "env": kw.pop("env", {}), "notes": []}
    assert not kw, kw
    return i


def _one(inst):
    p = plan.analyze([inst])
    return p, p["instances"][0]


def _rules(it):
    return [c["rule"] for c in it["changes"]]


def _change(it, rule):
    hits = [c for c in it["changes"] if c["rule"] == rule]
    assert hits, (rule, it["changes"])
    return hits[0]


def _all_text(p):
    return json.dumps(p, ensure_ascii=False) + plan.render(p)


@test
def test_empty_exit_3():
    p = plan.analyze([])
    assert p["exit"] == plan.EXIT_UNSUPPORTED == 3
    assert "合计：" in plan.render(p) and "退出码 3" in plan.render(p)


@test
def test_h0_sglang_mtp_known():
    p, it = _one(_inst(model_info={"mtp_layers": 1}))
    c = _change(it, "H0")
    assert "--speculative-algorithm EAGLE" in c["set"] and "--speculative-eagle-topk 1" in c["set"]
    assert "NGRAM" not in " ".join(x["set"] for x in it["changes"])
    assert it["verdict"] == "needs-change" and p["exit"] == 1
    assert it["features"]["recall"]["state"] == "needs-change"


@test
def test_h0_sglang_no_mtp_and_unknown():
    _, it = _one(_inst(model_info={"mtp_layers": 0}))
    assert [c["set"] for c in it["changes"] if c["rule"] == "H0"] == ["--speculative-algorithm NGRAM"]
    _, it = _one(_inst())
    sets = [c["set"] for c in it["changes"] if c["rule"] == "H0"]
    assert len(sets) == 2 and "NGRAM" in sets[1]


@test
def test_h0_vllm_trtllm_mlx():
    _, it = _one(_inst("vllm", model_info={"mtp_layers": 2}))
    assert '"method":"mtp"' in _change(it, "H0")["set"]
    _, it = _one(_inst("vllm", model_info={"mtp_layers": 0}))
    assert "eagle3" in _change(it, "H0")["set"]
    _, it = _one(_inst("trtllm"))
    assert _change(it, "H0")["kind"] == "config"
    _, it = _one(_inst("mlx_lm"))
    assert "--draft-model" in _change(it, "H0")["set"]


@test
def test_h1_topk():
    _, it = _one(_inst(alg="EAGLE", topk=4, num_steps=3))
    assert _change(it, "H1")["set"] == "--speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
    _, it = _one(_inst(alg="EAGLE", topk=1, num_steps=3, cgmb=64, max_running=32))
    assert "H1" not in _rules(it) and it["verdict"] == "accelerated"


@test
def test_h2_h5():
    _, it = _one(_inst(alg="FROZEN_KV_MTP"))
    assert _change(it, "H2")["set"] == "--speculative-algorithm EAGLE"
    _, it = _one(_inst(alg="WEIRD_HEAD"))
    assert "H5" in _rules(it)


@test
def test_h3_h4_vllm():
    _, it = _one(_inst("vllm", alg="NGRAM"))
    assert "H3" in _rules(it)
    _, it = _one(_inst("vllm", alg="MTP", num_draft_tokens=3))
    c = _change(it, "H4")
    assert c["set"] == "VLLM_USE_V2_MODEL_RUNNER=1" and c["kind"] == "env"
    _, it = _one(_inst("vllm", alg="MTP", env={"VLLM_USE_V2_MODEL_RUNNER": "1"}))
    assert "H4" not in _rules(it) and it["verdict"] == "accelerated"
    assert it["features"]["kad"]["state"] == "on"


@test
def test_h6_strata_p2():
    _, it = _one(_inst("strata", argv=["/opt/strata", "--batch", "2"]))
    assert _change(it, "H6")["set"] == "--spec 4"
    assert _change(it, "P2")["set"].startswith("yee patch apply strata-recall")
    assert it["verdict"] == "needs-patch"


@test
def test_h7_maya_and_p3():
    _, it = _one(_inst("maya", argv=["/opt/maya/build/maya", "--parallel", "8"], max_running=8))
    assert _change(it, "H7")["set"] == "MAYA_GLM_MULTI_MTP=3"
    assert "yee patch apply maya-recall" in _change(it, "P3")["set"]
    assert any(c["kind"] == "build" and "--target maya" in c["set"] for c in it["changes"])
    assert it["verdict"] == "needs-patch"


@test
def test_maya_patched_default_8way_c6_optional():
    inst = _inst("maya", alg="MTP", num_draft_tokens=3, max_running=8,
                 argv=["/opt/maya/build/maya-yee", "--glm-pack", "/p", "--parallel", "8"],
                 env={"MAYA_GLM_MULTI_MTP": "3", "RECALL_ACCEL": "1", "RECALL_ACCEL_DIR": "/r"})
    p, it = _one(inst)
    c = _change(it, "C6")
    assert c["optional"] and c["set"] == "MAYA_GLM_MULTI_MTP_MAX=0" and "65.1" in c["why"]
    assert it["verdict"] == "accelerated" and p["exit"] == 0
    assert "P3" not in _rules(it)
    inst["env"]["MAYA_GLM_MULTI_MTP_MAX"] = "0"
    _, it = _one(inst)
    assert "C6" not in _rules(it)


@test
def test_h8_llamacpp_p1():
    _, it = _one(_inst("llama.cpp", argv=["llama-server", "-m", "x.gguf"]))
    assert len([c for c in it["changes"] if c["rule"] == "H8"]) == 2
    assert "llama.cpp-recall" in _change(it, "P1")["set"]
    assert any(c["set"] == "LLAMA_ARG_RECALL=1" for c in it["changes"]), it["changes"]
    assert any("llama.cpp-kad" in n for n in it["notes"])


@test
def test_h9_unsupported_exit3():
    for eng in ("ollama", "tgi", "ktransformers", "lmdeploy", "unknown"):
        p, it = _one(_inst(eng))
        assert it["verdict"] == "unsupported", eng
        assert "H9" in _rules(it)
        assert p["exit"] == 3
    p = plan.analyze([_inst("ollama"), _inst("vllm", alg="MTP", env={"VLLM_USE_V2_MODEL_RUNNER": "1"})])
    assert p["exit"] == 1


@test
def test_p4_tensorfold_mlxserve_cuteafd():
    _, it = _one(_inst("tensorfold", argv=["tensorfold", "serve"]))
    assert _change(it, "P4")["set"] == "--recall"
    _, it = _one(_inst("tensorfold", alg="RECALL", argv=["tensorfold", "serve", "--recall"]))
    assert "P4" not in _rules(it)
    for eng in ("mlx-serve", "cuteafd"):
        _, it = _one(_inst(eng, alg="DSPARK", argv=[eng]))
        assert "P4" in _rules(it) and it["verdict"] == "needs-patch"


@test
def test_c1_c2_c3_c4_c8_image_scenario():
    a = _inst(alg="DSPARK", flavor="mia-dsv41", max_running=64, cgmb=32, host="a",
              argv=["python", "-m", "sglang.launch_server", "--speculative-dspark-block-size", "7"])
    b = _inst(alg="DSPARK", flavor="mia-dsv41", max_running=64, cgmb=32, host="b",
              argv=["python", "-m", "sglang.launch_server", "--speculative-dspark-block-size", "7"])
    p = plan.analyze([a, b])
    it = p["instances"][0]
    assert _change(it, "C1")["set"] == "--cuda-graph-max-bs 64"
    assert not any(n.startswith("C4") for n in it["notes"])
    assert any(n.startswith("C8") for n in it["notes"])
    assert p["cluster"]["groups"][0]["sharing"] == "cross-host"
    k4 = [c for c in p["cluster"]["changes"] if c["rule"] == "K4"]
    assert k4 and k4[0]["set"] == "sglang-router 用 --policy cache_aware" and k4[0]["optional"]
    assert it["features"]["shared_records"]["state"] == "off"
    _, it = _one(_inst(alg="EAGLE", topk=1, max_running=128, cgmb=256))
    assert any(n.startswith("C4") for n in it["notes"]) and "C1" not in _rules(it)
    _, it = _one(_inst(alg="EAGLE", topk=1))
    assert any(n.startswith("C2") for n in it["notes"])
    _, it = _one(_inst(alg="EAGLE", topk=1, cuda_graph=False))
    assert _change(it, "C3")["set"] == "去掉 --disable-cuda-graph"
    _, it = _one(_inst("vllm", alg="MTP", cuda_graph=False, env={"VLLM_USE_V2_MODEL_RUNNER": "1"}))
    assert _change(it, "C3")["set"] == "去掉 --enforce-eager"


@test
def test_c5_vllm_disable_by_batch_size():
    raw = {"method": "mtp", "num_speculative_tokens": 3, "disable_by_batch_size": 8}
    _, it = _one(_inst("vllm", alg="MTP", dbbs=8, raw=raw, max_running=64, env={"VLLM_USE_V2_MODEL_RUNNER": "1"}))
    c = _change(it, "C5")
    assert c["optional"] and '"disable_by_batch_size":64' in c["set"]
    assert it["verdict"] == "accelerated"
    nested = {"speculative_config": raw}  # the shape launch.parse_launch writes
    _, it = _one(_inst("vllm", alg="MTP", dbbs=8, raw=nested, max_running=64, env={"VLLM_USE_V2_MODEL_RUNNER": "1"}))
    s = _change(it, "C5")["set"]
    assert s.startswith("--speculative-config '{\"method\":\"mtp\"") and '"disable_by_batch_size":64' in s, s


@test
def test_c7_strata_batch():
    _, it = _one(_inst("strata", alg="MTP", num_draft_tokens=4, max_running=2,
                       argv=["/opt/strata", "--spec", "4", "--batch", "2", "--recall"], env={"RECALL_ACCEL": "1"}))
    assert any(n.startswith("C7") for n in it["notes"]) and it["verdict"] == "accelerated"


@test
def test_c9_c10_c11_kad():
    _, it = _one(_inst(alg="EAGLE", topk=1))
    assert any(n.startswith("C9") for n in it["notes"]) and it["features"]["kad"]["state"] == "n/a"
    _, it = _one(_inst(alg="EAGLE", topk=1, flavor="mia-dsv41"))
    assert _change(it, "C9")["set"] == "--speculative-adaptive" and _change(it, "C9")["optional"]
    _, it = _one(_inst(alg="EAGLE", topk=1, flavor="mia-dsv41", adaptive=True))
    assert it["features"]["kad"]["state"] == "on"
    _, it = _one(_inst("trtllm", alg="MTP"))
    assert _change(it, "C10")["kind"] == "config" and it["verdict"] == "needs-change"
    _, it = _one(_inst("trtllm", alg="MTP", raw={"speculative_config": {"draft_len_schedule": {"1": 4}}}))
    assert "C10" not in _rules(it) and it["features"]["kad"]["state"] == "on"
    _, it = _one(_inst("vllm", alg="MTP", num_draft_tokens=3, env={"VLLM_USE_V2_MODEL_RUNNER": "1"}))
    assert any(n.startswith("C11") for n in it["notes"])


@test
def test_k1_multinode_and_worker():
    head = _inst(alg="EAGLE", topk=1, nnodes=2, host="n1", cgmb=64)
    worker = _inst(alg="EAGLE", topk=1, nnodes=2, host="n2", role="worker", installed=False)
    p = plan.analyze([head, worker])
    k1 = [c for c in p["cluster"]["changes"] if c["rule"] == "K1"]
    assert k1 and k1[0]["set"] == "bash install.sh --hosts n1,n2" and not k1[0]["optional"]
    assert p["exit"] == 1
    w = p["instances"][1]
    assert not [c for c in w["changes"] if c["rule"][:1] in ("H", "C")]
    assert p["cluster"]["groups"][0]["replicas"] == 1


@test
def test_k2_container():
    _, it = _one(_inst("vllm", alg="MTP", container="vllm1", source="docker", env={"VLLM_USE_V2_MODEL_RUNNER": "1"}))
    c = _change(it, "K2")
    assert c["kind"] == "install" and "pip install yee-4.3.0" in c["set"] and "PYTHONPATH=/opt/yee" in c["set"]
    assert it["verdict"] == "accelerated"


@test
def test_k3_same_host():
    a = _inst(alg="EAGLE", topk=1, port=1, container="c1", cgmb=64)
    b = _inst(alg="EAGLE", topk=1, port=2, container="c2", cgmb=64)
    p = plan.analyze([a, b])
    assert p["cluster"]["groups"][0]["sharing"] == "same-host"
    k3 = [c for c in p["cluster"]["changes"] if c["rule"] == "K3"]
    assert k3 and "RECALL_ACCEL_DIR=/var/lib/yee-recall" in k3[0]["set"] and "-v /var/lib/yee-recall" in k3[0]["set"]
    assert p["exit"] == 1
    a = _inst(alg="EAGLE", topk=1, port=1, cgmb=64)
    b = _inst(alg="EAGLE", topk=1, port=2, cgmb=64)
    p = plan.analyze([a, b])
    assert not p["cluster"]["changes"] and p["exit"] == 0
    assert p["instances"][0]["features"]["shared_records"]["state"] == "on"


@test
def test_k5_k6_k7():
    p, it = _one(_inst(alg="EAGLE", role="prefill"))
    assert it["verdict"] == "not-needed" and any(n.startswith("K5") for n in it["notes"])
    r = _inst("sglang", role="router", router={"policy": "round_robin", "workers": []})
    a = _inst(alg="EAGLE", topk=1, host="a", cgmb=64)
    b = _inst(alg="EAGLE", topk=1, host="b", cgmb=64)
    p = plan.analyze([r, a, b])
    rules = [c["rule"] for c in p["cluster"]["changes"]]
    assert "K6" in rules and "K4" in rules and p["instances"][0]["verdict"] == "not-needed"
    a = _inst(alg="EAGLE", topk=1, host="a", cgmb=64, yee_version="4.2.0")
    b = _inst(alg="EAGLE", topk=1, host="b", cgmb=64, yee_version="4.3.0")
    p = plan.analyze([a, b])
    k7 = [c for c in p["cluster"]["changes"] if c["rule"] == "K7"]
    assert k7 and not k7[0]["optional"] and p["exit"] == 1


@test
def test_k8_expert_budget():
    env = {"DSV41_MOE_B12X_NEXT": "1", "DSV41_MOE_B12X_NEXT_DETERMINISTIC": "1", "EP_SIZE": "1"}
    _, it = _one(_inst(alg="DSPARK", flavor="mia-dsv41", cgmb=64, env=env,
                       argv=["python", "-m", "sglang.launch_server", "--speculative-dspark-block-size", "7"]))
    assert it["verdict"] == "accelerated" and it["features"]["expert_budget"]["state"] == "on"
    assert it["features"]["recall"]["state"] == "on"
    env["EP_SIZE"] = "2"
    _, it = _one(_inst(alg="DSPARK", flavor="mia-dsv41", cgmb=64, env=env))
    assert it["features"]["expert_budget"]["state"] == "n/a" and any(n.startswith("K8") for n in it["notes"])
    _, it = _one(_inst("vllm", alg="MTP"))
    assert it["features"]["expert_budget"]["state"] == "n/a"


@test
def test_master_off():
    _, it = _one(_inst(alg="EAGLE", topk=1, env={"YEE": "0"}, cgmb=64))
    assert it["features"]["recall"]["state"] == "off"


@test
def test_render_and_words():
    p = plan.analyze([_inst(model_info={"mtp_layers": 1}), _inst("ollama", host="b"),
                      _inst("maya", argv=["maya"], max_running=8)])
    txt = plan.render(p)
    assert txt.startswith("Yee 4.3.0 部署诊断：找到 3 个实例")
    assert "结论：要改参数才能加速" in txt and "[参数]" in txt and "（H0）" in txt
    assert "合计：" in txt and "退出码 1" in txt and "集群" in txt
    for bad in ("有损", "无损", "质量"):
        assert bad not in _all_text(p), bad
    for rid, title in plan.RULES.items():
        for bad in ("有损", "无损", "质量"):
            assert bad not in title


@test
def test_plan_json_roundtrip_has_instance():
    p = plan.analyze([_inst("vllm", alg="MTP")])
    q = json.loads(json.dumps(p, ensure_ascii=False))
    assert q["instances"][0]["instance"]["engine"] == "vllm"


@test
def test_startup_lines():
    off = plan.startup_lines(["python", "-m", "sglang.launch_server"], {"YEE_NOTICE": "0"})
    assert off == []
    if importlib.util.find_spec("yee.integrations.launch") is None:
        return "SKIP launch.py not present"
    lines = plan.startup_lines(["python", "-m", "sglang.launch_server", "--model-path", "/m",
                                "--speculative-algorithm", "EAGLE", "--speculative-eagle-topk", "4",
                                "--speculative-num-steps", "3"], {})
    assert lines and lines[0].startswith("[yee] 4.3.0 plan: sglang head=EAGLE ->"), lines
    assert any(l.startswith("[yee] 要改: --speculative-eagle-topk 1") for l in lines), lines
    assert lines[-1] == "[yee] 全部建议: python -m yee doctor"
    assert plan.startup_lines(["bash", "-c", "echo hi"], {}) == []


@test
def test_effect_lines():
    inst = _inst(alg="EAGLE", topk=4, num_steps=3)
    assert plan.effect_lines({"steps": 10, "rows": 0}, inst, {"YEE_NOTICE": "0"}) == []
    e1 = plan.effect_lines({"steps": 10, "rows": 0}, inst, {})
    assert e1[0].startswith("[yee] effect: E1") and any("H1" in l for l in e1)
    e2 = plan.effect_lines({"steps": 1000, "rows": 50000, "landed": 100}, _inst(alg="EAGLE", topk=1, cgmb=64), {})
    assert e2[0].startswith("[yee] effect: E2") and "投机头配置已可用" in e2[-1]
    e3 = plan.effect_lines({"steps": 5, "rows": 10, "landed": 50, "errors": 2}, None, {})
    assert e3[0].startswith("[yee] effect: E3")
    ok = plan.effect_lines({"steps": 5, "rows": 100, "landed": 300, "used": 40}, None, {})
    assert ok == ["[yee] effect: 召回在工作 used=40 landed=300 rows=100，每行约 3.00 token"]
    for line in e1 + e2 + e3:
        assert "有损" not in line and "无损" not in line


@test
def test_bad_input_never_raises():
    p = plan.analyze([None, {"engine": "vllm", "head": "x", "parallel": None}, 5])
    assert len(p["instances"]) == 3
    assert isinstance(plan.render(p), str)
    assert plan.effect_lines(None, None, {}) == []


def main():
    ok = fail = skip = 0
    for fn in RESULTS:
        try:
            res = fn()
            if isinstance(res, str) and res.startswith("SKIP"):
                skip += 1
                print("SKIP %s: %s" % (fn.__name__, res))
            else:
                ok += 1
                print("PASS %s" % fn.__name__)
        except Exception:
            fail += 1
            print("FAIL %s" % fn.__name__)
            traceback.print_exc()
    if fail:
        print("FAILED %d / %d (skip %d)" % (fail, ok + fail, skip))
        sys.exit(1)
    print("ALL PASS n=%d skip=%d" % (ok, skip))


if __name__ == "__main__":
    main()
