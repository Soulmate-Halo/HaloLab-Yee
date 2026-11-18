"""Plain-assert tests of ``yee.integrations.launch`` (Yee 4.3.0 doctor: launch command → instance).

Run: python tests/test_launch.py (also runs under pytest and in ``yee selftest``).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "yee")):
    sys.path.insert(0, ROOT)

from yee.integrations import launch  # noqa: E402

RESULTS = []

MAYA_JSON = {
    "exe": "/opt/maya/build/maya-yee",
    "args": ["--glm-pack", "/opt/maya/pack", "--max-context", "131072", "--parallel", "8"],
    "gpu": [0, 1, 2, 3, 4, 5, 6, 7],
    "port": 8080,
    "model_name": "glm-5.3-flash",
    "installer": {"draft": "mtp"},
    "env": {"MAYA_GLM_MULTI_MTP": "3", "RECALL_ACCEL": "1", "RECALL_ACCEL_DIR": "/opt/maya/recall",
            "MAYA_API_KEY": "x"},
}


def test(fn):
    RESULTS.append(fn)
    return fn


def _inst(cmd, env=None):
    inst = launch.parse_launch(cmd, env)
    assert inst is not None, "no instance for %r" % (cmd,)
    return inst


# --------------------------------------------------------------------------- 引擎识别
@test
def test_engines_tuple_has_15_names():
    assert len(launch.ENGINES) == 15, launch.ENGINES
    for name in ("sglang", "vllm", "llama.cpp", "trtllm", "lmdeploy", "mlx_lm", "mlx-serve", "tgi", "ollama",
                 "tensorfold", "strata", "maya", "cuteafd", "ktransformers", "unknown"):
        assert name in launch.ENGINES, name


@test
def test_instance_has_all_spec_fields():
    inst = _inst("python -m sglang.launch_server --model-path /models/X")
    for key in ("engine", "flavor", "role", "version", "model", "port", "head", "parallel", "concurrency", "router",
                "yee", "model_info", "load", "where", "argv", "env", "notes"):
        assert key in inst, key
    for key in ("algorithm", "draft_model", "num_steps", "topk", "num_draft_tokens", "adaptive",
                "disable_by_batch_size", "raw"):
        assert key in inst["head"], key
    for key in ("tp", "pp", "dp", "ep", "nnodes", "node_rank", "dp_attention", "backend"):
        assert key in inst["parallel"], key
    for key in ("max_running", "cuda_graph_max_bs", "cuda_graph"):
        assert key in inst["concurrency"], key
    for key in ("host", "pid", "container", "source"):
        assert key in inst["where"], key
    assert inst["parallel"]["tp"] == 1 and inst["parallel"]["pp"] == 1
    assert inst["parallel"]["dp"] == 1 and inst["parallel"]["ep"] == 1
    assert inst["parallel"]["nnodes"] == 1
    assert inst["head"]["adaptive"] is False and inst["parallel"]["dp_attention"] is False
    assert inst["notes"] == []


@test
def test_mia_sglang_dspark():
    cmd = ("python -m sglang.launch_server --model-path /models/DeepSeek-V4.1-Flash "
           "--speculative-algorithm DSPARK --speculative-dspark-block-size 7 --tp 4 --ep 1")
    inst = _inst(cmd, {"DSV41_MOE_B12X_NEXT": "1"})
    assert inst["engine"] == "sglang"
    assert inst["flavor"] == "mia-dsv41"
    assert inst["head"]["algorithm"] == "DSPARK"
    assert inst["parallel"]["tp"] == 4
    assert inst["head"]["raw"].get("dspark_block_size") == "7"
    assert inst["model"] == "/models/DeepSeek-V4.1-Flash"


@test
def test_upstream_sglang_eagle_topk4():
    cmd = ("python -m sglang.launch_server --model-path /models/Qwen3.5 --speculative-algorithm EAGLE "
           "--speculative-draft-model-path /models/draft --speculative-num-steps 3 "
           "--speculative-eagle-topk 4 --speculative-num-draft-tokens 8 --cuda-graph-max-bs 32 "
           "--max-running-requests 64")
    inst = _inst(cmd)
    assert inst["engine"] == "sglang"
    assert inst["flavor"] is None
    assert inst["head"]["algorithm"] == "EAGLE"
    assert inst["head"]["topk"] == 4
    assert inst["head"]["num_steps"] == 3
    assert inst["head"]["num_draft_tokens"] == 8
    assert inst["head"]["draft_model"] == "/models/draft"
    assert inst["concurrency"]["cuda_graph_max_bs"] == 32
    assert inst["concurrency"]["max_running"] == 64
    assert inst["concurrency"]["cuda_graph"] is True


@test
def test_sglang_router():
    cmd = ("python -m sglang_router.launch_router --policy round_robin "
           "--worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001")
    inst = _inst(cmd)
    assert inst["engine"] == "sglang"
    assert inst["role"] == "router"
    assert inst["router"] is not None
    assert inst["router"]["policy"] == "round_robin"
    assert "http://127.0.0.1:8000" in inst["router"]["workers"]


@test
def test_sglang_pd_decode():
    inst = _inst("python -m sglang.launch_server --model-path /m --disaggregation-mode decode --tp 2")
    assert inst["engine"] == "sglang"
    assert inst["role"] == "decode"
    inst2 = _inst("python -m sglang.launch_server --model-path /m --disaggregation-mode prefill")
    assert inst2["role"] == "prefill"


@test
def test_sglang_equals_form_and_draft_flags():
    inst = _inst("python -m sglang.launch_server --model-path=/m --tp=8 --dp=2 --pp=1 --ep=2 "
                 "--speculative-algorithm=EAGLE --speculative-eagle-topk=1 --enable-dp-attention "
                 "--nnodes=2 --node-rank=1 --max-running-requests=16 --cuda-graph-max-bs=16 --port=30000")
    assert inst["parallel"]["tp"] == 8 and inst["parallel"]["dp"] == 2
    assert inst["parallel"]["pp"] == 1 and inst["parallel"]["ep"] == 2
    assert inst["parallel"]["dp_attention"] is True
    assert inst["parallel"]["nnodes"] == 2 and inst["parallel"]["node_rank"] == 1
    assert inst["head"]["algorithm"] == "EAGLE" and inst["head"]["topk"] == 1
    assert inst["concurrency"]["max_running"] == 16
    assert inst["port"] == 30000


# --------------------------------------------------------------------------- vLLM
@test
def test_vllm_mtp_json():
    sc = json.dumps({"method": "mtp", "num_speculative_tokens": 3, "disable_by_batch_size": 16})
    inst = _inst(["vllm", "serve", "/models/X", "--speculative-config", sc, "--max-num-seqs", "32",
                  "--tensor-parallel-size", "2"])
    assert inst["engine"] == "vllm"
    assert inst["head"]["algorithm"] == "MTP"
    assert inst["head"]["num_draft_tokens"] == 3
    assert inst["head"]["disable_by_batch_size"] == 16
    assert inst["model"] == "/models/X"
    assert inst["parallel"]["tp"] == 2
    assert inst["concurrency"]["max_running"] == 32


@test
def test_vllm_ngram():
    sc = json.dumps({"method": "ngram"})
    inst = _inst(["vllm", "serve", "/models/X", "--speculative-config", sc])
    assert inst["engine"] == "vllm"
    assert inst["head"]["algorithm"] == "NGRAM"
    assert inst["head"]["num_draft_tokens"] is None


@test
def test_vllm_headless_is_worker():
    inst = _inst(["vllm", "serve", "/models/X", "--headless", "--port", "8000"])
    assert inst["engine"] == "vllm"
    assert inst["role"] == "worker"
    assert inst["port"] == 8000


@test
def test_vllm_kv_consumer_is_decode():
    kv = json.dumps({"kv_role": "kv_consumer"})
    inst = _inst(["vllm", "serve", "/models/X", "--kv-transfer-config", kv])
    assert inst["role"] == "decode"
    kv2 = json.dumps({"kv_role": "kv_producer"})
    inst2 = _inst(["vllm", "serve", "/models/X", "--kv-transfer-config", kv2])
    assert inst2["role"] == "prefill"


@test
def test_vllm_bad_json_keeps_raw():
    inst = _inst(["vllm", "serve", "/models/X", "--speculative-config", "{not json"])
    assert inst["engine"] == "vllm"
    assert inst["head"]["algorithm"] == "NONE"
    assert inst["head"]["raw"].get("speculative_config") == "{not json"
    assert any("JSON" in n for n in inst["notes"]), inst["notes"]


@test
def test_vllm_enforce_eager_turns_cuda_graph_off():
    inst = _inst(["vllm", "serve", "/models/X", "--enforce-eager"])
    assert inst["concurrency"]["cuda_graph"] is False


# --------------------------------------------------------------------------- 其它引擎
@test
def test_llama_server_draft():
    inst = _inst("/usr/local/bin/llama-server -m /models/main.gguf -md /models/draft.gguf "
                 "--draft-max 8 --port 8080 -np 4")
    assert inst["engine"] == "llama.cpp"
    assert inst["model"] == "/models/main.gguf"
    assert inst["head"]["draft_model"] == "/models/draft.gguf"
    assert inst["head"]["num_steps"] == 8
    assert inst["concurrency"]["max_running"] == 4
    assert inst["port"] == 8080


@test
def test_llama_server_mtp_flags():
    inst = _inst("llama-server -m /models/qwen35.gguf -ngl 99 --spec-type draft-mtp --spec-draft-n-max 3")
    assert inst["engine"] == "llama.cpp"
    assert inst["head"]["algorithm"] == "MTP"
    assert inst["head"]["num_draft_tokens"] == 3


@test
def test_trtllm_serve():
    yaml = ("speculative_config:\n  decoding_type: MTP\n  max_draft_len: 3\n"
            "  draft_len_schedule:\n    0: 1\n    4: 2\n")
    inst = _inst(["trtllm-serve", "/models/X", "--tp_size", "2", "--max_batch_size", "16",
                  "--extra_llm_api_options", yaml])
    assert inst["engine"] == "trtllm"
    assert inst["parallel"]["tp"] == 2
    assert inst["concurrency"]["max_running"] == 16
    assert inst["head"]["algorithm"] == "MTP"
    assert inst["head"]["num_draft_tokens"] == 3
    assert "draft_len_schedule" in inst["head"]["raw"].get("speculative_config", {})


@test
def test_lmdeploy_serve():
    inst = _inst("lmdeploy serve api_server /models/X --tp 2 --speculative-algorithm EAGLE "
                 "--speculative-num-draft-tokens 3 --speculative-draft-model /models/draft")
    assert inst["engine"] == "lmdeploy"
    assert inst["model"] == "/models/X"
    assert inst["parallel"]["tp"] == 2
    assert inst["head"]["algorithm"] == "EAGLE"
    assert inst["head"]["num_draft_tokens"] == 3
    assert inst["head"]["draft_model"] == "/models/draft"


@test
def test_mlx_lm_server():
    inst = _inst("python -m mlx_lm.server --model /models/Qwen3.5-4B --draft-model /models/draft "
                 "--num-draft-tokens 3 --port 8080")
    assert inst["engine"] == "mlx_lm"
    assert inst["model"] == "/models/Qwen3.5-4B"
    assert inst["head"]["draft_model"] == "/models/draft"
    assert inst["head"]["num_draft_tokens"] == 3


@test
def test_mlx_serve():
    inst = _inst("/opt/mlx-serve/bin/mlx-serve --model /models/DeepSeek-V4.1")
    assert inst["engine"] == "mlx-serve"


@test
def test_tgi_launcher():
    inst = _inst("text-generation-launcher --model-id /models/TinyLlama --num-shard 2 --speculate 3 "
                 "--max-concurrent-requests 8")
    assert inst["engine"] == "tgi"
    assert inst["model"] == "/models/TinyLlama"
    assert inst["head"]["num_draft_tokens"] == 3
    assert inst["concurrency"]["max_running"] == 8


@test
def test_ollama_serve():
    inst = _inst("ollama serve")
    assert inst["engine"] == "ollama"
    assert inst["head"]["algorithm"] == "NONE"


@test
def test_tensorfold_serve_recall():
    inst = _inst("tensorfold serve /models/DeepSeek-V4.1-Flash --recall")
    assert inst["engine"] == "tensorfold"
    assert inst["model"] == "/models/DeepSeek-V4.1-Flash"
    assert inst["head"]["algorithm"] == "RECALL"


@test
def test_strata_spec_and_batch():
    inst = _inst("strata --spec 4 --batch 2 --recall --recall-dir /var/recall")
    assert inst["engine"] == "strata"
    assert inst["head"]["num_draft_tokens"] == 4
    assert inst["concurrency"]["max_running"] == 2
    assert inst["head"]["raw"].get("recall_dir") == "/var/recall"


@test
def test_maya_launch_line():
    inst = _inst("/opt/maya/build/maya-yee --glm-pack /opt/maya/pack --parallel 8",
                 {"MAYA_GLM_MULTI_MTP": "3", "RECALL_ACCEL": "1"})
    assert inst["engine"] == "maya"
    assert inst["concurrency"]["max_running"] == 8
    assert inst["head"]["algorithm"] == "MTP"
    assert inst["head"]["num_draft_tokens"] == 3


@test
def test_cuteafd_and_ktransformers():
    inst = _inst("/opt/cuteafd/bin/cuteafd --model /models/DeepSeek-V4.1")
    assert inst["engine"] == "cuteafd"
    inst2 = _inst("python -m ktransformers.server --model_name /models/Qwen3-32B")
    assert inst2["engine"] == "ktransformers"


@test
def test_unknown_command_returns_none():
    assert launch.parse_launch("sleep 100") is None
    assert launch.parse_launch("") is None
    assert launch.parse_launch([]) is None
    assert launch.parse_launch(None) is None


# --------------------------------------------------------------------------- Maya JSON
@test
def test_from_maya_config_object():
    inst = launch.from_maya_config(MAYA_JSON)
    assert inst is not None
    assert inst["engine"] == "maya"
    assert inst["concurrency"]["max_running"] == 8
    assert inst["head"]["algorithm"] == "MTP"
    assert inst["head"]["num_draft_tokens"] == 3
    assert inst["port"] == 8080
    assert inst["env"].get("MAYA_GLM_MULTI_MTP") == "3"
    assert inst["env"].get("RECALL_ACCEL") == "1"
    assert "MAYA_API_KEY" not in inst["env"]
    assert inst["yee"]["env"].get("RECALL_ACCEL") == "1"
    assert inst["yee"]["env"].get("MAYA_GLM_MULTI_MTP") == "3"


@test
def test_from_maya_config_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "maya.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(MAYA_JSON, f)
        inst = launch.from_maya_config(path)
        assert inst is not None
        assert inst["engine"] == "maya"
        assert inst["head"]["algorithm"] == "MTP"
        assert inst["concurrency"]["max_running"] == 8


@test
def test_from_maya_config_without_mtp():
    obj = dict(MAYA_JSON)
    obj["env"] = {"RECALL_ACCEL": "1"}
    obj["installer"] = {}
    inst = launch.from_maya_config(obj)
    assert inst["head"]["algorithm"] == "NONE"
    assert inst["concurrency"]["max_running"] == 8


@test
def test_from_maya_config_bad_input():
    assert launch.from_maya_config(None) is None
    assert launch.from_maya_config("/no/such/file.json") is None
    assert launch.from_maya_config(123) is None


# --------------------------------------------------------------------------- redact
@test
def test_redact_keeps_only_yee_vars():
    env = {"YEE": "1", "YEE_GIB": "0.25", "RECALL_ACCEL": "1", "RECALL_ACCEL_DIR": "/x",
           "SGLANG_D_RECALL": "1", "VLLM_D_RECALL": "1", "DSV41_MOE_B12X_NEXT": "1",
           "MAYA_GLM_MULTI_MTP": "3", "STRATA_RECALL": "1", "VLLM_USE_V2_MODEL_RUNNER": "1",
           "CUDA_VISIBLE_DEVICES": "0,1", "EP_SIZE": "1", "PYTHONPATH": "/opt/yee",
           "PATH": "/usr/bin", "HOME": "/root", "HF_TOKEN": "secret", "API_KEY": "secret",
           "MY_TOKEN": "secret", "DB_PASSWORD": "secret", "CLIENT_SECRET": "secret"}
    out = launch.redact(env)
    assert out.get("YEE") == "1" and out.get("YEE_GIB") == "0.25"
    assert out.get("RECALL_ACCEL") == "1" and out.get("RECALL_ACCEL_DIR") == "/x"
    assert out.get("SGLANG_D_RECALL") == "1" and out.get("VLLM_D_RECALL") == "1"
    assert out.get("DSV41_MOE_B12X_NEXT") == "1" and out.get("MAYA_GLM_MULTI_MTP") == "3"
    assert out.get("STRATA_RECALL") == "1" and out.get("VLLM_USE_V2_MODEL_RUNNER") == "1"
    assert out.get("CUDA_VISIBLE_DEVICES") == "0,1" and out.get("EP_SIZE") == "1"
    assert out.get("PYTHONPATH") == "/opt/yee"
    assert "PATH" not in out and "HOME" not in out
    assert "HF_TOKEN" not in out and "API_KEY" not in out
    assert "MY_TOKEN" not in out and "DB_PASSWORD" not in out and "CLIENT_SECRET" not in out
    assert launch.redact(None) == {}


@test
def test_yee_env_only_keeps_yee_families():
    inst = _inst("python -m sglang.launch_server --model-path /m",
                 {"YEE": "1", "RECALL_ACCEL_DIR": "/x", "DSV41_A": "1", "MAYA_GLM_MULTI_MTP": "3",
                  "STRATA_X": "1", "VLLM_D_RECALL": "1", "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "/opt/yee"})
    yenv = inst["yee"]["env"]
    assert yenv.get("YEE") == "1" and yenv.get("RECALL_ACCEL_DIR") == "/x"
    assert yenv.get("DSV41_A") == "1" and yenv.get("MAYA_GLM_MULTI_MTP") == "3"
    assert yenv.get("STRATA_X") == "1" and yenv.get("VLLM_D_RECALL") == "1"
    assert "CUDA_VISIBLE_DEVICES" not in yenv and "PYTHONPATH" not in yenv
    assert "CUDA_VISIBLE_DEVICES" in inst["env"]


# --------------------------------------------------------------------------- model_info
@test
def test_model_info_reads_config():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
                       "num_nextn_predict_layers": 1}, f)
        info = launch.model_info(d)
        assert info.get("arch") == "Qwen3ForCausalLM"
        assert info.get("model_type") == "qwen3"
        assert info.get("mtp_layers") == 1


@test
def test_model_info_mtp_key_order():
    for key in ("num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers", "num_mtp_modules"):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"architectures": ["X"], "model_type": "x", key: 3}, f)
            assert launch.model_info(d).get("mtp_layers") == 3, key


@test
def test_model_info_missing_returns_empty():
    assert launch.model_info("/no/such/model") == {}
    assert launch.model_info(None) == {}
    assert launch.model_info("") == {}


# --------------------------------------------------------------------------- from_server_info
@test
def test_from_server_info_sglang():
    info = {"server_args": {"model_path": "/models/X", "tp_size": 4, "speculative_algorithm": "DSPARK",
                             "max_running_requests": 64, "cuda_graph_max_bs": 32}, "port": 30000}
    inst = launch.from_server_info("sglang", info)
    assert inst is not None
    assert inst["engine"] == "sglang"
    assert inst["head"]["algorithm"] == "DSPARK"
    assert inst["parallel"]["tp"] == 4
    assert inst["concurrency"]["max_running"] == 64
    assert inst["concurrency"]["cuda_graph_max_bs"] == 32
    assert inst["model"] == "/models/X"


@test
def test_from_server_info_bad_input():
    inst = launch.from_server_info("sglang", {})
    assert inst is not None
    assert inst["engine"] == "sglang"
    assert launch.from_server_info(None, None) is not None


@test
def test_vllm_speculative_config_quotes_eaten_by_shell():
    inst = _inst("vllm serve /m --speculative-config {method:mtp,num_speculative_tokens:3,disable_by_batch_size:8}")
    assert inst["head"]["algorithm"] == "MTP", inst["head"]
    assert inst["head"]["num_draft_tokens"] == 3 and inst["head"]["disable_by_batch_size"] == 8
    assert any("引号" in n for n in inst["notes"])
    assert launch._loose_dict("{a:{b:1}}") is None and launch._loose_dict("x") is None


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
