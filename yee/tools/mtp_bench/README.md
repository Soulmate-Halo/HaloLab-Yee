# mtp_bench（Yee 3.5.2）

llama.cpp 投机解码（`--spec-type draft-mtp` 等）的测速工具。只用 Python 标准库和 bash，在跑 llama-server 的机器上执行，不需要 curl。

| 文件 | 用途 | 命令 |
|---|---|---|
| `run_matrix.sh` | 每个配置单独起一次 llama-server，跑 15 条提示，最后出对比表 | `BIN=<build/bin> MODEL=<gguf> OUT=<目录> ./run_matrix.sh base gpu_k3 cpu_k3` |
| `bench.py` | 对一个已经在跑的 llama-server 测速，记录每条的 tok/s、草稿接受数、输出哈希 | `python3 bench.py --base http://127.0.0.1:8080 --prompts prompts.jsonl --tag x --out x.json --warmup` |
| `compare.py` | 汇总多份 bench 结果，按 (id, rep) 比对输出是否逐字一致 | `python3 compare.py --inputs a.json b.json --baseline a --out compare.md` |
| `prompts.jsonl` | 15 条提示：code、copy、rewrite、qa、creative 各 3 条 | - |
| `contention.sh` | CPU 与 GPU 同时读内存时各自慢多少，决定草稿头能不能放 CPU | `BIN=<build/bin> MODEL=<gguf> ./contention.sh` |
| `vocab_coverage.py` | 生成 token 的 id 有多少落在前 N 个，用来选 `YEE_MTP_VOCAB_N` | `python3 vocab_coverage.py --base http://127.0.0.1:8080` |
| `mock_llama_server.py` | 假的 llama-server，给上面几个脚本做自测 | `python3 mock_llama_server.py --port 18771` |

`run_matrix.sh` 的预定义配置：`base`（不投机）、`gpu_k1`～`gpu_k3`（MTP 草稿在 GPU，草稿数 1～3）、`cpu_k1`～`cpu_k3`（MTP 草稿在 CPU：`-md` 同一个 gguf、`-devd none -ngld 0 -td $DRAFT_THREADS`）。`EXTRA_ARGS` 追加 llama-server 参数；环境变量（例如 `YEE_MTP_VOCAB_N`）会传给 llama-server。脚本只停自己启动的进程。

mx4（Ryzen AI Max+ 395，Vulkan，Qwen3.6-27B Q4_K_M）的结果见 `python -m yee doc llama-mtp`。
