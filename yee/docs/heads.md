# 基座 × 加速头支持矩阵

这一页列出各个基座上哪些加速头能被 Yee 加速、哪些不能、以及不能的原因。

用 `yee bases` 可以看同一张表（`--json` 输出机器可读格式）。

## 1. 状态含义

| 状态 | 含义 |
|---|---|
| verified | 真机实测过，输出逐字一致，并且有提速 |
| container | 在真实镜像里检查过符号、签名和源码标记，钩子能装到真实类上，CPU 模拟测试通过；**还没有起 GPU 服务实测** |
| no-op | 能识别出来，但这条组合不安全，不打补丁，服务原样运行 |
| unsupported | 识别得出来，但没有可挂钩的点，不打补丁，服务原样运行 |
| native | 引擎自带这个能力，不需要 Yee 挂钩 |

## 2. SGLang

| 加速头（`--speculative-algorithm`） | 适配器 | 状态 | 原因 / 注入点 |
|---|---|---|---|
| `DSPARK` | `sglang_dspark` | verified | `DSparkWorkerV2._proposer.propose` 的输出；已在 ysy37–40（Mia 镜像）实测 |
| `DFLASH` | `sglang_dflash` | container | `DFlashWorkerV2` 的块草稿；构造 `DFlashVerifyInput` 之前改写 `_draft_block_tokens_buf` |
| `EAGLE` | `sglang_eagle` | container | `build_eagle_verify_input`，在 `build_tree_kernel_efficient` 之前替换 `draft_tokens` |
| `EAGLE3` | `sglang_eagle` | container | 与 EAGLE 同一条路径 |
| `NEXTN` | `sglang_eagle` | container | NEXTN 是 EAGLE 的别名 |
| `MTP`（及各类 `*_MTP`） | `sglang_eagle` | container | MTP 加速头跑的是 EAGLE worker |
| `STANDALONE` | `sglang_eagle` | container | `StandaloneWorkerV2(EAGLEWorkerV2)` |
| `UNO` | `sglang_eagle` | container | `uno_worker_v2.py:550` 也经由 `build_eagle_verify_input` |
| `NGRAM` | `sglang_ngram` | container | `NGRAMWorker._prepare_draft_tokens` 的返回值，整块换成链并换下三角 mask；不需要草稿模型 |
| `EAGLE*` 且 `--speculative-eagle-topk` > 1 | — | no-op | 树形草稿不安全，复制只在 topk=1 的链上跑 |
| `FROZEN_KV_MTP` | — | unsupported | 它的 `draft()` 自己构造验证输入，没有可挂钩的点 |
| 没有加速头 | — | unsupported | 加 `--speculative-algorithm NGRAM` 就能在没有草稿模型的情况下用上复制草稿 |

别名归并规则：`EAGLE-3` → `EAGLE3`，`DEEPSEEK_MTP` / `QWEN3_NEXT_MTP` / `MIMO_MTP` / `ERNIE_MTP` / `GLM4_MOE_MTP` / `LONGCAT_FLASH_MTP` → `MTP`，`DFLASH2` → `DFLASH`，`NGRAM_GPU` → `NGRAM`。

已验证版本：SGLang `0.0.0.dev0`（镜像 `dsv41-4x-spark:canary-roce-pd`）。

## 3. vLLM（V2 runner）

| 加速头（`--speculative-config` 的 `method`） | 适配器 | 状态 | 原因 / 注入点 |
|---|---|---|---|
| `dspark` | `vllm_spec` | container | `DSparkSpeculator`（继承 `DFlashSpeculator.propose`） |
| `dflash` / `dflash2` | `vllm_spec` | container | `DFlashSpeculator` / `DFlash2Speculator` 的 `propose` |
| `eagle` | `vllm_spec` | container | `EagleSpeculator.propose`（V2 runner） |
| `eagle3` | `vllm_spec` | container | 与 eagle 同一条路径 |
| `mtp` / Gemma4 MTP / MultiModule MTP | `vllm_spec` | container | `MTPSpeculator` / `Gemma4Speculator` / `MultiModuleMTPSpeculator` |
| `ngram` | — | unsupported | V2 的 `init_speculator` 里没有 ngram speculator |
| `suffix` | — | unsupported | 没有 V2 speculator |

已验证版本：vLLM `0.28.1rc1.dev388+g8a728663c`（镜像 `vllm-dsv41-x86:compat6`）。

**必须设 `VLLM_USE_V2_MODEL_RUNNER=1`**，否则钩子装不上。

## 4. TensorFold

| 项 | 状态 | 原因 |
|---|---|---|
| TensorFold 自带 Recall（≥ 1.0.12） | native | 不打补丁：用 `tensorfold serve MODEL --recall`，Yee 只生成启动参数、做记录双向转换、查状态和清空 |
| priced 提议器（DeepSeek V4.1 的 `TF_DSV41_LOOKUP_PLAN=1` lookup 计划、Llama drafter、Qwen4 device lanes） | 需 Yee 补丁 | 上游在 priced 提议器给空时不去问 Recall；Yee 补丁加了 `TF_RECALL_AFTER_PRICED=1` |
| Qwen4（native_host） | 拒绝 | `Qwen4: --recall and --record are unsupported` |
| Kimi Metal | 拒绝 | 自有驱动不实现学到的状态和磁盘 prompt 状态 |
| Drama | 拒绝 | 同上 |
| Mac 上的 Flash Next（`qwen4_exp`） | 启动前拒绝 | 它的 replay 引擎在 GPU 上从自己的历史复制，不读记录 |
| Mac 上单流 Nemotron | 不读记录 | 走自有驱动 |
| Mac 上单流 Qwen3.8-27B | 不读记录 | 走自有驱动 |

CUDA 上单流 Nemotron 的自有驱动会在上下文没东西可复制时问记录。

## 5. cuteafd

| 项 | 状态 | 原因 |
|---|---|---|
| DeepSeek V4.1 native serve | 需 Yee 补丁 | Rust 原生服务没有插件接口；打补丁后走进程内环形记录 |
| 只对 greedy 请求生效 | 限制 | 采样请求走不到复制草稿 |
| 记录只在进程内存 | 限制 | D 端重启后清空，需要重新积累 |

## 6. mlx-serve（M5 xypd 版）

| 项 | 状态 | 原因 |
|---|---|---|
| DeepSeek V4.1 DSpark 复制草稿 | 需 Yee 补丁 | Zig 原生服务，用 source-drop 替换四个源文件 |
| 贪心请求 | 支持 | 走召回草稿 |
| 采样请求 | 支持 | 走召回草稿 + one-hot q 拒绝采样验证，输出分布不变 |
| 带 penalty 的请求 | 不走投机 | 需要在 verify 里逐位累计 penalty，改动面大，本轮未做 |
| grammar / logprobs | 排除 | 同上 |

## 7. 全局的限制

- **跨 tokenizer 不能共用记录**：记录只认 token id，模型和基座无关，但不同 tokenizer 的记录不能混用，目录按 digest 自动隔离。
- **grammar / 结构化输出不参与复制**。
- **拒绝采样模式（`draft_probs` 非空）不替换草稿**；DFLASH 的 selector 采样器开启时也不替换。
- **块长或步数要按模型单独调**（如 `--speculative-dspark-block-size`、`--speculative-num-steps`）。
