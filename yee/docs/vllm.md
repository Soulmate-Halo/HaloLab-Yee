# vLLM 接法

这一页讲 vLLM 上怎么接 Yee：投机配置的写法、环境变量、钩子装在哪、以及限制。

## 1. 支持的加速头

vLLM 的加速头通过 `--speculative-config` 里的 `method` 指定。

| `method` | 适配器 | 状态 | 注入点 |
|---|---|---|---|
| `dspark` | `vllm_spec` | container | `DSparkSpeculator.propose`（继承 `DFlashSpeculator.propose`） |
| `dflash`（含 DFlash2） | `vllm_spec` | container | `DFlashSpeculator.propose` / `DFlash2Speculator.propose` |
| `eagle` | `vllm_spec` | container | `EagleSpeculator.propose`（V2 runner） |
| `eagle3` | `vllm_spec` | container | `EagleSpeculator.propose`（V2 runner） |
| `mtp` / Gemma4 MTP / MultiModule MTP | `vllm_spec` | container | `MTPSpeculator` / `Gemma4Speculator` / `MultiModuleMTPSpeculator` |
| `ngram` | — | unsupported | V2 的 `init_speculator` 里没有 ngram speculator |
| `suffix` | — | unsupported | 没有 V2 speculator |

状态含义见 [heads.md](heads.md)。已验证过的版本：vLLM `0.28.1rc1.dev388+g8a728663c`（镜像 `vllm-dsv41-x86:compat6`）。

## 2. 必须打开 V2 runner

钩子装在 `vllm.v1.worker.gpu` 这一层的类上，所以必须：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

不设的话 `yee auto` 会检测到 vLLM，但钩子装不上（`yee doctor` 会 FAIL）。

## 3. 装钩子

```bash
yee auto --cmdline "vllm serve /models/DeepSeek-V4.1-Flash --speculative-config '{\"method\":\"eagle\"}'" --apply
```

或者直接 `yee auto --apply`（从进程表找正在跑的服务）。

## 4. 环境变量

```bash
export YEE=1
export VLLM_USE_V2_MODEL_RUNNER=1
export YEE_BACKEND=memory
export YEE_GIB=0.25
export YEE_TOKENIZER_DIR=/models/DeepSeek-V4.1-Flash
```

## 5. 采样请求

vLLM V2 的投机器在默认的 `draft_sample_method=greedy` 下，`draft_logits` 为 None，此时采样批次可以参与复制（把复制 token 换成 one-hot 提议，走 target-only 验证）。

如果投机器给出概率草稿（probabilistic drafter，缓存了 q），复制 token 就必须沿用那个 q 的概率分布，不能再改成 one-hot，否则分布会变。这种情况保持只贪心。

开关是 `YEE_NONGREEDY`（等价 `RECALL_ACCEL_NONGREEDY`），默认 1，设 0 回到只贪心。详见 [nongreedy.md](nongreedy.md)。

## 6. 限制

- **V2 runner 限定**：V1 runner 上没有这些投机器类，Yee 不生效。
- **结构化输出**：带 structured output 的请求不参与复制。
- **概率草稿**：`draft_sample_method` 不是 greedy 时，采样批次不参与复制（贪心批次仍然参与）。
- **跨 tokenizer 不能共用**：记录目录按 tokenizer digest 隔离。
- **container 级别的组合没有 GPU 服务实测**，提速多少要用 `yee bench` 实测。

## 7. 验收

```bash
yee doctor
yee selftest
yee bench --url http://127.0.0.1:8000 --model <served-model-name> --rounds 2
```

`bench` 要求 `replies identical: True`，且第 2 次比第 1 次快。
