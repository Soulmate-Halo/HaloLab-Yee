# SGLang 接法

这一页讲 SGLang 上怎么接 Yee：各加速头的支持情况、环境变量、两种装钩子的方式、以及限制。

## 1. 支持的加速头

| 加速头（`--speculative-algorithm`） | 适配器 | 状态 | 注入点 |
|---|---|---|---|
| `DSPARK` | `sglang_dspark` | verified | `DSparkWorkerV2._proposer.propose` 的输出 |
| `DFLASH` | `sglang_dflash` | container | 构造 `DFlashVerifyInput` 之前改写 `_draft_block_tokens_buf` |
| `EAGLE` / `EAGLE3` / `NEXTN` / MTP 类 / `STANDALONE` / `UNO` | `sglang_eagle` | container（仅 topk=1） | `build_eagle_verify_input`，在 `build_tree_kernel_efficient` 之前替换 `draft_tokens` |
| `NGRAM` | `sglang_ngram` | container | `NGRAMWorker._prepare_draft_tokens` 的返回值，整块换成链，同时换成下三角 mask |
| `EAGLE*` 且 `--speculative-eagle-topk` > 1 | — | no-op | 树形草稿，不安全，不处理 |
| `FROZEN_KV_MTP` | — | unsupported | 它的 `draft()` 自己构造验证输入，没有挂钩 |

状态含义：

- **verified**：真机实测过，输出逐字一致，并且有提速。
- **container**：在真实镜像里检查过符号、签名和源码标记，钩子能装到真实类上，CPU 模拟测试通过；还没有起 GPU 服务实测。
- **no-op / unsupported**：能识别出来，但不打补丁，服务原样运行。

已验证过的版本：SGLang `0.0.0.dev0`（镜像 `dsv41-4x-spark:canary-roce-pd`）。其他版本先跑 `yee doctor`，结果为 FAIL 时钩子不会被安装。

## 2. 两种装钩子的方式

### 2.1 `.pth` 方式（普通环境）

```bash
yee auto --apply
```

把 `yee.pth` 写进当前解释器的 site-packages，解释器启动时导入钩子。服务 worker 进程继承环境变量，所以每个 worker 都会装上。

### 2.2 sitecustomize 模板方式（Mia 镜像）

Mia 镜像启动时会先执行它自己的 sitecustomize，用 `.pth` 方式会冲突。改用模板：

```bash
cp $(python3 -c "import yee,os;print(os.path.dirname(yee.__file__))")/integrations/sitecustomize_template.py /opt/yee/sitecustomize.py
export PYTHONPATH=/opt/yee:$PYTHONPATH   # 必须排在最前面
```

模板会先按原样执行 Mia 的 sitecustomize，再装 Recall 钩子。任何 Recall 失败只打印一行，服务照常起。

模板读的开关是 `SGLANG_D_RECALL`（等价于 `YEE`），Mia sitecustomize 的路径由 `SGLANG_D_RECALL_MIA_SITECUSTOMIZE` 指定，默认 `/opt/dsv41/adapter/sitecustomize.py`。

## 3. 环境变量

```bash
export YEE=1                      # 或 RECALL_ACCEL=1，等价
export YEE_BACKEND=memory         # disk / memory / cuda；想跨重启保留记录用 disk
export YEE_GIB=0.25               # 记录池大小
export YEE_TOKENIZER_DIR=/models/DeepSeek-V4.1-Flash
```

SGLang 特有的：

- `YEE_ADAPTER`（等价 `RECALL_ACCEL_ADAPTER`）：`auto` / `dspark` / `dflash` / `eagle` / `ngram` / `off`。默认 `auto`，此时所有可导入的 SGLang 适配器都装上，服务实际用的那个加速头的代码路径才会触发钩子。
- `YEE_NONGREEDY`（等价 `RECALL_ACCEL_NONGREEDY`）：默认 1，采样请求也走召回；设 0 只保留贪心。详见 [nongreedy.md](nongreedy.md)。
- `YEE_WINDOWS`（等价 `RECALL_ACCEL_WINDOWS`）：默认 `16,8,4`。16 token 的主窗口查不到时，退到 8、4 token 的短窗口；设成 `16` 关闭短窗口回退。

## 4. 特殊场景

### 没有加速头

```bash
--speculative-algorithm NGRAM
```

NGRAM 不需要草稿模型就能用上复制草稿。

### DSPARK 的采样请求

DSPARK 的非贪心验证用 `SoftmaxTemp(corrected_logits)` 作提议分布 q，被替换的行原地改成 one-hot；拿不到 `corrected_logits` 的批次不参与复制。

### EAGLE 的 topk

`--speculative-eagle-topk` 必须为 1（不设也算 1）。大于 1 是树形草稿，复制不安全，整条路是 no-op。

### 拒绝采样与 selector

- `draft_probs` 非空（拒绝采样模式）：不替换草稿。
- DFLASH 的 selector 采样器：开启时不替换草稿。

### grammar / 结构化输出

带 grammar 的批次不参与复制（所有 TP rank 行为一致）。

## 5. 限制

- **只对贪心生效（2.1.0 之前）**：2.1.0 起采样请求也可以走召回，但仅限 target-only 验证的加速头。
- **块长或步数要按模型单独调**：例如 DSPARK 的 `--speculative-dspark-block-size`、EAGLE 的 `--speculative-num-steps`。DeepSeek V4.1 在 Spark 上测得 7 或 8 最合适，换模型要重新测。
- **跨 tokenizer 不能共用**：记录只认 token id，模型和基座无关，但不同 tokenizer 的记录不能混用，目录按 digest 自动隔离。
- **container 级别的组合没有 GPU 服务实测**，提速多少要用 `yee bench` 实测。

## 6. 验收

```bash
yee doctor
yee selftest
yee bench --url http://127.0.0.1:8000 --model <served-model-name> --rounds 2
```

`bench` 要求 `replies identical: True`，且第 2 次比第 1 次快。
