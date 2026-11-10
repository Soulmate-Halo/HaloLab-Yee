# TensorFold 接法

这一页讲 TensorFold 上怎么用 Recall：它自带的能力、需要 Yee 补丁的缺口、哪些引擎不读 Recall。

## 1. TensorFold 自带 Recall

TensorFold 是 Zig 写的原生程序，没有 Python 插件接口，所以**不需要打 Yee 钩子**，用它自己的 Recall：

```bash
tensorfold serve MODEL --recall
```

| 开关 | 含义 |
|---|---|
| `--recall` | 保留记录并从中起草稿 |
| `--recall-dir DIR` | 记录自己的目录。不设时在 `~/.cache/tensorfold/recall/<分词器身份>` 下 |
| `--recall-gib GIB` | 记录的磁盘额度，默认 1/16 GiB（64 MiB）；超了先删最旧的 |
| `--recall-read-only` | 只从记录起草稿、不往里写 |

后三个开关都隐含 `--recall`。

- **版本要求**：1.0.12 起才有 Recall。低于这个版本 `yee tensorfold detect` 会提示需要升级，`yee tensorfold args` 会报错退出，不会硬加参数。
- **记录按分词器身份隔离**：目录名是分词器的 identity（词表、merges、added tokens 的摘要，加词表大小和模型名）。tokenizer 不同的服务会拒绝打开同一个目录，并说明是哪个 tokenizer 写的。
- **重启后还在**：记录在磁盘上，服务重启不丢。
- **全局共享**：同一台机器上所有用户、所有请求共享同一份记录。含义见 [sharing.md](sharing.md)。
- **最短匹配 16 个 token**：记录里连续 128 个或更多 id 已经有的部分不再重复写入，所以同一条请求发两次，第二次不会新增内容。

## 2. 状态查询与清空

```bash
GET /v1/recall      # 报告记录：状态、磁盘与内存占用、提供过多少次复制、复制落地了多少 token
DELETE /v1/recall   # 清空记录
```

服务配了 API key 时这两个接口都要带 key。

Yee 侧也有等价命令：

```bash
yee tensorfold status --url http://127.0.0.1:8080
yee tensorfold clear  --url http://127.0.0.1:8080
```

## 3. 记录互转

用 SGLang/vLLM 积累的记录给 TensorFold 预热，或者反过来：

```bash
yee tensorfold export --from ~/.cache/recall_accel/<digest16> --to ~/.cache/tensorfold/recall/<id16>
yee tensorfold import --from ~/.cache/tensorfold/recall/<id16> --to ~/.cache/recall_accel/<digest16> --tokenizer-dir /models/<模型>
```

- export 需要 TensorFold 自己的 identity digest（它对 tokenizer 内部数据做 sha256，和 Yee 的 digest 不同）。做法是先用 `--recall-dir <目标目录>` 启动一次 TensorFold，它会写出 `identity` 文件，export 会自动读取。也可以用 `--digest <64 位 hex> --vocab <n>` 手动指定。
- 两个方向都会先校验 vocab 是否一致，不一致就拒绝。export 时含有越界 id 的段会被丢弃，并计数报告。

## 4. 上游的缺口：priced 提议器给不出草稿时不问 Recall

TensorFold 里有一类 **priced 提议器**（它给出的草稿自带可信度，不走普通的复制窗口路径）：DeepSeek V4.1 设 `TF_DSV41_LOOKUP_PLAN=1` 时的 lookup 计划、Llama 的 drafter、Qwen4 的 device lanes。它们给不出草稿时，**上游不会去问 Recall**。

Yee 的补丁 `tensorfold/0001-recall-after-priced.patch` 加了开关 `TF_RECALL_AFTER_PRICED=1`：priced 提议器给空时改问 Recall。

```bash
yee patch list
yee patch show tensorfold
yee patch apply tensorfold --src <TensorFold 仓库根> [--check]
```

打完按 TensorFold 自己的构建方式重新编译（Zig 版本见仓库的 `.zig-version`，当前 0.17.0）。启动：

```bash
TF_RECALL_AFTER_PRICED=1 tensorfold serve MODEL --recall
```

**不设 `TF_RECALL_AFTER_PRICED` 时，行为与上游完全一致。**

### 补丁基线与规模

- 基线：`ashhart/TensorFold` commit `33ada77c3122fca844a6825316ce8bbabaca05be`
- 改动文件：`zig/src/core/lanes/windows.zig`、`zig/src/core/lanes/config.zig`、`zig/src/core/lanes/recall_engine_test.zig`
- 规模：74 行增、7 行删

### 验证情况

- Windows 上 Zig 0.17.0 跑 lanes 召回相关 26 个测试全过（含新增 2 个，其中一个覆盖贪心、T=0.8、T=1.0 三档输出逐 token 一致且轮数更少）。
- WSL Linux 上 `test-dsv41-draft`、`test-lane-api-host`、`test-shared-lane-regressions-host`、`test-family-copies-host` 在原版和打补丁版都通过；`compile-dsv41-server` 打补丁后编译通过。
- **没有在真 GPU 上跑过。**

## 5. 哪些引擎不读 Recall

| 引擎 | 状态 | 原因 |
|---|---|---|
| Qwen4（native_host） | 拒绝 | `Qwen4: --recall and --record are unsupported` |
| Kimi Metal | 拒绝 | 自有驱动不实现学到的状态和磁盘 prompt 状态 |
| Drama | 拒绝 | 同上 |
| Mac 上的 Flash Next（`qwen4_exp`） | 启动前拒绝 | 它的 replay 引擎在 GPU 上从自己的历史复制，不读记录 |
| Mac 上单流 Nemotron | 不读 | 走自有驱动 |
| Mac 上单流 Qwen3.8-27B | 不读 | 走自有驱动 |

上游原文：Recall 在 lane 引擎的复制窗口和嫁接 lane 里起草稿；CUDA 上单流 Nemotron 的自有驱动会在上下文没东西可复制时问记录；Mac 上单流 Nemotron 和单流 Qwen3.8-27B 跑自己的驱动，还不读记录。

## 6. 配方里默认没开

MiaAI-Lab 的 **DeepSeek-V4.1-Flash 4x DGX Spark TensorFold 配方默认没有加 `--recall`**，要自己在启动参数里加：

```bash
tensorfold serve MODEL --recall
```

需要 priced 兜底时再加 `TF_RECALL_AFTER_PRICED=1` 和 Yee 补丁。

## 7. 采样请求也是无损的

TensorFold 的采样请求（temperature>0、top-k/top-p）用**带种子的确定性采样逐位验证**，召回草稿同样无损。上游测试 `"greedy and sampled"` 覆盖了这两种情形。

## 8. 跑基准时的注意事项

记录被填满之后，任何重复 prompt 的基准都会比全新服务快。做 release gate 和速度对比时，请关掉 Recall，或者用空的记录加 `--recall-read-only`，并在结论里写清楚用的是哪种。

服务的启动行和 `GET /v1/recall` 会给出记录的状态和大小。
