# HaloLab Yee 加速模块 4.2.0

Yee 是一个面向大模型推理服务的投机解码加速模块。它把服务读过、写过的 token 记下来，新请求和历史内容重复时，直接把后续内容复制成草稿，交给服务正在用的加速头（草稿头）去验证。每个复制来的 token 都由目标模型逐个核对，命中得越多，解码步数越少。

4.2.0 在召回之外还带了：逐请求自适应草稿数（KAD）、提问哈希索引（qhash）、热专家验证预算、逐请求 conf（vcap），**全部默认打开**，不需要设任何环境变量，也不改你的启动参数。

- 下载：[Releases · v4.2.0](https://github.com/Soulmate-Halo/HaloLab-Yee/releases/tag/v4.2.0)
- 一键安装包 `Yee-4.2.zip`：897,516 字节，sha256 `e8718a6b294d51a4d33c653f3891389890af83f09c38642ef794cd298ece6a8d`
- wheel `yee-4.2.0-py3-none-any.whl`：846,668 字节，sha256 `f6d1e0fde883b60b8d05bbab413bb40c7f8ca3bc155795723ea4ddb7444c6005`

## 1. 支持的基座和加速头

状态说明：

| 状态 | 含义 |
|---|---|
| verified | 真机实测过，输出逐字一致，并且有提速 |
| container | 在真实镜像里检查过符号、签名和源码标记，钩子能装到真实类上，CPU 模拟测试通过，还没有起 GPU 服务实测 |
| source-checked / built | 对照引擎源码检查过（或补丁已编译），还没有上线实测 |
| native | 引擎自带这项能力，Yee 不挂钩，服务原样运行 |
| unsupported / no-op | 能识别，但没有可挂钩的点或组合不安全，Yee 不介入，服务原样运行 |

用 `python -m yee bases` 可以在本机看同一张表（`--json` 为机器可读格式）。

### 1.1 SGLang（`--speculative-algorithm`）

| 加速头 | 召回 | KAD | 说明 |
|---|---|---|---|
| `DSPARK` | verified | native | 4 台 DGX Spark（Mia 镜像，DeepSeek-V4.1-Flash）实测；块长 7 + `DSV41_VERIFY_CAP=conf:0.1` 最快 |
| `DFLASH`（含 `DFLASH2`） | container | native | 改写 `DFlashWorkerV2` 的块草稿 |
| `EAGLE` / `EAGLE3` / `NEXTN` | container | source-checked | 链式草稿（`--speculative-eagle-topk 1`） |
| `MTP` 及各类 `*_MTP`（`DEEPSEEK_MTP`、`QWEN3_NEXT_MTP`、`MIMO_MTP`、`ERNIE_MTP`、`GLM4_MOE_MTP`、`LONGCAT_FLASH_MTP`） | container | source-checked | 走 EAGLE worker |
| `STANDALONE` | container | source-checked | `StandaloneWorkerV2` |
| `UNO` | container | native | 经由 `build_eagle_verify_input` |
| `NGRAM`（含 `NGRAM_GPU`） | container | native | 不需要草稿模型；没有加速头的服务加 `--speculative-algorithm NGRAM` 就能用召回 |
| `EAGLE*` 且 topk > 1（树形草稿） | no-op | native | 不介入 |
| `FROZEN_KV_MTP` | unsupported | — | 没有可挂钩的点 |

已验证版本：SGLang `0.0.0.dev0`（镜像 `dsv41-4x-spark`）。SGLang 上游正式版 0.5.10.post1 没有逐请求变草稿数的机制，KAD 不装，召回照常。

### 1.2 vLLM（V2 runner，`--speculative-config` 的 `method`）

| 加速头 | 召回 | KAD |
|---|---|---|
| `dspark` | container | verified（生产实测） |
| `dflash` / `dflash2` | container | 已适配 |
| `eagle` / `eagle3` | container | 已适配 |
| `mtp` / Gemma4 MTP / 多模块 MTP | container | 已适配 |
| `draft_model` | — | 已适配 |
| `ngram` / `suffix` | unsupported | V2 runner 下不支持 |

已验证版本：vLLM `0.28.1rc1.dev388+g8a728663c`。**必须设 `VLLM_USE_V2_MODEL_RUNNER=1`**，钩子装在 V2 runner 上。KAD 以 `num_speculative_tokens = 12` 启动，每个请求在 5～12 之间选草稿数。

### 1.3 其它基座

| 基座 | 加速头 / 路径 | 接入方式 | 状态 |
|---|---|---|---|
| llama.cpp | 召回：带草稿模型（`-md`）或不带草稿模型 | 补丁 `llama.cpp-recall`（先打 `llama.cpp-kad`） | 已提供补丁 |
| llama.cpp | KAD：`draft-simple`（`-md`） | 补丁 `llama.cpp-kad` | verified |
| llama.cpp | KAD：`eagle3`、`mtp` | 补丁 `llama.cpp-kad` | built |
| llama.cpp | `ngram-simple` / `ngram-mod` / `ngram-map` / `ngram-cache` | — | native |
| llama.cpp | qwen35 / qwen35moe 的 MTP 草稿头（`--spec-type draft-mtp`）截断词表 | 补丁 `llama.cpp-mtp-vocab`，`YEE_MTP_VOCAB_N=131072` | mx4 实测 k=3 快 5.0% |
| TensorRT-LLM | 配了 `draft_len_schedule` 的投机方法（Draft-Target、NGram、两模型 EAGLE3 等）、一模型 EAGLE3 / MTP | Python 钩子 `trtllm_kad` | source-checked |
| LMDeploy | deepseek_mtp、eagle3、dflash、hy3_mtp、qwen3_5_mtp | — | native |
| mlx-lm | `--draft-model`（`speculative_generate_step`） | Python 钩子 `mlx_lm_kad` | verified |
| transformers | 辅助生成（AssistedCandidateGenerator 及子类）、prompt lookup | Python 钩子 `hf_kad` | verified |
| transformers | MTP（`use_mtp`）、DFlash | — | native |
| TensorFold | 自带 Recall（≥ 1.0.12，`tensorfold serve MODEL --recall`） | 不用补丁 | native |
| TensorFold | priced 提议器（DeepSeek V4.1 lookup 计划、Llama drafter、Qwen4 device lanes）、lane 引擎 head 草稿 | 补丁 `tensorfold`、`tensorfold-kad` | built |
| cuteafd | DeepSeek V4.1 native serve 的 DSpark（贪心请求） | 补丁 `cuteafd`（v1 / v2） | 已提供补丁 |
| mlx-serve（M5 xypd 版） | DeepSeek V4.1 DSpark：贪心与采样请求 | 补丁 `mlx-serve`、`mlx-serve-2.4`、`mlx-serve-kad` | source-checked |
| strata 0.1.59 | MTP 草稿单流窗口、`--batch-mtp` 批处理槽 | 补丁 `strata-recall` | 补丁检查与单测通过，未上 GPU 实测 |

### 1.4 SGLang Mia DeepSeek-V4.1 专项

| 项 | 默认 | 说明 |
|---|---|---|
| 逐请求 conf（`mia-dsv41-vcap`） | 打开（`{"mode": "adapt"}`） | 镜像里的 `verify_cap.py` 是补丁所基于的原件时才放入 |
| 热专家验证预算（`YEE_EXPERT_BUDGET`） | 打开，`YEE_EXPERT_BUDGET_P=0.9` | 每次 MoE 调用保留覆盖本步 90% 路由权重的热专家；4 台 DGX Spark 实测合计吞吐 +17.3%（并发 8 的 new 类 +22.5%） |
| 召回感知验证预算（`YEE_RECALL_BUDGET`） | 打开 | 只在 compact 变长验证下起作用 |
| 图片 token 预算对齐（`mia-dsv41-image-budget`） | 需显式打 | vLLM 做 P、SGLang Mia 做 D 时使用 |

## 2. 部署条件

### 2.1 通用

- **Python ≥ 3.8**，唯一依赖是 `numpy`（推理镜像里一般已有）。
- 必须装在**服务实际运行的那个 Python 环境**里（通常是镜像里的 python）。
- 多卡 / 多机（TP）部署时，**每个 TP rank 所在的机器都要装同一版本的 Yee**。版本不一致或有机器没装时，召回广播和专家预算在所有 rank 上一起关闭，服务照常运行，日志打印 `rank check off`。
- 召回只认 token id：不同 tokenizer 的记录不能混用，记录目录按 tokenizer digest 自动隔离。
- 召回对和历史内容重复的部分起作用，同一段内容第二次出现时才明显变快。grammar / 结构化输出不参与复制。
- 默认召回记录容量 0.25 GiB（`RECALL_ACCEL_GIB`），索引密度 1，写满约占 1 GiB 内存。多进程可以共用一个磁盘记录目录。

### 2.2 各基座

| 基座 | 条件 |
|---|---|
| SGLang | 已启用上表中的某个加速头；已验证镜像为 Mia `dsv41-4x-spark` |
| vLLM | `VLLM_USE_V2_MODEL_RUNNER=1` |
| 热专家验证预算 | Mia 镜像的 `moe_b12x_next` 确定性路径：`DSV41_MOE_B12X_NEXT=1`、`DSV41_MOE_B12X_NEXT_DETERMINISTIC=1`、`EP_SIZE=1`；其它 MoE 路径上自动不介入，其余加速照常 |
| llama.cpp | 基线 v0.9.0 `cd959bae`（`llama.cpp-kad`、`llama.cpp-recall`）或 `a3b9c23e`（`llama.cpp-mtp-vocab`），打补丁后重新编译 llama-server |
| TensorFold | ≥ 1.0.12 直接用自带 Recall；补丁基线 `ashhart/TensorFold 33ada77c` |
| cuteafd | 源码 revision `9f53cc6c`，`yee patch apply cuteafd --src <源码根> --variant v2` 后重新编译 |
| mlx-serve | M5 xypd 版源码，替换源文件后重新编译 |
| strata | `Niko1221/Strata 712fec6b`（引擎 0.1.59），打补丁后重新编译 CUDA 版 |

### 2.3 一键安装（MiaAI dsv41-4x-spark，多台 DGX Spark）

- 在 **head 宿主机**上，用平时运行 `./start-tp4.sh` 的那个用户操作。
- head 能用 `.env.tp4` 里的 `SSH_IDENTITY` 免密 ssh 到每台 worker（与 `start.sh` 的要求相同）。
- 机器上有 `bash`、`python3`、`docker`。安装脚本只用 Python 3.8+ 标准库。
- 方案目录里同时有 `start.sh` 和 `.env.tp4`（或 `.env`）。脚本会在当前目录、脚本上级目录、`/var/tmp/mia/*`、`~/*`、`/opt/*` 里自动找；找到多个时用 `--recipe <方案目录>` 指定。

## 3. 安装

### 3.1 Mia dsv41-4x-spark

```bash
unzip Yee-4.2.zip && cd Yee-4.2
bash install.sh --check       # 只检查，不写任何东西；最后一行是「检查通过，可以安装」再继续
bash install.sh --restart     # 4 台都装 + 重启 + 健康检查 + 冒烟；失败自动回退
```

脚本把 Yee 装到每台机器的 `~/.cache/yee42/` 并逐台核对 sha256，备份 env 文件后在末尾追加一段只写 `PYTHONPATH` 和 `YEE=1` 的标记块。块长、并发、EP、MoE 后端等启动参数都不改；Yee 从 `--speculative-dspark-block-size N` 读出块长，只给没设的变量补上配套值（`DSPARK_BLOCK_SIZE=N`、`DSV41_VERIFY_CAP=conf:0.1`、`DSV41_VERIFY_CAP_STRIDE=N+1`、`SGLANG_DSPARK_FOLDED_SAMPLING=2`）。

退出码：0 成功；1 检查或安装出错（env 文件没动）；2 服务已起来但 Yee 自己关掉了（日志里有 `rank check off`）；3 重启失败且已回退。

完整参数和常见问题见 [install/安装说明.md](install/安装说明.md)。

### 3.2 其它单机 Python 推理服务

SGLang、vLLM（V2 runner）、TensorRT-LLM、LMDeploy、mlx-lm、transformers 等，在服务所用的 Python 环境里：

```bash
bash install.sh
# 等价于：
python3 -m pip install --upgrade yee-4.2.0-py3-none-any.whl && python3 -m yee install-hook
```

之后从这个 Python 启动的服务都会自动加载 Yee，照常启动即可。

### 3.3 源码基座（llama.cpp、TensorFold、cuteafd、mlx-serve、strata）

```bash
yee patch list                                # 列出全部补丁和基线
yee patch apply <补丁名> --src <源码根>        # 应用前核对 sha256，不匹配就拒绝
```

打完补丁重新编译。各补丁的开关和回退方法见 `yee doc patches`。

## 4. 确认生效

```bash
yee doctor             # 各适配器需要的符号、签名、源码标记；全部 PASS 才算通过
yee selftest           # 自带测试组，不需要 pytest
python -m yee config   # 每个开关的生效值与来源（env / default / off）
```

服务日志里应能看到：

| 日志 | 含义 |
|---|---|
| `[yee] 4.2.0 defaults: ...` | 默认值已补上 |
| `[recall] patched ...` | 召回钩子已装上 |
| `[recall] rank check on: N ranks` | 各 rank 版本一致，召回广播启用 |
| `[recall] steps=... landed=... errors=0` | 召回统计，随请求增长 |
| `[yee-eb] armed geometry ...` | 热专家验证预算已启用 |

## 5. 开关与关闭

| 开关 | 默认 | 关闭 |
|---|---|---|
| `YEE`（`RECALL_ACCEL`）总开关 | 1 | `YEE=0`，整个 Yee 不加载 |
| `RECALL_ACCEL_NONGREEDY` 采样请求用召回 | 1 | 设 0 |
| `RECALL_ACCEL_QHASH` 提问哈希索引 | 1 | 设 0 |
| `YEE_KAD` 逐请求草稿数 | 1 | 设 0 |
| `YEE_EXPERT_BUDGET` 热专家验证预算 | 1，`_P=0.9` | 设 0；不重启时控制文件写 `armed=0` |
| 逐请求 conf（vcap） | adapt | 控制文件写 `{"mode": "legacy"}` 或删掉 |
| `VLLM_D_RECALL` / `_NONGREEDY` / `_KAD` | 1 | 设 0 |

- 只停复制：`yee pause on`，立即生效。
- 卸载：Mia 方案 `bash install.sh --uninstall --restart`（加 `--purge` 连安装目录一起删）；单机 `yee uninstall-hook && pip uninstall yee`。

全部环境变量见 `yee doc env`。

## 6. 仓库结构

| 路径 | 内容 |
|---|---|
| `yee/` | 主包：召回核心、各基座适配器、补丁、内置文档（`yee/docs/`） |
| `vllm_d_recall/` | vLLM DSpark 生产适配器（含 KAD） |
| `recall_accel/` | 旧包名兼容层 |
| `install/` | 一键安装脚本 `install.sh`、`yee_install.py`、安装说明和脚本测试 |
| `tests/` | 包测试（`pytest`） |
| `tools/` | 补丁清单构建工具 |
| `dist/` | 4.2.0 wheel |

## 7. 文档

装好后用 `yee doc` 列出全部主题，源文件在 [yee/docs/](yee/docs/)：

[quickstart](yee/docs/quickstart.md) · [heads（支持矩阵）](yee/docs/heads.md) · [sglang](yee/docs/sglang.md) · [vllm](yee/docs/vllm.md) · [kad](yee/docs/kad.md) · [expert-budget](yee/docs/expert-budget.md) · [tensorfold](yee/docs/tensorfold.md) · [cuteafd](yee/docs/cuteafd.md) · [mlx-serve](yee/docs/mlx-serve.md) · [llama-mtp](yee/docs/llama-mtp.md) · [patches](yee/docs/patches.md) · [env](yee/docs/env.md) · [troubleshooting](yee/docs/troubleshooting.md) · [changelog](yee/docs/changelog.md)

## 8. 验证

- 包内 `pytest`：538 通过、19 跳过、0 失败；干净 venv 里装 wheel 后 `python -m yee selftest` 通过。
- 安装脚本测试：Linux（WSL）16 项通过，用假的 ssh / docker / 启动脚本跑完整的安装、重启、失败回退、卸载流程。
- 在 4 台 DGX Spark 的真实方案目录上跑过 `--check`，并在测试目录真装、用现网镜像起一次性容器加载验证。

## 9. 历史版本

- [v2.2.0](https://github.com/Soulmate-Halo/HaloLab-Yee/releases/tag/v2.2.0)：改名 Yee，扩展到多个推理框架。
- [v1.2.0](https://github.com/Soulmate-Halo/HaloLab-Yee/releases/tag/v1.2.0)：首发 sglang-d-recall。

变更记录见 [CHANGELOG.md](CHANGELOG.md)。

## 许可

Apache-2.0，见 [LICENSE](LICENSE)。
