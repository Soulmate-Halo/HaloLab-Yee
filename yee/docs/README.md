# Yee 4.2.0 使用文档

本文档是 Yee 的总览：它是什么、怎么装、各个基座怎么用、去哪一页看细节。

Yee 是一个「跨请求召回」加速模块：把服务读过、写过的 token id 记下来，当新请求和历史内容重复时，直接把后续内容复制成草稿，交给服务正在用的加速头去验证。复制来的每一个 token 都由目标模型逐个验证，命中得多就少走几步。

## 0. 4.2.0 最短用法

4.2.0 起所有加速开关默认打开，不用设任何环境变量。显式设 0 可以单独关掉某一项；设 `YEE=0`（或 `RECALL_ACCEL=0`）整个 Yee 都不加载。

### Mia dsv41-4x-spark

在 head 机上：

```bash
bash install.sh --check      # 只检查，不改任何东西
bash install.sh --restart    # 装到 4 台机器，装完重启并做健康检查和冒烟
```

回退：

```bash
bash install.sh --uninstall --restart
```

`install.sh` 会自动找方案目录、4 台机器都装、在 env 文件末尾追加一段带标记的块（只写 `PYTHONPATH` 与 `YEE=1`）。其它机器上它自动走 `pip install` + `yee install-hook`。

### 其它 Python 推理服务

```bash
pip install yee-4.2.0-py3-none-any.whl && python -m yee install-hook
```

然后照常启动服务，不需要再手动加环境变量。

### 怎么确认生效

```bash
python -m yee config
```

它会打印每个开关的生效值与来源（env / default / off）。服务日志里应能看到：

| 日志 | 含义 |
|---|---|
| `[yee] 4.2.0 defaults:` | 4.2 的默认值已生效 |
| `[recall] patched ...` | 召回钩子已装上 |
| `[recall] rank check on` | 各 TP rank 装的是同一版本 Yee，召回广播与专家预算启用 |
| `[recall] steps=...` | 召回统计，数字随请求增长 |
| `[yee-eb] armed geometry ...` | 热专家验证预算已就绪 |

### 怎么关

- 单项：把对应开关设成 0（例如 `RECALL_ACCEL=0`、`YEE_KAD=0`、`VLLM_D_RECALL=0`、`YEE_EXPERT_BUDGET=0`），重启后该项关闭，其它项照常。
- 整体：设 `YEE=0`（或 `RECALL_ACCEL=0`），整个 Yee 不加载。

## 1. 名字与版本

| 项目 | 2.2.0 | 说明 |
|---|---|---|
| Python 包名 | `yee` | 2.2.0 的新包名 |
| 命令 | `yee` | 2.2.0 的新命令 |
| 旧包名 | `recall_accel` | 仍可导入，仍可用 |
| 旧命令 | `recall-accel` | 仍可执行，与 `yee` 等价 |
| 旧环境变量 | `RECALL_ACCEL_*` | 仍可读，优先级低于新的 `YEE_*` |

### 4.2.0：所有加速开关默认打开

- 召回总开关 `RECALL_ACCEL`（`YEE`）缺省 1，KAD 总开关 `YEE_KAD` 缺省 1，热专家验证预算 `YEE_EXPERT_BUDGET` / `_ARMED` 缺省 1、`YEE_EXPERT_BUDGET_P` 缺省 0.9；vLLM 的 `VLLM_D_RECALL` / `VLLM_D_RECALL_NONGREEDY` / `VLLM_D_RECALL_KAD` 缺省 1；采样批次用召回 `RECALL_ACCEL_NONGREEDY` 缺省 1。显式设 0 仍可单独关掉某一项，设 `YEE=0`（或 `RECALL_ACCEL=0`）整个 Yee 不加载。
- 召回容量 `RECALL_ACCEL_GIB` 缺省 0.25（原 0.0625），`RECALL_ACCEL_INDEX_DENSITY` 缺省 1（原 0.25）。
- 逐请求 conf（`mia-dsv41-vcap`）默认打开：`install.sh` 确认镜像里的 `verify_cap.py` 是补丁所基于的原件后，放入两个 vcap 文件，并在 head 的 state 目录写控制文件 `{"mode": "adapt"}`（已存在则保留；改成 `{"mode": "legacy"}` 或删掉即回到固定 conf）。
- Mia DSV41 镜像里按启动参数自动补默认值，只在没设时补：从 `--speculative-dspark-block-size N` 读出块长，补 `DSPARK_BLOCK_SIZE=N`；块长已知时补 `DSV41_VERIFY_CAP=conf:0.1` 与 `DSV41_VERIFY_CAP_STRIDE=N+1`；再补 `SGLANG_DSPARK_FOLDED_SAMPLING=2`。块长、EP、MoE 后端等启动参数一律不改。
- 热专家验证预算只在 Mia 的 `moe_b12x_next` 确定性路径上起作用（`DSV41_MOE_B12X_NEXT=1` 且 `DSV41_MOE_B12X_NEXT_DETERMINISTIC=1`，`EP_SIZE=1`）；别的 MoE 路径上自动不介入，其余加速照常。控制文件与统计文件缺省在 `/state`（不存在时在 `~/.cache/yee/`）。
- 新增跨 rank 一致性检查：各 TP rank 都装了同一版本 Yee 才启用召回广播和专家预算，否则所有 rank 一起关闭并打印 `[recall] rank check off: ...` / `[yee-eb] rank check off: ...`。
- 新增 `python -m yee config`：打印每个开关的生效值与来源（env / default / off）。
- 发布包里新增一键安装 `install.sh`（Mia dsv41-4x-spark 方案：自动找方案目录、4 台机器都装、在 env 文件末尾追加一段标记块，只写 `PYTHONPATH` 与 `YEE=1`；`--check` 只检查，`--restart` 装完重启并做健康检查和冒烟，失败自动回退，`--uninstall` 卸载；其它机器自动走 `pip install` + `yee install-hook`）。

### 3.0.0 内置的 vLLM 生产适配器与 KAD

- wheel 里同时带生产 vLLM DSpark 适配器 `vllm_d_recall`。它原来是独立版本线（1.0.0 → 1.1.0 → 1.2.0），3.0.0 起版本号与 Yee 一致。
- 它带 KAD（逐请求自适应草稿块宽）：以 `num_speculative_tokens = 12` 启动一次，之后每个请求按自己的接受情况在 5～12 之间选块宽，默认策略是 ladder，策略文件热加载。详见 [kad.md](kad.md)（`yee doc kad`）。
- Yee 2.4 的 M5 冷层以补丁 `mlx-serve-2.4` 入包，见 [mlx-serve.md](mlx-serve.md)。
- 3.1.0：KAD 扩到 vLLM 的全部草稿头（dspark、eagle、eagle3、mtp、draft_model、dflash、多模块 MTP 等），`K_max` 统一为 12，原理和默认 ladder 策略不变；各头的草稿侧路径与启动参数见 [kad.md](kad.md)（`yee doc kad`）。

### 4.0.0：召回感知验证预算与 STS 采集钩子——compact 7 少验的主因是召回刚命中时的预算滞后

compact 变长验证的 planner 用两步前的 confidence 算本步验证 token 总数，召回刚命中的请求这一步仍按低置信度分预算。4.0.0 在 `yee/integrations/sglang_dspark.py` 里加了两个只在 compact 下起作用的钩子：

- 召回感知预算（默认开，`YEE_RECALL_BUDGET=0` 关）：rank0 记下最近一步带召回计划的请求池行，`HostConfidenceBudgetPlanner.compute_budget` 算预算时把这些行当成确定接受。
- STS 采集（`YEE_STS_COLLECT_PATH=<路径前缀>`）：折叠提议下也能采置信度标签，跳过召回行，分片可直接给镜像里的 `python3 -m sglang.benchmark.dspark_sts_fit` 拟合。
- 4 台 DGX Spark 上实测（2026-11-18）：compact 7 + SPS 表 + STS 表 + 4.0.0，合计比现网 static 慢 1.9%，按真实流量加权慢 1.1%；new 并发 8 快 6.0%；比上一版的 compact 7 合计好 1.7%（copy 并发 8 从 −13.8% 收到 −5.5%，rewrite 并发 1 从 −12.3% 收到 −5.6%）。STS 拟合显示草稿头置信度本来就基本准（ECE ≤ 0.025），不是主因。
- static 模式下两个钩子都不触发，现网仍用 static。
### 3.9.0：解码单步耗时拆分工具——把每个解码步的墙钟、通信与计算分开看

新工具 `yee.tools.step_breakdown`：读同一次 torch profiler 采集的各 TP rank chrome trace，按 verify CUDA graph 把 trace 切成解码步，给出每 rank 每步的墙钟、GPU 忙与空闲、通信（RoCE 单发 all-reduce、RoCE all-gather、NCCL）与计算的分类耗时、暴露在关键路径上的通信时长（GPU 忙但没有计算 kernel 在跑的时间）、跨 rank 最慢 rank 与等待估计，结果写 `breakdown.json` 与 `breakdown.md`。

- 用法：`python -m yee.tools.step_breakdown <trace 文件或目录...> --out <目录>`。只读本地文件，不连服务、不改任何运行时参数。
- 4 台 DGX Spark 现网实测（块长 7 static，每请求验 8 行，2026-11-17）：并发 1 每步 32.4 ms，通信 5.4～6.0 ms，其中暴露在关键路径上的 2.4～2.6 ms（约 8%）；RoCE 单发 all-reduce 每步 90 次、单次均值 44 微秒，RoCE all-gather 每步 51 次、34～37 微秒，NCCL 每步 0.1～0.3 ms。并发 8 每步 86.6 ms，通信 17.2～18.2 ms，暴露 13.4～14.3 ms（约 14%），RoCE 单发 all-reduce 单次 132～139 微秒。
- 结论：解码里 2 MiB 以下的 TP 集合通信已经走 RoCE 单发内核，NCCL 每步只占 0.1～0.85 ms，调 NCCL 的算法和协议最多省约 1%；即使跨机通信降到 0，并发 1 也只快约 8%，达不到方案 B 并发 1 快 10% 以上的通过线，所以 3.9.0 不改任何运行时参数。
- 3.9.0 的引擎侧文件（`verify_cap.py`、`vcap_policy.py`）与 3.8.0 逐字节相同，现网无需重新部署。

### 3.8.0：vcap 选宽度器影子模式——先估"每步换验证宽度"能快多少，再决定改不改引擎

SGLang Mia（DeepSeek-V4.1）的验证矩形宽度固定为 [并发, 块长+1]，vcap 截掉的草稿行仍按死行计算，所以门槛怎么调都只有 ±1% 的空间。另一条路是每个验证宽度一套 CUDA graph、每步选"期望出 token ÷ 该宽度步时"最大的宽度（tideLM 的做法）。3.8.0 在 `mia-dsv41-vcap` 的 `vcap_policy.py` 里加了这个选择器的影子模式：只算、只写统计，引擎决策不变。

- 开法：控制文件 `/state/vcap_ctl.json` 写 `{"mode": "width", "tag": "<标签>"}`。verify_cap.py 把它当 legacy 走，每步的 k 与 `conf:0.1` 相同；删掉控制文件就回到不观测。
- 每步：rank0 只用此前已观测的信息，给候选宽度（默认 4 / 8，即块长 3 / 7）算期望出 token 数与预计步时，选比值最大的（好 3% 以上才换）；结果到 CPU 后按实际接受数结算"假如本步用这个宽度"。不超过现网宽度的部分是精确的。宽度 13 可以加，但对照实测它的估算偏高，默认不开。在现网 D 上跑压测矩阵：只在 4 / 8 里选估计合计快 3.8%，主要来自 new 类（并发 8 时 +16.5%）。
- 看结果：`python -m yee.tools.vcap_width_report vcap_stats.jsonl`，给出相对现网的提速和 `VERDICT: GO / STOP`（阈值 3%）。字段与限度见 [patches.md](patches.md) 和 [changelog.md](changelog.md)。

### 3.7.0：提问哈希索引（qhash）——同一个问题再问一次，直接从上次那份回答开始起草

请求正常结束时，把这次的回答按提示的哈希存起来（整段提示、末尾 512 / 128 / 32 个 token，共 4 级）。新请求开始时依次查这 4 级，命中就从第 1 个输出 token 起，用那份旧回答当草稿；输出与旧回答分叉后，用最近 4 个输出 token 在旧回答里重新对齐，对上就接着出草稿。草稿照常由目标模型逐 token 验证，命中错了只是草稿被拒。

- 开关：`RECALL_ACCEL_QHASH`（vLLM 是 `VLLM_D_RECALL_QHASH`），缺省 1；设 0 关闭，关闭后行为与 3.6.0 完全相同。
- 存储：disk 后端时每个键一个文件 `<召回目录>/qhash/<键>.i4`，多个进程共用目录时彼此立刻可见，重启后仍在，最多 8192 个键；memory / cuda 后端只在进程内存里。
- 限制：只有输出与旧回答逐 token 相同的部分能被接受；温度大于 0 时每次采样不同，命中后也只在回答一致的片段上加速；温度 0 且提示相同时效果最好。跨重启要用 disk 后端。
- 统计：`qhash`、`qhash_hits`、`qhash_realigns`、`qhash_puts`、`qhash_entries`。见 [sharing.md](sharing.md) 与 [changelog.md](changelog.md)。

### 3.6.0：多个进程共用一个磁盘记录目录（修 bug）

多副本共用一个记录目录时，3.5.x 只有第一个打开目录的进程写入，其它进程写不进去，也看不到它之后写的记录。3.6.0 起每个进程都写，每 2 秒读进别的进程写的记录（`RECALL_ACCEL_SHARED`，缺省 1；`0` 回到 3.5.x）。strata 补丁同步修好（`STRATA_RECALL_SHARED`）。见 [sharing.md](sharing.md)。

### 3.5.2：llama.cpp MTP 草稿头截断词表、测速工具

- 新补丁 `llama.cpp-mtp-vocab`（git-patch，基线 llama.cpp `a3b9c23`）：qwen35 / qwen35moe 的 MTP 草稿头只算词表前 N 个 token（`YEE_MTP_VOCAB_N=N`），logits 用 -1000 补回整个词表宽度。草稿都经主模型核对。mx4（Ryzen AI Max+ 395，Vulkan，Qwen3.6-27B Q4_K_M）上 k=3、N=131072 比原版快 5.0%。
- 新工具 `yee/tools/mtp_bench`：llama-server 投机解码测速与逐条输出比对、CPU/GPU 内存争用测试、生成 token 的词表覆盖率。
- 实测：MTP 草稿头放 CPU（`-md` 同一个 gguf、`-devd none -ngld 0`）在 mx4 上比放 GPU 慢 15%；GPU 忙时 CPU 读内存只剩 28% 的速度。CPU 与 GPU 共用内存的机器上草稿头留在 GPU。见 [llama-mtp.md](llama-mtp.md)。

### 3.5.1：strata 引擎召回补丁

- 新补丁 `strata-recall`（git-patch）：给 C++ MoE 引擎 strata 0.1.59（Niko1221/Strata 提交 `712fec6b`；也能打在同源的专家服务器版源码上）加跨请求召回。strata 的 token 循环在 C++ 进程里，Python 钩子碰不到，所以以源码补丁接入：召回库是 `yee.core` 的 C++ 移植（与 llama.cpp 补丁同一份），召回提议替换该窗口的 MTP 草稿，验证窗口逐个核对草稿。
- 开关：引擎参数 `--recall` 或环境变量 `RECALL_ACCEL=1`；不开时与上游逐位一致。接入单流窗口和 `--batch-mtp` 的批处理槽；`STRATA_BATCH_DF2`、`STRATA_PIPED_MTP` 的批处理窗口不接。
- 本机验证：补丁在 712fec6b 与专家服务器版源码上都能干净应用；改后的 `generate.cpp` 以 Linux 目标做 host 侧语法与类型检查通过；召回库单测 18 项、strata 适配层单测 25 项通过。还没有在 GPU 上编译和实测。

### 3.5.0：vLLM 门槛策略与非贪心召回、两个 SGLang Mia 补丁

- vLLM（`vllm_d_recall`）：KAD 策略新增 `mode=threshold`（每请求宽度 1～12）；`VLLM_D_RECALL_NONGREEDY=1` 让温度大于 0 的请求也用召回（被覆写的位置改成单点分布）；`VLLM_D_RECALL_SHADOW=1` 写影子统计。三项默认关，只做了单元测试。
- 新补丁 `mia-dsv41-vcap`（source-drop）：SGLang Mia 镜像 DeepSeek-V4.1 的逐请求 conf，控制文件 `/state/vcap_ctl.json`；没有控制文件时与 `DSV41_VERIFY_CAP=conf:0.1` 逐位一致。
- 新补丁 `mia-dsv41-image-budget`（source-drop）：vLLM 做 P、SGLang Mia 做 D 时，D 的图片 token 预算与 P 对齐，配 `XY_DSV41_IMAGE_BUDGET_RESERVE=1`。
- `yee patch` 的 source-drop 支持 `orig_sha256`（只替换指定原件）与 `remove_after`（拷完删掉过期的 .pyc）。
- 4 台 DGX Spark 上实测：DeepSeek-V4.1-Flash + SGLang Mia DSPARK 用块长 7 + `conf:0.1`；块长 12 的几种做法合计慢 5%～12%。

### 3.3.0：llama.cpp 召回补丁

- 新增补丁基座 `llama.cpp-recall`（git-patch），基线是 ggml-org/llama.cpp v0.9.0 commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8` 加上已打的 `llama.cpp-kad/0001-kad.patch`，**必须在 `llama.cpp-kad` 之后打**。
- 它把服务读过、写过的 token id 记下来，新请求与历史内容重复时把后续内容复制成草稿，交给 llama-server 正在用的加速头核对；带草稿模型时召回的提议替换草稿头的草稿，不带草稿模型时召回有提议就投机核对。记录可以放内存，也可以用 `--recall-dir` 落盘，重启后继续用。
- 开关：`--recall`、`LLAMA_ARG_RECALL=1`、`RECALL_ACCEL=1`、`YEE=1` 或 `YEE_RECALL=1`。生效日志：`[yee-recall] llama.cpp enabled: backend=memory dir=- read_only=0 gamma=12 head=none`。
- 详见 [patches.md](patches.md)。

### 3.2.0：KAD 覆盖全部基座

- KAD 从只支持 vLLM 扩到所有基座，`K_max` 仍统一为 12，策略文件格式不变，同一份 `kad_policy.json` 在所有基座通用。
- Python 基座新增适配器 `sglang_kad`、`trtllm_kad`、`lmdeploy_kad`、`mlx_lm_kad`、`hf_kad`，以及自动装配模块 `kad_auto`；`YEE_KAD=1` 时 `yee install-hook` 写的 `.pth` 在解释器启动时装配它，各引擎模块被导入时自动装上对应适配器。
- 源码基座新增三个补丁：`llama.cpp-kad`、`tensorfold-kad`（在 `tensorfold` 之后打）、`mlx-serve-kad`（在 `mlx-serve-2.4` 之后拷），见 [patches.md](patches.md)。
- vLLM 侧不变：仍用自己的开关 `VLLM_D_RECALL_KAD`。
- LMDeploy 和 cuteafd 是 native（引擎只有一个固定草稿数，KAD 省不了算力），只打印原因。
- 各基座的开关、启动参数、日志行和当前验证状态见 [kad.md](kad.md)；安装步骤见交付目录里的 `安装说明.md`。

## 2. 安装

在**服务实际使用的 Python 环境**里执行（通常是镜像里的那个 python）：

```bash
pip install --no-deps yee-4.2.0-py3-none-any.whl
```

只依赖 numpy，镜像里一般已经有；`--no-deps` 可以避免它去动服务环境里的其它包。

装完先自检：

```bash
yee doctor          # 版本，以及各适配器需要的符号、签名、源码标记；全部 PASS 才算通过
yee selftest        # 自带的测试组，不需要 pytest
python -m yee config   # 每个开关的生效值与来源（env / default / off）
```

Mia dsv41-4x-spark 用发布包里的一键安装脚本，见第 0 节。

## 3. 各基座一句话快速开始

- **Mia dsv41-4x-spark**：head 机上 `bash install.sh --check`，再 `bash install.sh --restart`；回退 `bash install.sh --uninstall --restart`。
- **SGLang**：`yee auto --apply` 识别引擎和加速头并写入 `.pth` 钩子，按它打印的环境变量重启服务即可（镜像自带 sitecustomize 时改用模板，见 [sglang.md](sglang.md)）。
- **vLLM**：`yee auto --apply` 同上，另外必须加 `VLLM_USE_V2_MODEL_RUNNER=1`（钩子装在 vLLM V2 runner 上），见 [vllm.md](vllm.md)。
- **TensorFold**：TensorFold 1.0.12 起自带 Recall，不需要打补丁，直接 `tensorfold serve MODEL --recall`；需要 priced 提议器兜底时再打 Yee 的 TensorFold 补丁，见 [tensorfold.md](tensorfold.md)。
- **cuteafd**（DeepSeek V4.1 native serve，Rust）：`yee patch apply cuteafd --src <源码根> --variant v2`，重新编译后用控制目录里的 `enabled` 文件开关，见 [cuteafd.md](cuteafd.md)。
- **mlx-serve**（M5 上的 xypd 版，Zig）：`yee patch apply mlx-serve --src <源码根>`，替换四个源文件后重新编译，见 [mlx-serve.md](mlx-serve.md)。
- **strata**（C++ MoE 引擎，Niko1221/Strata 0.1.59）：`yee patch apply strata-recall --src <仓库根>`，重新编译 CUDA 版 strata，启动时加 `RECALL_ACCEL=1`（或 `--recall`），见 [patches.md](patches.md)。
- **llama.cpp 的 MTP 草稿头**（qwen35 / qwen35moe，`--spec-type draft-mtp`）：`yee patch apply llama.cpp-mtp-vocab --src <仓库根>`，重编 llama-server，启动时加 `YEE_MTP_VOCAB_N=131072`，见 [llama-mtp.md](llama-mtp.md)。

## 4. 文档目录

| 文档 | 内容 |
|---|---|
| [index.md](index.md) | 全部 topic 的一页目录（`yee doc` 不带参数时列的就是它） |
| [quickstart.md](quickstart.md) | 从零到跑起来的完整步骤与验收方法 |
| [sglang.md](sglang.md) | SGLang 各加速头的接法、环境变量、sitecustomize 模板 |
| [vllm.md](vllm.md) | vLLM V2 runner 的接法与限制 |
| [tensorfold.md](tensorfold.md) | TensorFold 自带 Recall、Yee 补丁、引擎缺口 |
| [cuteafd.md](cuteafd.md) | cuteafd 补丁、开关、统计与回退 |
| [mlx-serve.md](mlx-serve.md) | mlx-serve 补丁、上线与回退 |
| [heads.md](heads.md) | 基座 × 加速头支持矩阵，含不支持项和原因 |
| [nongreedy.md](nongreedy.md) | 非贪心（采样）请求怎么用召回草稿，各基座怎么做 |
| [sharing.md](sharing.md) | 全局共享的含义与隐私风险 |
| [env.md](env.md) | 全部环境变量表：名称、默认值、作用，含 4.2 的自动默认值 |
| [patches.md](patches.md) | `yee patch` 用法、各补丁基线与回退方法 |
| [troubleshooting.md](troubleshooting.md) | 常见故障与定位步骤 |
| [changelog.md](changelog.md) | 各版本变化（4.2.0 起所有加速开关默认打开并新增一键安装脚本，4.1.1 起修热专家验证预算在 bfloat16 默认 dtype 下的自检误判，4.1.0 起含热专家验证预算，4.0.0 起含召回感知验证预算与折叠提议下的 STS 采集钩子，3.9.0 起含解码单步耗时拆分工具，3.8.0 起含 vcap 选宽度器影子模式，3.7.0 起含提问哈希索引 qhash，3.6.0 起多个进程共用磁盘记录目录，3.5.2 起含 llama.cpp MTP 草稿头补丁，3.5.1 起含 strata 召回补丁，3.5.0 起含 vLLM 门槛策略与两个 SGLang Mia 补丁，3.3.0 起含 llama.cpp 召回补丁，3.2.0 起含全部基座的 KAD 与三个新补丁） |
| [kad.md](kad.md) | 逐请求自适应草稿块宽（KAD），3.2.0 起覆盖全部基座 |
| [expert-budget.md](expert-budget.md) | 热专家验证预算（4.1.0 引入，4.2.0 默认开）：原理、开关、不重启改档、统计 |

## 5. 与 2.1 的兼容说明

2.2.0 是 2.1.0 的改名升级版，**升级不需要改任何部署脚本**：

1. **旧命令还在**：`recall-accel auto --apply`、`recall-accel doctor`、`recall-accel selftest` 等与 `yee` 的同名子命令完全等价。
2. **旧包名还在**：`import recall_accel` 仍然可用；`recall_accel.core`、`recall_accel.integrations` 等模块路径不变。
3. **旧环境变量还在**：`RECALL_ACCEL_*` 全部继续生效，语义和 2.1.0 完全一致。
4. **新增 `YEE_*` 前缀**：`YEE_*` 与同名的 `RECALL_ACCEL_*` 等价（`YEE_GIB` = `RECALL_ACCEL_GIB`，`YEE_NONGREEDY` = `RECALL_ACCEL_NONGREEDY`，以此类推），并且优先于 `RECALL_ACCEL_*` 被读取。
5. **更老的名字仍在更底层**：2.1.0 支持的 `SGLANG_D_RECALL_*` / `VLLM_D_RECALL_*` / `DSV41_RECALL_*` / `RECALL_*` 回退链在 2.2.0 里原样保留，顺序见 [env.md](env.md)。
6. **新增能力不改变旧行为**：TensorFold 补丁、cuteafd 补丁、mlx-serve 补丁都要**显式打**才会生效；不设 `TF_RECALL_AFTER_PRICED` 时 TensorFold 的行为与上游完全一致。KAD 与召回在 4.2.0 起默认开，需要旧行为就显式设 `YEE_KAD=0` / `RECALL_ACCEL=0`。
7. **记录目录默认没变**：仍按 tokenizer digest 分目录，`RECALL_ACCEL_DIR` 不设时是 `~/.cache/recall_accel/<digest16>`；要沿用 2.0/2.1 的旧记录目录，显式把 `RECALL_ACCEL_DIR` 指过去。

## 6. 回滚

按影响从小到大：

1. **只停复制**：`yee pause on`（写 `paused` 文件），立即生效，不用重启。
2. **关掉 KAD**：Python 基座设 `YEE_KAD=0`（或各自的开关）后重启；vLLM 设 `VLLM_D_RECALL_KAD=0` 后重启。
3. **关掉专家预算**：设 `YEE_EXPERT_BUDGET=0` 后重启，或把控制文件改成 `armed=0`。
4. **关闭钩子**：设 `YEE=0`（或 `RECALL_ACCEL=0`），或 `YEE_ADAPTER=off`，然后重启。
5. **彻底移除**：`yee uninstall-hook && pip uninstall yee`（用 sitecustomize 模板的，还要从 PYTHONPATH 里去掉模板目录），然后重启。
6. **回退补丁**：见 [patches.md](patches.md) 里每个补丁各自的回退方法。
