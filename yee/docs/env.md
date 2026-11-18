# 环境变量表

这一页列出 Yee 读取的全部环境变量，名称、默认值和作用都以源码为准（`yee/core.py`、`yee/integrations/*.py`、`yee/__main__.py`）。

## 4.2 自动默认值

4.2.0 起所有加速开关默认打开，**不用设任何环境变量**：

| 变量 | 4.2 缺省 | 作用 |
|---|---|---|
| `RECALL_ACCEL`（`YEE`） | `1` | 召回总开关 |
| `YEE_KAD` | `1` | KAD 总开关 |
| `YEE_EXPERT_BUDGET` | `1` | 热专家验证预算总开关 |
| `YEE_EXPERT_BUDGET_ARMED` | `1` | 热专家验证预算启动初值：是否 armed |
| `YEE_EXPERT_BUDGET_P` | `0.9` | 热专家验证预算启动初值：保留覆盖多少比例的路由质量 |
| `VLLM_D_RECALL` | `1` | vLLM 召回总开关 |
| `VLLM_D_RECALL_NONGREEDY` | `1` | vLLM 采样批次也用召回 |
| `VLLM_D_RECALL_KAD` | `1` | vLLM 的 KAD 开关 |
| `RECALL_ACCEL_NONGREEDY` | `1` | 采样批次也用召回 |

- **单项关闭**：把对应变量显式设成 `0`，该项关闭，其余照常。
- **整体关闭**：设 `YEE=0`（或 `RECALL_ACCEL=0`），整个 Yee 不加载。

另外两项容量类默认值也变了：

| 变量 | 4.2 缺省 | 原缺省 |
|---|---|---|
| `RECALL_ACCEL_GIB` | `0.25` | `0.0625` |
| `RECALL_ACCEL_INDEX_DENSITY` | `1` | `0.25` |

### Mia DSV41 镜像里自动补的默认值

在 SGLang Mia DSV41 镜像里（容器有 `/opt/dsv41/adapter`），Yee 会按服务的启动参数自动补默认值，**只在没设时补**，显式设过的值一律不动：

| 自动补的变量 | 值从哪来 | 补的值 |
|---|---|---|
| `DSPARK_BLOCK_SIZE` | 启动参数 `--speculative-dspark-block-size N` | `N` |
| `DSV41_VERIFY_CAP` | 块长已知时 | `conf:0.1` |
| `DSV41_VERIFY_CAP_STRIDE` | 块长已知时 | `N+1` |
| `SGLANG_DSPARK_FOLDED_SAMPLING` | 总是 | `2` |

块长、EP、MoE 后端等启动参数一律不改，用户怎么启动就怎么用。

### 逐请求 conf 的缺省模式

逐请求 conf（`mia-dsv41-vcap`）默认打开：`install.sh` 确认镜像里的 `verify_cap.py` 是补丁所基于的原件后，放入两个 vcap 文件，并在 head 的 state 目录写控制文件 `{"mode": "adapt"}`（已存在则保留；改成 `{"mode": "legacy"}` 或删掉即回到固定 conf）。

### 看当前生效值

```bash
python -m yee config
```

它打印每个开关的生效值与来源（`env` = 环境变量设的，`default` = 4.2 默认值，`off` = 关着）。

## 4.3 新增：诊断与提示

这两个变量只影响 Yee 自己的提示输出，不改变任何加速行为。

| 变量 | 默认值 | 作用 |
|---|---|---|
| `YEE_NOTICE` | `1` | `0` 关掉启动时的 `[yee] 4.3.0 plan:` 行与运行中的 `[yee] effect:` 行（`yee doctor` 的输出不受影响） |
| `YEE_EFFECT_S` | `600` | 运行中 `[yee] effect:` 行的间隔秒数；设小一点可以更频繁看到效果判定 |

`YEE_NOTICE=0` 时服务日志里只剩 `[recall] steps=` 这类原本就有的行，不会多打 `[yee]` 开头的内容。

## 新名字 YEE_*

2.2.0 起，任何 `YEE_XXX` 都等价于 `RECALL_ACCEL_XXX`，`YEE` 等价于 `RECALL_ACCEL`。`import yee` 时会把 `YEE_*` 复制成对应的 `RECALL_ACCEL_*`，但只在后者没设时才复制，所以两者都设时以 `RECALL_ACCEL_*` 为准。

例：`YEE=1 YEE_GIB=0.5` 等同于 `RECALL_ACCEL=1 RECALL_ACCEL_GIB=0.5`。

## 回退顺序

主表里的每个变量，按下面的顺序找第一个非空值（空字符串算没设）：

`RECALL_ACCEL_X` → `SGLANG_D_RECALL_X` → `VLLM_D_RECALL_X` → 更老的名字（`DSV41_RECALL_X` 或 `RECALL_X`，见最后一列）

## 主表

| 变量 | 默认值 | 作用 | 更老的名字 |
|---|---|---|---|
| `RECALL_ACCEL` | `1` | `1` / `true` / `yes` / `on` 打开召回钩子；4.2 起缺省 1，设 `YEE=0` 或 `RECALL_ACCEL=0` 整体关闭 | `DSV41_RECALL` |
| `RECALL_ACCEL_DIR` | `~/.cache/recall_accel/<分词器摘要前 16 位>` | 记录目录 | `DSV41_RECALL_DIR` |
| `RECALL_ACCEL_GIB` | `0.25` | 记录容量（GiB）；4.2 起缺省 0.25（原 0.0625） | `DSV41_RECALL_GIB` |
| `RECALL_ACCEL_BACKEND` | `disk` | `disk` / `memory` / `cuda`（`gpu` 同 `cuda`）；cuda 不可用时退回 memory | `RECALL_BACKEND` |
| `RECALL_ACCEL_DEVICE` | 空（cuda 后端用 `cuda:0`） | cuda 后端的设备 | `RECALL_DEVICE` |
| `RECALL_ACCEL_MIN_MATCH` | `16` | 最短匹配长度（token） | `DSV41_RECALL_MIN_MATCH` |
| `RECALL_ACCEL_PRIOR` | `0.8` | 新匹配的预期接受长度先验，按 gamma 的比例计（短窗口匹配减半） | `DSV41_RECALL_PRIOR` |
| `RECALL_ACCEL_READ_ONLY` | `0` | `1` 时只读已有记录、不写入 | `DSV41_RECALL_READ_ONLY` |
| `RECALL_ACCEL_SHARED` | `1` | 多个进程共用一个磁盘记录目录：每个进程都写，每 2 秒读进别的进程写的记录（3.6.0）；`0` 回到 3.5.x 的单 writer。vLLM 是 `VLLM_D_RECALL_SHARED` | `DSV41_RECALL_SHARED` |
| `RECALL_ACCEL_QHASH` | `1` | 提问哈希索引（3.7.0）：把回答按提示的哈希（整段提示、末尾 512 / 128 / 32 个 token）存起来，新请求命中就从第 1 个输出 token 起拿那份旧回答当草稿；`0` 关闭，关闭后行为与 3.6.0 完全相同。vLLM 是 `VLLM_D_RECALL_QHASH` | `DSV41_RECALL_QHASH` |
| `RECALL_ACCEL_MIN_FREE_GIB` | `4` | 磁盘剩余空间低于此值（GiB）时不再写入 | `DSV41_RECALL_MIN_FREE_GIB` |
| `RECALL_ACCEL_TOKENIZER_DIR` | 空（`yee status` 等命令找记录目录时用 `/models/DeepSeek-V4.1-Flash`） | 分词器目录，用来算记录的身份 | `DSV41_RECALL_TOKENIZER_DIR` |
| `RECALL_ACCEL_MODEL_NAME` | 空 | 模型名，写进记录身份 | `DSV41_RECALL_MODEL_NAME` |
| `RECALL_ACCEL_MIA_SITECUSTOMIZE` | `/opt/dsv41/adapter/sitecustomize.py` | sitecustomize 模板先执行的 Mia 原 sitecustomize 路径 | `DSV41_RECALL_MIA_SITECUSTOMIZE` |
| `RECALL_ACCEL_EVICT` | `heat` | 淘汰方式：`heat` / `fifo` | `RECALL_EVICT` |
| `RECALL_ACCEL_HOT_FRAC` | `0.5` | heat 模式下热区占容量的比例（0～0.95；0 等于 fifo） | `RECALL_HOT_FRAC` |
| `RECALL_ACCEL_DEDUP_RUN` | heat 模式 `80`，fifo 模式 `128` | 跳过重复片段的最短长度（下限 32） | `RECALL_DEDUP_RUN` |
| `RECALL_ACCEL_HEAT_HALFLIFE_IDS` | 冷区容量的 1/4（按 id 数） | 热度衰减半衰期 | `RECALL_HEAT_HALFLIFE_IDS` |
| `RECALL_ACCEL_HEAT_SEEN_WEIGHT` | `0.25` | 记录片段再次出现在请求里时累加的热度权重（0 = 不计） | `RECALL_HEAT_SEEN_WEIGHT` |
| `RECALL_ACCEL_PROMOTE_TOKENS` | `32` | 片段升入热区的热度阈值 | `RECALL_PROMOTE_TOKENS` |
| `RECALL_ACCEL_INDEX_DENSITY` | `1` | 每个存储 id 分到的哈希槽数（1/64～4）；4.2 起缺省 1（原 0.25） | `RECALL_INDEX_DENSITY` |

主表的 `SGLANG_D_RECALL_*` 和 `VLLM_D_RECALL_*` 名字就是把 `RECALL_ACCEL` 换成对应前缀，例如 `SGLANG_D_RECALL_DIR`、`VLLM_D_RECALL_GIB`。

## 不走回退链的变量

| 变量 | 默认值 | 作用 |
|---|---|---|
| `RECALL_ACCEL_ADAPTER` | `auto` | 只装指定适配器：`auto` / `dspark` / `dflash` / `eagle` / `ngram` / `vllm_spec` / `off` |
| `RECALL_ACCEL_NONGREEDY` | `1` | 采样批次是否接受召回草稿，见 [nongreedy.md](nongreedy.md) |
| `RECALL_ACCEL_WINDOWS` | `16,8,4` | 查找窗口长度（先长后短回退） |

这三个也可以写成 `YEE_ADAPTER`、`YEE_NONGREEDY`、`YEE_WINDOWS`。

## 基座侧变量（不是 Yee 读的）

| 变量 | 由谁读 | 作用 |
|---|---|---|
| `TF_RECALL_AFTER_PRICED` | 打了 Yee 补丁 0001 的 TensorFold | `1` 时 priced 提议器给不出草稿就改问 Recall；不设则与上游一致 |
| `TF_DSV41_LOOKUP_PLAN` | TensorFold | `1` 时 DeepSeek V4.1 用 lookup 计划（priced 提议器之一） |
| `VLLM_USE_V2_MODEL_RUNNER` | vLLM | 设为 `1` 才走 V2 runner，Yee 的 vLLM 钩子在 V2 里 |
| `DSV41_VCAP_CTL` | 打了 `mia-dsv41-vcap` 的 SGLang Mia 镜像 | 逐请求 conf 的控制文件路径，缺省 `/state/vcap_ctl.json`；`install.sh` 放 vcap 时写入 `{"mode": "adapt"}`；文件不存在时与 `DSV41_VERIFY_CAP=conf:T` 相同 |
| `XY_DSV41_IMAGE_BUDGET_RESERVE` | 打了 `mia-dsv41-image-budget` 的 SGLang Mia 镜像 | `1` 时 D 的图片 token 预算减 1，与 vLLM 的 P 一致；缺省 `0` 与原版一致 |

## KAD（逐请求自适应草稿数）

KAD 的变量由 `yee.kad` 核心和各基座适配器直接读，**不走上面那条 `RECALL_ACCEL_*` 回退链**，也不被 `RECALL_ACCEL_ADAPTER` 收窄。4.2 起 KAD 总开关缺省打开；关掉时策略文件的格式在所有基座通用，见 [kad.md](kad.md)。

### 通用（Python 基座与原生基座都认）

| 变量 | 默认值 | 作用 | 适用基座 |
|---|---|---|---|
| `YEE_KAD` | `1` | KAD 总开关，取 `1` / `true` / `yes` / `on` 打开；4.2 起缺省 1，设 0 单独关闭 | 全部（SGLang、TensorRT-LLM、mlx-lm、transformers、TensorFold、mlx-serve、llama.cpp） |
| `YEE_KAD_DIR` | `~/.cache/yee/kad` | KAD 的默认目录，策略文件和统计文件的默认位置都在它下面 | 全部 |
| `YEE_KAD_POLICY` | `<YEE_KAD_DIR>/kad_policy.json` | 策略文件路径（JSON，格式各基座通用，热加载） | 全部 |
| `YEE_KAD_STATS_FILE` | `<YEE_KAD_DIR>/kad_stats_<基座名>.json` | 统计文件路径，原子写 | 全部 |
| `YEE_KAD_LOG_S` | `30` | 日志与写统计的间隔（秒） | 全部 |
| `YEE_KAD_RELOAD_S` | `1.0` | 策略文件 stat 的最小间隔（秒） | 全部 |
| `YEE_KAD_RELOAD_EVERY` | `64` | 多 rank 时每多少步同步一次策略（`step % 值 == 0` 时由 rank0 读文件再广播） | SGLang、TensorRT-LLM |

### vLLM（沿用自己的前缀）

| 变量 | 默认值 | 作用 | 适用基座 |
|---|---|---|---|
| `VLLM_D_RECALL_KAD` | `1` | KAD 开关；还要 `VLLM_D_RECALL=1`（总开关）；4.2 起缺省 1，设 0 单独关闭 | vLLM |
| `VLLM_D_RECALL_KAD_POLICY` | `<VLLM_D_RECALL_DIR>/kad_policy.json` | 策略文件 | vLLM |
| `VLLM_D_RECALL_KAD_STATS_FILE` | `<VLLM_D_RECALL_DIR>/kad_stats.json` | 调度器侧统计 | vLLM |
| `VLLM_D_RECALL_KAD_LOG_S` | `30` | 日志与统计间隔（秒） | vLLM |
| `VLLM_D_RECALL_KAD_RELOAD_S` | `1.0` | 策略文件 stat 最小间隔（秒） | vLLM |
| `VLLM_D_RECALL_NONGREEDY` | `1` | 3.5.0：`1` 时温度大于 0 的行也接受召回草稿，被覆写的位置把草稿分布改成单点分布；4.2 起缺省 1，设 0 单独关闭 | vLLM |
| `VLLM_D_RECALL_SHADOW` | `0` | 3.5.0：`1` 时写影子统计（只记录、不改行为） | vLLM |
| `VLLM_D_RECALL_SHADOW_DIR` | 临时目录下的 `vllm_d_recall_shadow` | 影子统计 JSONL 的目录 | vLLM |
| `VLLM_D_RECALL_SHADOW_TOPK_RATE` | `0.01` | 影子统计里 topk 记录的抽样率 | vLLM |

### TensorFold

| 变量 | 默认值 | 作用 | 适用基座 |
|---|---|---|---|
| `TF_KAD` | `0` | KAD 开关，取 `1` / `true` / `yes` / `on`；也可只设 `YEE_KAD=1` | TensorFold |
| `TF_KAD_POLICY` | `<YEE_KAD_DIR>/kad_policy.json` | 策略文件；未设时用 `YEE_KAD_POLICY` | TensorFold |
| `TF_KAD_STATS_FILE` | `<YEE_KAD_DIR>/kad_stats_tensorfold.json` | 统计文件；未设时用 `YEE_KAD_STATS_FILE` | TensorFold |
| `TF_KAD_LOG_S` | `30` | 日志与写统计的间隔（秒）；未设时用 `YEE_KAD_LOG_S` | TensorFold |
| `TF_KAD_RELOAD_S` | `1.0` | 策略热加载的最小间隔（秒）；未设时用 `YEE_KAD_RELOAD_S` | TensorFold |

### mlx-serve

| 变量 | 默认值 | 作用 | 适用基座 |
|---|---|---|---|
| `MLX_SERVE_KAD` | `0` | KAD 开关，取 `1` / `true` / `yes` / `on`；也可只设 `YEE_KAD=1` | mlx-serve |
| `MLX_SERVE_KAD_POLICY` | `<YEE_KAD_DIR>/kad_policy.json` | 策略文件；未设时用 `YEE_KAD_POLICY` | mlx-serve |
| `MLX_SERVE_KAD_STATS_FILE` | `<YEE_KAD_DIR>/kad_stats_mlx-serve.json` | 统计文件；未设时用 `YEE_KAD_STATS_FILE` | mlx-serve |
| `MLX_SERVE_KAD_LOG_S` | `30` | 日志与写统计的间隔（秒）；未设时用 `YEE_KAD_LOG_S` | mlx-serve |
| `MLX_SERVE_KAD_RELOAD_S` | `1.0` | 策略热加载的最小间隔（秒）；未设时用 `YEE_KAD_RELOAD_S` | mlx-serve |

### llama.cpp（命令行参数优先，参数名带 `LLAMA_ARG_` 前缀）

| 变量 | 默认值 | 作用 | 适用基座 |
|---|---|---|---|
| `LLAMA_ARG_KAD` | `0` | KAD 开关，等价于命令行 `--kad`；也可只设 `YEE_KAD=1` | llama.cpp |
| `LLAMA_ARG_KAD_POLICY` | `<YEE_KAD_DIR>/kad_policy.json` | 策略文件，等价于命令行 `--kad-policy FNAME`；未设时用 `YEE_KAD_POLICY` | llama.cpp |

llama.cpp 的策略与统计路径也可用 `YEE_KAD_POLICY` / `YEE_KAD_STATS_FILE`（默认 `<YEE_KAD_DIR>/kad_stats_llama.cpp.json`）、`YEE_KAD_LOG_S`、`YEE_KAD_RELOAD_S`。

### strata（命令行参数优先）

| 变量 / 参数 | 默认值 | 作用 | 适用基座 |
|---|---|---|---|
| `RECALL_ACCEL` / `YEE_RECALL`，参数 `--recall` | `1` | 召回开关（4.2 起缺省 1）；`--no-recall` 压过环境变量 | strata |
| `RECALL_ACCEL_DIR`，参数 `--recall-dir` | 空（内存记录） | 记录目录，多个进程可共用 | strata |
| `RECALL_ACCEL_GIB`，参数 `--recall-gib` | `0.0625` | 记录容量（GiB） | strata |
| `RECALL_ACCEL_READ_ONLY`，参数 `--recall-read-only` | `0` | 只用旧记录，不写新记录 | strata |
| `STRATA_RECALL_SHARED` | `1` | 多副本共用 `--recall-dir`：每组副本都写，每 2 秒读进其它组写的记录（3.6.0）；`0` 回到 3.5.x 的单 writer | strata |
| `RECALL_ACCEL_MIN_MATCH` | `16` | 最短匹配（token） | strata |
| `YEE_RECALL_STATS_FILE` | `~/.cache/yee/recall/recall_stats_strata.json` | 统计文件 | strata |
| `YEE_RECALL_LOG_S` | `30` | 周期日志间隔（秒） | strata |

### llama.cpp MTP 草稿头（3.5.2，需补丁 llama.cpp-mtp-vocab）

| 变量 | 默认值 | 作用 | 适用基座 |
|---|---|---|---|
| `YEE_MTP_VOCAB_N` | 空（不截断） | MTP 草稿头只算 token id 0～N-1 的 logits；0 或不小于词表大小时不截断 | llama.cpp（qwen35 / qwen35moe） |

### LMDeploy 与 cuteafd

两者都是 native：KAD 不介入，只按 12 的草稿数原生跑，没有自己的 KAD 环境变量。LMDeploy 仍读 `YEE_KAD`（autoload 装上后只打印一行 native 原因），cuteafd 没有 KAD 开关。

### 热专家验证预算（expert_budget，4.1.0 引入，4.2.0 默认开）

只在 Mia 的 `moe_b12x_next` 确定性路径上起作用（`DSV41_MOE_B12X_NEXT=1`、`DSV41_MOE_B12X_NEXT_DETERMINISTIC=1`、`EP_SIZE=1`）；别的 MoE 路径自动不介入，其余加速照常。详见 [expert-budget.md](expert-budget.md)。

| 变量 | 默认值 | 作用 |
|---|---|---|
| `YEE_EXPERT_BUDGET` | `1` | 总开关；设 0 关闭 |
| `YEE_EXPERT_BUDGET_ARMED` | `1` | 启动初值：是否 armed；设 0 启动即不启用 |
| `YEE_EXPERT_BUDGET_P` | `0.9` | 启动初值：保留覆盖多少比例的路由质量 |
| `YEE_EXPERT_BUDGET_KCAP` | `4096` | 启动初值：最多保留多少个专家 |
| `YEE_EXPERT_BUDGET_E` | `384` | 打补丁的几何专家数 |
| `YEE_EXPERT_BUDGET_MAXM` | `512` | 行数上限，超过走原路径 |
| `YEE_EXPERT_BUDGET_CTRL` | `/state/yee_expert_budget.json` | 控制文件（每个节点各一份） |
| `YEE_EXPERT_BUDGET_STATS` | `/state/yee_expert_budget_stats.jsonl` | 统计文件（全局 rank 0 写） |
| `YEE_EXPERT_BUDGET_STATS_S` | `10` | 统计间隔（秒） |

`/state` 不存在时，控制文件与统计文件落在 `~/.cache/yee/` 下。

## 补丁基座侧的默认值

下面几张表里的变量由各基座自己的源码读（strata 的 C++ 召回库、llama.cpp 的命令行参数、mlx-serve 的 `env.zig`），它们的缺省值由补丁里的代码决定，不随 Yee 的 Python 侧默认值变化。4.2 起各基座自身的加速开关（`RECALL_ACCEL` / `--recall`、`MLX_SERVE_D_RECALL`）仍要按各基座文档里的方式打开。

## selftest 的处理

`yee selftest` 在子进程里跑测试前会清掉 `YEE`、`YEE_*`、`RECALL_ACCEL*`、`SGLANG_D_RECALL*`、`VLLM_D_RECALL*`、`DSV41_RECALL*`，避免线上配置影响测试结果。
