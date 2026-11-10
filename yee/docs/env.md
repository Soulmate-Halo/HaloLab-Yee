# 环境变量表

这一页列出 Yee 读取的全部环境变量，名称、默认值和作用都以源码为准（`yee/core.py`、`yee/integrations/*.py`、`yee/__main__.py`）。

## 新名字 YEE_*

2.2.0 起，任何 `YEE_XXX` 都等价于 `RECALL_ACCEL_XXX`，`YEE` 等价于 `RECALL_ACCEL`。`import yee` 时会把 `YEE_*` 复制成对应的 `RECALL_ACCEL_*`，但只在后者没设时才复制，所以两者都设时以 `RECALL_ACCEL_*` 为准。

例：`YEE=1 YEE_GIB=0.5` 等同于 `RECALL_ACCEL=1 RECALL_ACCEL_GIB=0.5`。

## 回退顺序

主表里的每个变量，按下面的顺序找第一个非空值（空字符串算没设）：

`RECALL_ACCEL_X` → `SGLANG_D_RECALL_X` → `VLLM_D_RECALL_X` → 更老的名字（`DSV41_RECALL_X` 或 `RECALL_X`，见最后一列）

## 主表

| 变量 | 默认值 | 作用 | 更老的名字 |
|---|---|---|---|
| `RECALL_ACCEL` | `0` | `1` / `true` / `yes` / `on` 打开召回钩子 | `DSV41_RECALL` |
| `RECALL_ACCEL_DIR` | `~/.cache/recall_accel/<分词器摘要前 16 位>` | 记录目录 | `DSV41_RECALL_DIR` |
| `RECALL_ACCEL_GIB` | `0.0625` | 记录容量（GiB） | `DSV41_RECALL_GIB` |
| `RECALL_ACCEL_BACKEND` | `disk` | `disk` / `memory` / `cuda`（`gpu` 同 `cuda`）；cuda 不可用时退回 memory | `RECALL_BACKEND` |
| `RECALL_ACCEL_DEVICE` | 空（cuda 后端用 `cuda:0`） | cuda 后端的设备 | `RECALL_DEVICE` |
| `RECALL_ACCEL_MIN_MATCH` | `16` | 最短匹配长度（token） | `DSV41_RECALL_MIN_MATCH` |
| `RECALL_ACCEL_PRIOR` | `0.8` | 新匹配的预期接受长度先验，按 gamma 的比例计（短窗口匹配减半） | `DSV41_RECALL_PRIOR` |
| `RECALL_ACCEL_READ_ONLY` | `0` | `1` 时只读已有记录、不写入 | `DSV41_RECALL_READ_ONLY` |
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
| `RECALL_ACCEL_INDEX_DENSITY` | `0.25` | 每个存储 id 分到的哈希槽数（1/64～4） | `RECALL_INDEX_DENSITY` |

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

## selftest 的处理

`yee selftest` 在子进程里跑测试前会清掉 `YEE`、`YEE_*`、`RECALL_ACCEL*`、`SGLANG_D_RECALL*`、`VLLM_D_RECALL*`、`DSV41_RECALL*`，避免线上配置影响测试结果。
