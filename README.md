# Halo实验室 Yee 加速模块

SGLang 推理服务 **D 端（decode 阶段）** 的投机草稿加速模块。把服务读过的提示词和写过的回复按 **token ID** 存下来，新请求生成时如果最近若干个 token 与历史完全一致，就把历史中紧随其后的内容作为草稿交给主模型验证。验证通过的 token 直接采用，未通过的丢弃照常生成——**输出与不装时逐字相同**。

首发收录版本：**sglang-d-recall 1.2.0**（Python 包名 `sglang-d-recall`，导入名 `sglang_d_recall`）。

## 适用范围（先读这一节）

本模块只服务一种场景，且必须同时满足：

- 推理引擎是 **SGLang**
- 服务已启用 **DSpark 投机解码**（`--speculative-algorithm` 为 DSpark）

以下组合 **`doctor` 会 FAIL**，本仓库没有对应适配层，遇到请直接停下，不要改 SGLang 源码硬套：

- 未启用 DSpark 的普通 SGLang
- EAGLE / MTP / DFlash
- vLLM
- TensorFlow

功能边界：**只对 temperature=0 的贪心解码生效**。整批都是 temperature=0 才复制草稿；含采样请求的批次自动跳过。此外**只对重复内容有效**——重跑同一任务、代码改写时大段照抄、引用上下文原文等场景有收益，全新内容没有可复制历史，速度与不装时持平。

## 实测效果

DeepSeek V4.1 Flash，4×DGX Spark TP4，DSpark 投机解码，验证窗口 8，GSM8K，temperature=0：

| 场景 | 单路吞吐 | 相对首次 |
|---|---|---|
| 同一批题，第一次请求 | 111.85 tok/s | 基准 |
| 同一批题，第二次请求 | 180.09 tok/s | **+61%** |
| 8 路并发，第二遍合计 | 283.7 tok/s | — |

## 工作原理

模块由三层组成，全部在 `sglang_d_recall/` 下：

1. **记录层 `core.py`（约 2000 行，纯 numpy，引擎无关）**
   按 chunk 文件（每个最多 1,048,576 个 int32 token ID，小端）顺序追加记录。每条请求追加两段内容：提示词中此前任何请求都没读过的部分，以及请求结束时的回复。已存过的连续 128 个以上 token 不重复写入，因此同一请求发第二次不会增加任何记录。查找时用流末尾 16 个以上 token 直接映射哈希查续写；是否提供草稿取决于命中率先验与近期收益，匹配长度按 16–31 / 32–63 / 64+ 三档分别估计，连续四轮无收益的流暂停供应，试探性供应随连续未命中逐步变稀。**流自身上下文优先查询，平局时胜出。**

2. **存储后端（三选一，同一套记录逻辑）**

   | 后端 | 存放位置 | 重启后 | 适用场景 |
   |---|---|---|---|
   | `disk`（默认） | 记录目录下 4 MiB 一块的文件 | 保留 | 想跨重启保留历史；磁盘余量低于 `SGLANG_D_RECALL_MIN_FREE_GIB`（默认 4 GiB）停止写入 |
   | `memory` | 进程内 numpy int32 块 | 清空 | DGX Spark 等统一内存机器（现网使用） |
   | `cuda` | GPU torch int32 张量 | 清空 | 独立显存机器；需要 torch |

   三者语义完全一致，`tests/test_backends_equiv.py` 用随机序列逐一比对。

3. **SGLang 集成层 `integrations/sglang_dspark.py`**
   向 SGLang DSpark spec-v2 的 decode 路径装钩子，把记录层给出的候选草稿替换进去，并带上 DeepSeek V4 草稿置信度按实际块长计算的修正。`autoload.py` 提供解释器启动即装钩子的入口（`.pth` 文件机制），仅在 `SGLANG_D_RECALL=1` 时生效；任何安装失败只打印 `[sglang-d-recall] install failed`，不影响服务启动。

**正确性保证**：每一个被召回的 token 都要经过目标模型验证，因此开启、关闭或只读模式下回复完全一致。这一条是全部设计的前提。

### 容量与淘汰

容量 token 数 = `GIB × 2^30 / 4`，占用内存约 **2 × GIB**（token ID 1 份，哈希索引 1 份）。

| GIB | 容量 token | 内存约 |
|---|---|---|
| 0.0625（默认） | 16,777,216 | 128 MiB |
| 0.25 | 67,108,864 | 512 MiB |
| 1 | 268,435,456 | 2 GiB |

「历史还能找回 80% 的间隔」指同一内容两次出现之间服务一共写入多少 token：间隔为容量 5% 时约找回 81%，25% 时约 37%，50% 时约 13%，写满即为 0。**占用远没满时调大容量没有收益**，先看 `status`。统一内存机器上这块内存与 KV 缓存共享，建议不超过 1 GiB。

淘汰策略 `SGLANG_D_RECALL_EVICT` 默认 `heat`（冷热两级）：

- 每 256 token 一段统计热度 = 草稿被主模型采用的 token 数 + 写入时因已存过而跳过的重复 token × 0.25，按半衰期（默认冷区容量的 1/4，按累计写入 token 计）衰减。
- 热度达 32 的段连同前 16 个、后最多 64 个 token 复制进热区；热区满时淘汰衰减后热度最低的段；冷热都能匹配时，热区匹配长度不短于冷区就用热区。
- 热区存的是原样复制的 token，照常过主模型验证，输出仍然逐字不变。
- `fifo` 即 1.0.0 的单一环形存储行为。`disk` 后端的热区只在内存，重启后为空，冷区照常从磁盘恢复。

索引密度 `SGLANG_D_RECALL_INDEX_DENSITY`（1.2.0 新增，默认 `0.25`，范围 1/64–4）：每个存储 token 配多少个索引槽。默认每 4 个 token 一个槽，池子写满后大部分旧窗口查不到；设为 `1` 后写满时 16-gram 找回率从 12.8% 提升到 31.75%（离线测试），池子没写满时无影响。

离线模拟数据（合成 token，总容量 2^20，写入 8 倍容量，三套负载 × 三套配置 × 四档密度）见 [`docs/DENSITY_RESULTS.md`](docs/DENSITY_RESULTS.md)。

## 目录结构

```
HaloLab-Yee/
├── README.md            本文件
├── CHANGELOG.md         版本变更记录
├── LICENSE              Apache-2.0
├── pyproject.toml       包与依赖声明（依赖仅 numpy）
├── dist/                构建好的 wheel
│   └── sglang_d_recall-1.2.0-py3-none-any.whl
├── sglang_d_recall/     模块源码
│   ├── core.py          记录层与三种存储后端（引擎无关，纯 numpy）
│   ├── autoload.py      解释器启动时装钩子的入口
│   ├── __main__.py      CLI 入口
│   └── integrations/
│       ├── sglang_dspark.py          SGLang DSpark 钩子
│       └── sitecustomize_template.py sitecustomize 模板（容器免改镜像安装用）
├── tests/               测试（5 个文件，覆盖核心逻辑、淘汰、密度、后端等价性）
└── docs/
    ├── SGLang-D端加速模块-sglang-d-recall-部署文档.md   完整部署文档（权威）
    ├── package-README.md                            上游包说明
    └── DENSITY_RESULTS.md                           索引密度离线模拟结果
```

## 安装

详细步骤（含容器免改镜像、多机 TP 每 rank 配置、验收清单）见 [`docs/SGLang-D端加速模块-sglang-d-recall-部署文档.md`](docs/SGLang-D端加速模块-sglang-d-recall-部署文档.md)。要点：

```bash
# 方式 A：容器/镜像不想改动（推荐，现网 ysy37–40 使用）
mkdir -p /opt/sglang-d-recall
python3 -m pip install --no-deps --target /opt/sglang-d-recall \
    dist/sglang_d_recall-1.2.0-py3-none-any.whl
cp /opt/sglang-d-recall/sglang_d_recall/integrations/sitecustomize_template.py \
    /opt/sglang-d-recall/sitecustomize.py

# 方式 B：能直接改 SGLang 所在的 Python 环境
pip install --no-deps dist/sglang_d_recall-1.2.0-py3-none-any.whl
python -m sglang_d_recall install-hook
```

## 快速开始

安装后先跑两个检查，**都在运行 SGLang 服务的那个 Python 解释器里执行**：

```bash
python -m sglang_d_recall doctor      # 最后一行 doctor PASS 才继续；FAIL 会列出缺失模块
python -m sglang_d_recall selftest    # 最后一行 selftest PASS
```

最小可用配置（容器部署时每个 rank 的容器都要加，`PYTHONPATH` 里本模块目录排最前）：

```bash
PYTHONPATH=/opt/sglang-d-recall:<容器原来的 PYTHONPATH>
SGLANG_D_RECALL=1
SGLANG_D_RECALL_BACKEND=memory
SGLANG_D_RECALL_GIB=0.0625
SGLANG_D_RECALL_DIR=/root/.cache/sglang_d_recall/dsv41-flash
SGLANG_D_RECALL_TOKENIZER_DIR=/models/DeepSeek-V4.1-Flash
SGLANG_D_RECALL_MODEL_NAME=deepseek-v4.1-flash
```

真机用过的密度配置：`SGLANG_D_RECALL_GIB=0.25` 且 `SGLANG_D_RECALL_INDEX_DENSITY=1`（约 1.01 GiB，容量 6710 万 token）。

**验证窗口（块长）**：引擎一次验证 `DSPARK_BLOCK_SIZE + 1` 个 token，这也是复制一步最多能拿到的 token 数。DeepSeek V4.1 Flash + DSpark + 4×Spark 实测甜点为 `DSPARK_BLOCK_SIZE=7`（窗口 8，现网值）或 `6`（窗口 7，新内容多时更稳）；重复内容超过约 54% 选 7，否则选 6。块长不是 5 时必须开着本模块（它带 DeepSeek V4 草稿置信度按实际块长计算的修正）；关闭模块前先把块长改回 5，否则启动报 `mat1 and mat2 shapes cannot be multiplied`。换模型/草稿模型/硬件需重新测甜点。

## 命令入口

```bash
python -m sglang_d_recall doctor            # 环境自检
python -m sglang_d_recall selftest          # 功能自检
python -m sglang_d_recall install-hook      # 在 site-packages 写 .pth，启动即装钩子
python -m sglang_d_recall uninstall-hook    # 移除钩子
python -m sglang_d_recall status            # 打印 stats.json（提供次数、命中、记录占用）
python -m sglang_d_recall pause on|off      # 不重启暂停/恢复复制
python -m sglang_d_recall clear             # 清空记录
```

## 环境变量

新名字优先，旧名字仍可回退读取，空值视为未设置。完整说明见部署文档第 8 节。

| 变量 | 旧名 | 默认 | 含义 |
|---|---|---|---|
| `SGLANG_D_RECALL` | `DSV41_RECALL` | `0` | `1/true/yes/on` 开启 |
| `SGLANG_D_RECALL_DIR` | `DSV41_RECALL_DIR` | `~/.cache/sglang_d_recall/<tokenizer 摘要前 16 位>` | 记录目录；memory/cuda 下只放控制文件 |
| `SGLANG_D_RECALL_GIB` | `DSV41_RECALL_GIB` | `0.0625` | 记录容量（GiB） |
| `SGLANG_D_RECALL_BACKEND` | `RECALL_BACKEND` | `disk` | `disk` / `memory` / `cuda` |
| `SGLANG_D_RECALL_DEVICE` | `RECALL_DEVICE` | `cuda:0` | cuda 后端设备 |
| `SGLANG_D_RECALL_MIN_MATCH` | `DSV41_RECALL_MIN_MATCH` | `16` | 最短匹配长度 |
| `SGLANG_D_RECALL_PRIOR` | `DSV41_RECALL_PRIOR` | `0.8` | 命中率先验 |
| `SGLANG_D_RECALL_READ_ONLY` | `DSV41_RECALL_READ_ONLY` | `0` | 只复制不写新记录 |
| `SGLANG_D_RECALL_MIN_FREE_GIB` | `DSV41_RECALL_MIN_FREE_GIB` | `4` | disk 后端磁盘余量下限 |
| `SGLANG_D_RECALL_TOKENIZER_DIR` | `DSV41_RECALL_TOKENIZER_DIR` | `/models/DeepSeek-V4.1-Flash` | 计算 tokenizer 身份的目录 |
| `SGLANG_D_RECALL_MODEL_NAME` | `DSV41_RECALL_MODEL_NAME` | `deepseek-v4.1-flash` | 记录身份里的模型名 |
| `SGLANG_D_RECALL_MIA_SITECUSTOMIZE` | `DSV41_RECALL_MIA_SITECUSTOMIZE` | `/opt/dsv41/adapter/sitecustomize.py` | 先执行的原有 sitecustomize |
| `SGLANG_D_RECALL_EVICT` | `RECALL_EVICT` | `heat` | `heat` 冷热两级 / `fifo` 与 1.0.0 相同 |
| `SGLANG_D_RECALL_HOT_FRAC` | `RECALL_HOT_FRAC` | `0.5` | 热区占总容量比例（0 = fifo） |
| `SGLANG_D_RECALL_DEDUP_RUN` | `RECALL_DEDUP_RUN` | heat `80` / fifo `128` | 跳过已存内容的最短连续长度（最小 32） |
| `SGLANG_D_RECALL_PROMOTE_TOKENS` | `RECALL_PROMOTE_TOKENS` | `32` | 进热区的热度门槛 |
| `SGLANG_D_RECALL_HEAT_HALFLIFE_IDS` | `RECALL_HEAT_HALFLIFE_IDS` | 冷区容量 / 4 | 热度半衰期（累计写入 token 数） |
| `SGLANG_D_RECALL_HEAT_SEEN_WEIGHT` | `RECALL_HEAT_SEEN_WEIGHT` | `0.25` | 去重跳过的每个 token 记多少热度 |
| `SGLANG_D_RECALL_INDEX_DENSITY` | `RECALL_INDEX_DENSITY` | `0.25` | 索引密度（1.2.0 新增，范围 1/64–4） |

## 测试

```bash
pip install pytest numpy
python -m pytest tests/ -q
```

覆盖：记录核心逻辑与命中策略（`test_recall_core.py`）、SGLang 钩子行为（`test_recall_patch.py`）、冷热淘汰（`test_heat.py`）、索引密度（`test_density.py`）、三种存储后端等价性（`test_backends_equiv.py`）。

## 真机验证情况

- 已在 **Mia 镜像**（`dsv41-4x-spark` 一套）的 DSpark 场景实测，现网 ysy37–40 运行本版本。
- **1017 镜像不是同一套**，本版本未在 1017 镜像上重测。
- 因此安装前必须先跑 `doctor`，最后一行是 PASS 才继续。
- 部署文档示例密度为 `SGLANG_D_RECALL_GIB=0.0625`；真机用过的配置是 `0.25` 且 `SGLANG_D_RECALL_INDEX_DENSITY=1`。两套都保留，部署文档原文未改。

## wheel 校验值

```
文件    sglang_d_recall-1.2.0-py3-none-any.whl
大小    42999 字节
sha256  0DE37D6C03F7DA5FE5A1C6F21A767155F1E4C87D66392B84EC3A7527409B806A
```

## 已知边界

- 只支持 SGLang + DSpark 一种组合；EAGLE / MTP / DFlash / vLLM / TensorFlow 不在本版范围。
- 只对 temperature=0 生效，采样批次自动跳过。
- 只对重复内容有收益，全新内容速度持平。
- 换模型、换草稿模型或换硬件，验证窗口甜点需重新实测，不能照搬。

## 路线图

后续更新都加在本仓库，当前规划中：

- 跨用户全局后缀索引（所有请求的输出都进索引，分叉后数个 token 即可重新对上），提升命中率。
- 任意 temperature / top-k 下的非贪心拒绝采样（召回草稿按实际采样分布做接受判定，拒绝时从去掉该 token 后重新归一化的分布采样），保持输出分布不失真。
- 多端统一：同一套算法覆盖 SGLang / vLLM 等更多投机解码头。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。
