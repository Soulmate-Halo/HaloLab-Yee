# sglang-d-recall 1.2.0：SGLang D 端（decode）加速模块 部署文档

> 本文写给执行部署的人或 AI。按顺序做完第 2～6 节即可上线。配套只有一个文件：
> `sglang_d_recall-1.2.0-py3-none-any.whl`（sha256 `0DE37D6C03F7DA5FE5A1C6F21A767155F1E4C87D66392B84EC3A7527409B806A`）。
> 1.2.0 相比 1.1.0 只新增索引密度参数 `SGLANG_D_RECALL_INDEX_DENSITY`（默认 0.25，与 1.1.0 相同，见第 7 节末尾）。
> 1.1.0 相比 1.0.0 只新增"热度两级淘汰 + 去重粒度可调"（第 7 节），默认开启；设 `SGLANG_D_RECALL_EVICT=fifo` 即与 1.0.0 行为完全相同。

## 1. 它是什么，能加速什么

- **用途**：SGLang 推理服务 **D 端（decode 阶段）** 的投机草稿加速模块。它把服务读过的提示词和写过的回复按 **token ID** 存下来。新请求生成时，如果最近 16 个以上 token 和历史完全一样，就把历史里后面跟着的内容作为草稿交给主模型验证。验证通过的 token 直接采用，没通过的丢掉照常生成，**输出和不装时逐字相同**。
- **实测**（DeepSeek V4.1 Flash，4×DGX Spark TP4，DSpark 投机解码，验证窗口 8，GSM8K，temperature=0）：同一批题第一次请求单路 111.85 tok/s，第二次 180.09 tok/s（+61%）；8 路并发第二遍合计 283.7 tok/s。全新题目和不装时基本持平。
- **只对重复内容有效**：重跑同一任务、代码改写时大段照抄、引用上下文原文等。全新内容没有可复制的历史，速度不变。
- **只对贪心解码生效**：整批都是 temperature=0 才复制，有采样请求的批次自动跳过。

## 2. 适用前提（先检查，不满足就停下）

| 条件 | 说明 |
|---|---|
| SGLang 版本 | 必须带 DSpark spec-v2 worker：`sglang.srt.speculative.dspark_components.dspark_worker_v2`、`sglang.srt.managers.scheduler_components.batch_result_processor`、`sglang.srt.models.deepseek_v4_dspark`、`sglang.srt.distributed`。即 Mia 的 `dsv41-4x-spark` 镜像这一套。 |
| 投机解码 | 服务以 DSpark 投机解码启动（`--speculative-algorithm` 为 DSpark，块长由 `DSPARK_BLOCK_SIZE` 控制）。 |
| Python 依赖 | 只需要 numpy（SGLang 环境自带）。`cuda` 后端额外需要 torch。 |
| 不适用 | 普通 SGLang（无 DSpark）、EAGLE/MTP、vLLM：核心库能用，但 SGLang 钩子不匹配，需要另写适配层。`doctor` 会报 FAIL。 |

检查方法（在**运行 SGLang 服务的那个 Python 解释器**里，容器部署就在容器里执行）：

```bash
pip install --no-deps sglang_d_recall-1.2.0-py3-none-any.whl   # 或用第 3 节的 --target 方式
python -m sglang_d_recall doctor        # 最后一行 doctor PASS 才继续；FAIL 会列出缺失的模块
python -m sglang_d_recall selftest      # 最后一行 selftest PASS
```

## 3. 安装（二选一）

### 方式 A：容器/镜像不想改动（推荐，现网 ysy37–40 用的就是这种）

1. 在**每台**节点上把 wheel 解到一个目录，这个目录要能被容器看到（挂载进去或直接在容器里执行）：

   ```bash
   mkdir -p /opt/sglang-d-recall
   python3 -m pip install --no-deps --target /opt/sglang-d-recall sglang_d_recall-1.2.0-py3-none-any.whl
   cp /opt/sglang-d-recall/sglang_d_recall/integrations/sitecustomize_template.py /opt/sglang-d-recall/sitecustomize.py
   ```

2. **先确认镜像里原来有没有 sitecustomize**（我们的 `sitecustomize.py` 会遮住它，所以要让它先执行原来那个）：

   ```bash
   docker exec <容器> python3 -c "import sitecustomize,sys; print(sitecustomize.__file__)"
   ```

   - 打印出路径：把这个路径填到 `SGLANG_D_RECALL_MIA_SITECUSTOMIZE`。Mia 镜像是 `/opt/dsv41/adapter/sitecustomize.py`，也就是默认值，可以不填。
   - 报 `ModuleNotFoundError`：原来没有，不用管。

3. 给容器加环境变量（多机 TP 时**每个 rank 的容器都要加**，值相同）。`PYTHONPATH` 必须把 `/opt/sglang-d-recall` 放在**最前面**，后面接容器原来的 `PYTHONPATH`：

   ```bash
   PYTHONPATH=/opt/sglang-d-recall:<容器原来的 PYTHONPATH>
   SGLANG_D_RECALL=1
   SGLANG_D_RECALL_BACKEND=memory          # 见第 5 节
   SGLANG_D_RECALL_GIB=0.0625              # 见第 7 节
   SGLANG_D_RECALL_DIR=/root/.cache/sglang_d_recall/dsv41-flash
   SGLANG_D_RECALL_TOKENIZER_DIR=/models/DeepSeek-V4.1-Flash   # 模型目录，记录按 tokenizer 隔离
   SGLANG_D_RECALL_MODEL_NAME=deepseek-v4.1-flash
   ```

   容器原来的 `PYTHONPATH` 用 `docker exec <容器> printenv PYTHONPATH` 查。
   Mia 的 `start-tp4.sh` 是在 `.env.tp4` 末尾追加一行：

   ```bash
   EXTRA_CONTAINER_ENV="$EXTRA_CONTAINER_ENV PYTHONPATH=/opt/sglang-d-recall:<原值> SGLANG_D_RECALL=1 SGLANG_D_RECALL_BACKEND=memory SGLANG_D_RECALL_GIB=0.0625 SGLANG_D_RECALL_DIR=/root/.cache/sglang_d_recall/dsv41-flash"
   ```

   改之前先备份：`cp -p .env.tp4 .env.tp4.bak-recall`。

### 方式 B：能直接改 SGLang 所在的 Python 环境

```bash
pip install --no-deps sglang_d_recall-1.2.0-py3-none-any.whl
python -m sglang_d_recall install-hook     # 在 site-packages 写 sglang_d_recall.pth，解释器启动即自动装钩子
```

然后只加第 3 步里除 `PYTHONPATH` 外的环境变量。钩子只在 `SGLANG_D_RECALL=1` 时生效，装钩子出错只打印 `[sglang-d-recall] install failed`，服务照常运行。

## 4. 验证窗口（块长）

- 引擎一次验证 `DSPARK_BLOCK_SIZE + 1` 个 token，这也是复制一步最多能拿到的 token 数。
- 本模型（DeepSeek V4.1 Flash + DSpark，4×Spark）实测甜点：`DSPARK_BLOCK_SIZE=7`（窗口 8，现网值）或 `6`（窗口 7，新内容多时更稳）。重复内容超过约 54% 选 7，否则选 6。
- 块长不是 5（checkpoint 默认值）时，**必须同时开着本模块**，因为它带了 DeepSeek V4 草稿置信度按实际块长计算的修正。关模块前先把块长改回 5，否则启动时报 `mat1 and mat2 shapes cannot be multiplied`。
- 换模型、换草稿模型或换硬件，甜点要重新测，不能照搬。

## 5. 存储后端

| `SGLANG_D_RECALL_BACKEND` | 存在哪 | 重启后 | 什么时候用 |
|---|---|---|---|
| `memory` | 进程内存 | 清空 | DGX Spark 这类统一内存机器（现网用这个） |
| `disk`（默认） | `SGLANG_D_RECALL_DIR` 下 4 MiB 一块的文件 | 保留 | 想跨重启保留历史。磁盘剩余低于 `SGLANG_D_RECALL_MIN_FREE_GIB`（默认 4）就停止写入 |
| `cuda` | 显存（torch 张量，设备 `SGLANG_D_RECALL_DEVICE`，默认 `cuda:0`） | 清空 | 独立显存的机器；需要 torch |

## 6. 启动与验收

1. 重启 SGLang 服务（Mia：`./stop.sh` 后 `./start-tp4.sh serve`，约 8–11 分钟）。
2. 日志里必须有：`[recall] hooks installed` 和 `[recall] engine ready ... backend=<后端> ... gamma=<块长>`。没有就看有没有 `install failed`，并重跑第 2 节的 `doctor`。
3. 功能验收：用 temperature=0 把同一条长回答请求（max_tokens≥512）连发两次。
   - 两次输出必须**逐字相同**。
   - 第二次 tok/s 应明显高于第一次（窗口 8 时约 1.5 倍）。
   - 注意：重启后的第一条请求可能和后面的有微小差异（引擎预热），先发一条热身请求再测。
4. 状态：`python -m sglang_d_recall status --dir <SGLANG_D_RECALL_DIR>` 打印 stats.json（提供次数、命中、记录占用）。

## 7. 运维、容量与回滚

**不重启的开关**（都在 `SGLANG_D_RECALL_DIR` 目录里）：

```bash
python -m sglang_d_recall pause on|off --dir <目录>   # 暂停 / 恢复复制（等于建/删 paused 文件）
python -m sglang_d_recall clear --dir <目录>          # 清空记录（建 clear.request 文件）
```

**容量 `SGLANG_D_RECALL_GIB`**：容量 token 数 = GIB × 2^30 / 4。占用内存约 **2 × GIB**（token ID 1×，哈希索引 1×）。

| GIB | 容量 token | 内存约 | 历史还能找回 80% 的间隔 | 还能找回 37% 的间隔 |
|---|---|---|---|---|
| 0.0625（默认） | 16,777,216 | 128 MiB | 其间写入 ≤ 0.84M token | ≤ 4.2M token |
| 0.25 | 67,108,864 | 512 MiB | ≤ 3.4M | ≤ 16.8M |
| 1 | 268,435,456 | 2 GiB | ≤ 13.4M | ≤ 67M |

- "间隔"指同一任务第一次和再次出现之间，服务一共写入了多少 token（提示词 + 回复）。按上表规律：间隔 = 容量的 5% 时能找回约 81%，25% 时约 37%，50% 时约 13%，达到容量就是 0。
- 先看 `status` 里的记录占用。**占用远没满时调大容量没有任何收益。**
- 统一内存机器上，这部分内存和 KV 缓存抢同一块内存，建议不超过 1。
- **heat 模式（默认）下冷区只有容量的一半**（`SGLANG_D_RECALL_HOT_FRAC=0.5`），不热的内容保留时间也减半，上表"间隔"一栏对冷内容要按一半算；热内容进了热区后不受这个间隔限制。总内存不变。

**淘汰策略**（1.1.0 新增，`SGLANG_D_RECALL_EVICT`，默认 `heat`）：

- 记录每 256 个 token 一段，统计热度：草稿被主模型采用的 token 数，加上写入时因已存过而跳过的重复 token × 0.25。热度按半衰期（默认冷区容量的 1/4，按累计写入 token 数计）衰减。
- 冷区是原来的环形存储，照旧按最旧淘汰。一段热度达到 32 就复制进热区（连同前 16 个、后最多 64 个 token）。热区满了淘汰衰减后热度最低的一段。冷热都能匹配时，热区匹配长度不短于冷区就用热区。
- 热区存的是原样复制的 token，草稿照样经主模型验证，**输出仍然逐字不变**。
- 去重：heat 模式默认跳过连续 80 个以上已存过的 token（fifo 为 128）。低于 66 时两侧各 32 个重叠加分隔符比跳过的还多，所以最小有效值是 80。
- 持久化：memory/cuda 后端 `save()`/`load()` 快照带热区和热度表；**disk 后端热区只在内存，重启后为空**，冷区照常从磁盘恢复。`clear` 同时清空热区。
- `status` 新增：`evict_mode`、`dedup_run`、`hot_ids`、`hot_cap_ids`、`hot_segments`、`promotions`、`hot_evictions`、`hot_hits`、`cold_hits`。上线后看 `hot_hits` 和 `cold_hits` 的比例；池子没写满（`evicted_chunks` 为 0）时热区不会带来收益。
- 离线模拟（合成 token，总容量 2^20，写入 8 倍容量）：Zipf 热门模板负载可复制比例从 31.24% 升到 31.78–31.89%（相对 +1.7–2.1%）；热门集合中途整体换掉时从 30.04% 升到 30.41–30.46%（+1.2–1.4%），说明衰减生效、热区没有被旧热点长期占住；全唯一内容为 0，写入速度不降。去重 80 在这组负载里比 128 低约 0.1 个百分点，没有收益。详见 `sim/RESULTS.md`。

**回滚**：只想退回 1.0.0 的淘汰行为，加 `SGLANG_D_RECALL_EVICT=fifo` 并重启即可，不用换包。

**回滚**：方式 A 恢复 `.env.tp4.bak-recall`（或删掉加的环境变量）并重启；方式 B 执行 `python -m sglang_d_recall uninstall-hook` 并去掉 `SGLANG_D_RECALL=1` 后重启。块长不是 5 时先改回 5。

**索引密度**（1.2.0 新增，`SGLANG_D_RECALL_INDEX_DENSITY`，旧名 `RECALL_INDEX_DENSITY`，默认 `0.25`，范围 1/64–4）：每个存储 token 配多少个索引槽。索引是直接映射哈希表，后写覆盖先写；默认每 4 个 token 一个槽，池子写满后大部分旧窗口查不到。设 `1` 后写满时 16-gram 找回率从 12.8% 升到 31.75%（离线测试）。池子没写满时没有影响。
- 内存 ≈ GIB + 冷索引 16 B×槽 + 热索引 8 B×槽 + 热度表，槽数取不小于"区容量×密度"的 2 的幂。
- 推荐的 1 GiB 配置：`SGLANG_D_RECALL_GIB=0.25 SGLANG_D_RECALL_INDEX_DENSITY=1`（heat 模式），写满约 1.01 GiB，容量 6710 万 token。
- 回到 1.1.0 行为：删掉该变量即可，不用换包。

## 8. 环境变量全表

新名字优先，旧名字（现网早期部署用的）仍可回退读取，空值视为未设置。

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

## 9. 给执行部署的 AI 的检查清单

1. `doctor` 和 `selftest` 都 PASS，否则停下报告缺失项，不要改 SGLang 源码硬套。
2. 每个节点、每个 rank 的容器环境变量一致，`PYTHONPATH` 里本模块目录排第一。
3. 原有 sitecustomize 已通过 `SGLANG_D_RECALL_MIA_SITECUSTOMIZE` 接上（第 3 节第 2 步）。
4. 改配置前留备份；重启后日志有 `engine ready`；同一请求连发两次逐字相同且第二次更快。
5. 块长不是 5 时，关闭模块前先把块长改回 5。
6. 从 1.0.0 升级：只换 wheel、其余环境变量不动即可；要保持旧行为加 `SGLANG_D_RECALL_EVICT=fifo`。升级后 `status` 里应出现 `evict_mode`。
