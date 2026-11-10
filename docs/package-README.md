# sglang-d-recall 1.2.0

SGLang decode(D) 端的 Recall 投机草稿加速器：把推理服务读过的提示词和写过的回复按 **token ID** 记下来。新请求的末尾出现和历史完全一样的 16 个以上 token 时，取出历史里后面跟着的内容当草稿，交给主模型验证。验证通过的 token 直接采用，没通过的丢掉、照常生成，所以输出和不开 Recall 时一致。

- 只存 token ID（int32），不存文本。
- 精确匹配，不用向量相似度：16-gram 滚动哈希定位候选位置，再逐个 token 比对排除碰撞，最多往前延伸 128 个 token。
- 按匹配长度分三档（16–31、32–63、64+），各自统计历史命中率。只有预计比引擎自己的草稿划算时才提供复制草稿；连续落空的请求冷却 4 轮。
- 只对整批都是贪心解码（temperature=0）的批次生效。

## 安装

```
pip install sglang_d_recall-1.0.0-py3-none-any.whl          # 只依赖 numpy
pip install "sglang_d_recall-1.0.0-py3-none-any.whl[torch]"  # 要用 cuda 后端时
```

装完可以用命令行工具：

```
python -m sglang_d_recall doctor          # 检查环境与 SGLang 依赖
python -m sglang_d_recall install-hook     # 在当前解释器 site-packages 写 sglang_d_recall.pth
python -m sglang_d_recall uninstall-hook   # 撤掉
python -m sglang_d_recall status           # 打印记录目录的 stats.json
python -m sglang_d_recall pause on|off     # 暂停/恢复复制
python -m sglang_d_recall clear            # 请求清空记录
python -m sglang_d_recall selftest         # 内置自检，不需要 pytest
```

## 三种存储后端

| backend | token ID 放在哪 | 重启后 | 适用 |
|---|---|---|---|
| `disk`（默认） | 目录下的 `chunk-XXXXXXXX.ids` 文件，每块 1,048,576 个 ID（4 MiB） | 保留 | 想跨重启复用记录 |
| `memory` | 进程内 numpy 块 | 丢失，可用 `save()`/`load()` 存快照 | 不想写盘；DGX Spark 这类统一内存机器首选 |
| `cuda`（别名 `gpu`） | `torch.int32` 张量，设备默认 `cuda:0` | 丢失，可快照 | 独立显存的机器，想把记录放进显存 |

三种后端的 `add / lookup / continuation / clear / stats` 语义相同，`tests/test_backends_equiv.py` 用随机序列对拍验证过。哈希索引在三种后端里都放在主机内存。`cuda` 后端每次 `continuation()` 会把取出的那一小段拷回主机（返回 numpy），在统一内存机器上没有好处，所以 Spark 上用 `memory`。

**内存估算**（`gib` 是记录容量，默认 1/16 GiB = 16,777,216 个 token）：
- 索引：固定 `cap_ids / 4` 个槽，每槽 16 字节，也就是每个容量 token 4 字节。默认 64 MiB，启动就分配。
- token ID：每 token 4 字节，按 4 MiB 一块随用随分配，满了淘汰最旧的块。默认最多 64 MiB。
- 合计默认上限约 128 MiB。本机实测（`tests/test_recall_core.py` 的 perf 项）：写入 100 万 token 后进程内记录占 72 MiB，`add` 100 万 token 约 140–200 ms，`propose` p50 约 21 µs、p99 约 30 µs。

## 独立 API（任何推理引擎都能接）

```python
from sglang_d_recall import RecallRecord, RecallEngine, tokenizer_identity

ident = tokenizer_identity(tokenizer, model_name="my-model")   # 记录按 tokenizer 隔离
rec = RecallRecord(None, ident, gib=1/16, backend="memory")     # disk 后端第一个参数给目录
eng = RecallEngine(rec, gamma=7)                                # gamma = 引擎一次验证的草稿 token 数

eng.begin_stream(rid, prompt_ids)            # 请求进来时
# 每个解码步：
p = eng.propose(rid, own_ema=engine_accept_len)   # None 表示这步不复制，用引擎自己的草稿
if p is not None:
    draft = p.tokens                          # 正好 gamma 个 ID；p.real 是其中真正来自历史的个数
    ...                                       # 交给主模型验证，得到 accepted 个
    eng.feedback(rid, landed=accepted)
else:
    eng.observe_own(rid, accepted=own_accepted)
eng.extend_stream(rid, newly_committed_ids)  # 每步把实际输出的 token 追加进去
eng.tick()                                   # 每个调度步调一次：每 2 秒写 stats.json、处理 clear.request
eng.end_stream(rid, output_ids, finished=True)   # 结束时，回复写入记录（后台线程）
```

`RecallRecord.save(path)` / `RecallRecord.load(path, identity, backend="memory")` 用于 memory/cuda 的快照，文件里带 tokenizer 身份，身份不符会抛 `RecallIdentityError`。

## 接入 SGLang（DSpark 投机解码）

`sglang_d_recall.integrations.sglang_dspark` 是现网 ysy37–40 上在用的钩子，挂在以下模块上：
`sglang.srt.speculative.dspark_components.dspark_worker_v2`、`sglang.srt.managers.scheduler_components.batch_result_processor`、`sglang.srt.models.deepseek_v4_dspark`。

三种接入方式，任选其一：

1. **自动加载（推荐）**：`python -m sglang_d_recall install-hook` 在当前解释器的 site-packages 写一个 `sglang_d_recall.pth`（内容 `import sglang_d_recall.autoload`）。解释器一启动就装钩子，只有 `SGLANG_D_RECALL=1` 时才真正安装；任何异常只打印 `[sglang-d-recall] install failed` 并继续，服务不受影响。
2. **sitecustomize**：把 `integrations/sitecustomize_template.py` 复制成某个目录下的 `sitecustomize.py`，再把同一目录放到容器 `PYTHONPATH` 的最前面。它会先执行 `SGLANG_D_RECALL_MIA_SITECUSTOMIZE`（旧名 `DSV41_RECALL_MIA_SITECUSTOMIZE`）指向的 Mia sitecustomize，再装 Recall 钩子。
3. **代码里手动装**：`from sglang_d_recall.integrations import sglang_dspark; sglang_dspark.install()`。

然后设置环境变量 `SGLANG_D_RECALL=1`，以及下表里需要改的变量，再重启服务。日志里出现 `[recall] engine ready ... backend=memory ... gamma=N` 就是生效了。

## 环境变量（取自 `core.engine_from_env` 和 `sitecustomize_template.py`）

新名字优先，旧名字作为回退仍然生效；空值按"未设置"处理。

| 变量（新） | 旧名（回退） | 默认 | 含义 |
|---|---|---|---|
| `SGLANG_D_RECALL` | `DSV41_RECALL` | `0` | `1/true/yes/on` 开启 |
| `SGLANG_D_RECALL_DIR` | `DSV41_RECALL_DIR` | `~/.cache/sglang_d_recall/<tokenizer 摘要前 16 位>` | disk 后端的记录目录；memory/cuda 下只放控制文件（`paused`、`clear.request`、`stats.json`） |
| `SGLANG_D_RECALL_GIB` | `DSV41_RECALL_GIB` | `0.0625` | 记录容量（GiB，按每 token 4 字节换算） |
| `SGLANG_D_RECALL_BACKEND` | `RECALL_BACKEND` | `disk` | `disk` / `memory` / `cuda`（`gpu`） |
| `SGLANG_D_RECALL_DEVICE` | `RECALL_DEVICE` | `cuda:0` | cuda 后端的设备 |
| `SGLANG_D_RECALL_MIN_MATCH` | `DSV41_RECALL_MIN_MATCH` | `16` | 最短匹配长度 |
| `SGLANG_D_RECALL_PRIOR` | `DSV41_RECALL_PRIOR` | `0.8` | 各档命中率的先验 |
| `SGLANG_D_RECALL_READ_ONLY` | `DSV41_RECALL_READ_ONLY` | `0` | 只读：只复制，不写新记录 |
| `SGLANG_D_RECALL_MIN_FREE_GIB` | `DSV41_RECALL_MIN_FREE_GIB` | `4` | disk 后端磁盘剩余低于此值就停止写入 |
| `SGLANG_D_RECALL_TOKENIZER_DIR` | `DSV41_RECALL_TOKENIZER_DIR` | `/models/DeepSeek-V4.1-Flash` | 算 tokenizer 摘要的目录 |
| `SGLANG_D_RECALL_MODEL_NAME` | `DSV41_RECALL_MODEL_NAME` | `deepseek-v4.1-flash` | 写进记录身份里的模型名 |
| `SGLANG_D_RECALL_MIA_SITECUSTOMIZE` | `DSV41_RECALL_MIA_SITECUSTOMIZE` | `/opt/dsv41/adapter/sitecustomize.py` | 先执行的原有 sitecustomize |

不重启的运维开关（记录目录里）：建 `paused` 文件暂停复制，删掉恢复；建 `clear.request` 文件清空记录。

## 淘汰策略（1.1.0 新增）

| 变量（新） | 旧名（回退） | 默认 | 含义 |
|---|---|---|---|
| `SGLANG_D_RECALL_EVICT` | `RECALL_EVICT` | `heat` | `heat`：冷区加热区两级淘汰；`fifo`：和 1.0.0 完全一样，只按最旧淘汰 |
| `SGLANG_D_RECALL_HOT_FRAC` | `RECALL_HOT_FRAC` | `0.5` | 热区占总容量的比例，总容量不变，冷区相应变小；设为 `0` 等同 `fifo` |
| `SGLANG_D_RECALL_DEDUP_RUN` | `RECALL_DEDUP_RUN` | heat `80` / fifo `128` | 已存过的内容连续多少个 token 以上才跳过不存（最小 32） |
| `SGLANG_D_RECALL_PROMOTE_TOKENS` | `RECALL_PROMOTE_TOKENS` | `32` | 一段的热度达到多少就复制进热区 |
| `SGLANG_D_RECALL_HEAT_HALFLIFE_IDS` | `RECALL_HEAT_HALFLIFE_IDS` | 冷区容量的 1/4 | 热度半衰期，按累计写入的 token 数计 |
| `SGLANG_D_RECALL_HEAT_SEEN_WEIGHT` | `RECALL_HEAT_SEEN_WEIGHT` | `0.25` | 去重时跳过的每个重复 token 记多少热度 |
| `SGLANG_D_RECALL_INDEX_DENSITY` | `RECALL_INDEX_DENSITY` | `0.25` | 1.2.0 新增。每个存储 token 配多少个索引槽（冷区、热区都按这个比例，范围 1/64–4） |

- **索引密度（1.2.0）**：索引是直接映射的哈希表，记录里每个位置都插入，后写的覆盖先写的。默认 0.25（每 4 个 token 一个槽，和 1.1.0 相同），池子写满后大部分旧窗口会被覆盖、查不到；设成 `1` 时每个 token 一个槽。池子没写满时密度对命中几乎没有影响。
- **内存估算**：总内存 ≈ 记录 token（GIB）+ 冷区索引 16 B × 槽数 + 热区索引 8 B × 槽数 + 热度表；槽数 = 不小于"区容量 × 密度"的 2 的幂。例：heat 模式、`GIB=0.25`、`INDEX_DENSITY=1`，冷区、热区各 3355 万 token，冷索引 512 MiB + 热索引 256 MiB + token 256 MiB + 热度表约 9 MiB，实测 `bytes_mem` 1032.7 MiB（约 1.01 GiB，memory 后端写满后的值）。
- **密度的离线结果**（总容量 2^20、写入 8 倍容量，见 `sim/DENSITY_RESULTS.md`）：heat 模式下密度 0.25 → 1 时可复制比例相对提升 4.4%–4.6%，fifo 模式 5.6%–5.9%；写满后 16-gram 找回率从 12.8% 升到 31.75%。代价是写入吞吐从约 6.4M 降到约 2.1M token/s，单次查找 p50 4.8 → 5.3 µs。

- **热度怎么算**：记录按全局写入位置每 256 个 token 切成一段。热度有两个来源：草稿被主模型采用的 token 数，记到草稿来源所在的段；写入时因为已存过而跳过的重复 token，按"个数 × 0.25"记到已有内容所在的段。热度按半衰期指数衰减，所以老热点会慢慢让位给新热点。
- **冷区和热区**：冷区就是原来的环形存储，照旧按最旧淘汰。某段热度达到 32 时，把这段连同前 16 个、后最多 64 个 token 复制进热区并建索引。复制进热区以后，热度继续累加在热区这份上。热区满了就淘汰衰减后热度最低的一段，被淘汰的段要重新攒够热度才能再进来。同一段内容冷区和热区都能匹配上、而且热区的匹配长度不短于冷区时，用热区。
- **输出不变**：热区存的是原样复制的 token，草稿照样交给主模型验证，输出仍然逐字不变。
- **去重粒度为什么是 80 而不是 32**：每跳过一段重复内容，左右两边保留的新内容各自要多存 32 个 token 的重叠（`OVERLAP`，保证边界处的 16-gram 还能被索引），再加 1 个分隔符。重复段短于 66 个 token 时，多存的比跳过的还多。80 = 2×32+16，每段至少净省 15 个 token。
- **内存开销**（GIB=1，总容量约 2.68 亿 token）：冷区和热区的 token 合计仍约 1 GiB；冷区索引约 512 MiB，热区索引约 256 MiB（键只存 32 位，每槽 8 字节），合计比 1.0.0 的 1 GiB 索引还少；热度表约 24 MiB（`_seg_tag`/`_seg_L`/`_seg_hot` 各 8 字节 × 2^20 段）。写入和查找都不做全表扫描；热区淘汰用堆，过期的堆项累计超过存活段 4 倍时才重建一次。
- **持久化**：memory/cuda 后端的 `save()`/`load()` 快照会带上热区和热度表（按热度从高到低装回，装不下的丢掉）。**disk 后端的热区只在内存里**，重启后热区为空，冷区照常从磁盘恢复。`clear()` 同时清空热区和热度。1.0.0 也能读 1.1.0 的快照（多出来的数组会被忽略）。
- `stats()` / `status` 新增：`evict_mode`、`dedup_run`、`hot_ids`、`hot_cap_ids`、`hot_segments`、`promotions`、`hot_evictions`、`hot_hits`、`cold_hits`。
- 离线模拟结果见交付目录 `sim/RESULTS.md`。

## 能不能用在别的模型上

- **`sglang_d_recall.core`（记录、匹配、决策）与模型无关**，只认 token ID。任何模型都能用，前提是引擎有"一次验证多个草稿 token"的路径。记录按 tokenizer 隔离，换 tokenizer 不能共用记录（会报 `RecallIdentityError`）。
- **`integrations/sglang_dspark.py` 只适用于当前这套**：Mia 镜像里的 SGLang 版本 + DSpark spec-v2 worker（含 DeepSeek V4 的 `compute_confidence` 修正）。换成 SGLang 的 EAGLE/MTP 或 vLLM，要另写一个适配层，把复制草稿并进那个引擎的草稿张量里，现有约 450 行可以照着改。
- 没有投机解码的引擎要先有验证路径。SGLang 自带的 `NGRAM` 投机模式、vLLM 的 prompt lookup / suffix decoding 原理类似，但记录只在单进程内存里、不跨重启。
- **加速多少取决于流量里的重复比例和验证窗口**。窗口的甜点值和模型、草稿模型、硬件内核都有关，换模型要重新测，不能直接沿用这里的 7/8。

## 测试

```
python tests/test_recall_core.py      # disk + memory (+ cuda，有 torch 和 GPU 时)
python tests/test_backends_equiv.py   # 三后端对拍
python tests/test_recall_patch.py     # SGLang 钩子的模拟器测试（需要 torch）
python tests/test_heat.py             # 热度两级淘汰（也可 pytest 收集）
RECALL_EVICT=fifo python tests/test_recall_core.py   # fifo 模式回归
```
