# 变更记录

本文件按版本倒序记录模块的变更。仓库首发版本为 sglang-d-recall 1.2.0（tag `v1.2.0`），2.2.0 起发行包改名为 `yee`，当前版本 4.2.0（tag `v4.2.0`）。

## 4.2.0（Yee）

所有加速开关默认打开，不依赖启动参数，新增一键安装。2.3.0～4.1.1 各版本的详细变化见 `yee/docs/changelog.md`（`yee doc changelog`）。

- 默认打开：召回（`RECALL_ACCEL`，采样请求也用）、KAD（`YEE_KAD`）、热专家验证预算（`YEE_EXPERT_BUDGET`，`_P=0.9`）、vLLM 的 `VLLM_D_RECALL` / `_NONGREEDY` / `_KAD`、逐请求 conf（vcap，`{"mode": "adapt"}`）。召回容量默认 0.25 GiB、索引密度 1。显式设 0 可单独关闭，`YEE=0` 整体不加载。
- 不改启动参数：块长、并发、EP、MoE 后端按原配置运行；SGLang Mia DSV41 镜像里从 `--speculative-dspark-block-size` 读块长，只给没设的变量补配套值。
- 新增跨 rank 版本一致性检查：各 TP rank 都装了同版本 Yee 才启用召回广播和专家预算，否则所有 rank 一起关闭并打印 `rank check off`，服务照常运行。
- 新增 `python -m yee config`：打印每个开关的生效值与来源。
- 新增一键安装 `install/install.sh`：MiaAI dsv41-4x-spark 自动找方案目录、多台机器都装、`--check` / `--restart` / `--uninstall`，失败自动回退；其它机器走 `pip install` + `yee install-hook`。
- 支持范围：SGLang（DSPARK/DFLASH/EAGLE/EAGLE3/NEXTN/MTP/STANDALONE/UNO/NGRAM）、vLLM V2（dspark/dflash/eagle/eagle3/mtp）、llama.cpp、TensorRT-LLM、LMDeploy、mlx-lm、transformers、TensorFold、cuteafd、mlx-serve、strata。
- wheel：`dist/yee-4.2.0-py3-none-any.whl`，sha256 `f6d1e0fde883b60b8d05bbab413bb40c7f8ca3bc155795723ea4ddb7444c6005`。
## 2.2.0（Yee）

recall-accel 2.1.0 改名为 Yee，并扩展到多个推理框架。详细说明见 `yee doc changelog`（源文件 `yee/docs/changelog.md`）。

- 发行包、Python 包、命令统一为 `yee`，wheel 为 `dist/yee-2.2.0-py3-none-any.whl`。
- 支持范围：SGLang（DSPARK/DFLASH/EAGLE/EAGLE3/MTP/STANDALONE/UNO/NGRAM）、vLLM V2（DSpark/DFlash/EAGLE/MTP）、TensorFold、cuteafd、mlx-serve。
- 兼容层：`recall-accel` 命令、`recall_accel` 包、旧环境变量（`RECALL_ACCEL_*`、`SGLANG_D_RECALL_*` 等）全部继续生效；新增等价前缀 `YEE_*`。记录目录默认位置不变。
- 补丁入包：TensorFold `0001-recall-after-priced`、cuteafd v1/v2、mlx-serve 替换文件，sha256 记录在 `yee/patches/MANIFEST.json`，应用前强制核对。
- 新命令：`yee doc [topic]`、`yee bases [--json]`、`yee patch list|show|path|apply`。
- 从 sglang-d-recall 1.2.0 升级：包名和命令变了，`SGLANG_D_RECALL_*` 变量照常生效。

## 1.2.0（首发收录）

新增索引密度参数，让写满后的旧窗口仍可命中。

- 新增 `SGLANG_D_RECALL_INDEX_DENSITY`（旧名 `RECALL_INDEX_DENSITY`），默认 `0.25`，范围 1/64–4。
- 直接映射哈希表按密度分配槽位（槽数取不小于「区容量 × 密度」的 2 的幂）；默认每 4 个 token 一个槽，池子写满后大部分旧窗口查不到。设为 `1` 后，写满时 16-gram 找回率从 12.8% 提升到 31.75%（离线测试）。
- 内存公式更新：`GIB + 冷索引 16 B×槽 + 热索引 8 B×槽 + 热度表`。
- 推荐的 1 GiB 配置：`SGLANG_D_RECALL_GIB=0.25` 且 `SGLANG_D_RECALL_INDEX_DENSITY=1`，写满约 1.01 GiB，容量 6710 万 token。
- 删除该变量即回到 1.1.0 行为，无需换包。

## 1.1.0

热度两级淘汰与去重粒度可调，默认开启。

- 新增 `SGLANG_D_RECALL_EVICT`（默认 `heat`）：记录按每 256 token 一段统计热度（草稿被采用的 token 数 + 写入时因已存过而跳过的重复 token × 0.25），热度按半衰期衰减；热度达 32 的段连同前后文一起复制进热区，冷热都能匹配时热区匹配长度不短于冷区就优先用热区。
- 新增去重粒度 `SGLANG_D_RECALL_DEDUP_RUN`（heat 默认 80，fifo 默认 128），低于 66 时两侧重叠加分隔符比跳过本身还贵。
- `status` 新增 `evict_mode`、`dedup_run`、`hot_ids`、`hot_cap_ids`、`hot_segments`、`promotions`、`hot_evictions`、`hot_hits`、`cold_hits`。
- 设 `SGLANG_D_RECALL_EVICT=fifo` 即与 1.0.0 行为完全相同。

## 1.0.0

首个可用版本：SGLang D 端（decode）token ID 历史召回 + DSpark 草稿替换，输出与不装时逐字相同。
