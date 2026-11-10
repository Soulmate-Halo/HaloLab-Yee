# 变更记录

本文件按版本倒序记录模块的变更。首发版本为 1.2.0。

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
