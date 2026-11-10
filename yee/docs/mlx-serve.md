# mlx-serve 接法

这一页讲 mlx-serve（M5 上的 xypd 版，Zig）上的跨请求召回补丁：替换哪些文件、怎么编译、上线与回退。

## 1. 这是 source-drop 补丁

不是 diff 补丁，而是**直接替换四个源文件**：

```
src/engines/mlx/spec/dspark.zig
src/engines/mlx/spec/dspark_group.zig
src/engines/mlx/spec/d_recall.zig
src/engines/mlx/spec/d_recall_live.zig
```

```bash
yee patch list
yee patch show mlx-serve
yee patch apply mlx-serve --src <源码根>
```

`apply` 会**先把原文件备份到 `.yee-backup-<时间戳>/`**，再替换成补丁里的版本。

## 2. 改了什么

- `dspark.zig` / `dspark_group.zig`：去掉召回只给贪心请求的条件，采样请求也取召回草稿，走 DSpark 现有 one-hot q 拒绝采样验证（接受概率 p(x)，拒绝从去掉 x 的 p 采样），输出分布不变。
- `d_recall.zig`：16 id 窗口查不到时退到 8、4 id 窗口（短窗口索引各占主索引 1/4 槽位）；短匹配进单独档位（先验减半），草稿长度上限为匹配长度的 2 倍；Gate 按贪心/采样分开统计。
- `d_recall_live.zig`：日志加 `short_hits` 与短档期望。

**penalty 请求仍不走投机**（需要在 verify 里逐位累计 penalty，改动面大，本轮未做）。grammar / logprobs 仍排除。

## 3. 重新编译

```bash
zig build test -Dtest-filter=d_recall     # 79/79 PASS（含新增 6 条）
zig build test -Dtest-filter=dspark       # 72/72 PASS
zig build -Doptimize=ReleaseFast
```

## 4. 上线状态

r593 已在 M5 上线。部署二进制的 sha256（重签名后）：

```
4eb95736d8fefb467663e83bef6d304a88a3358999a124d60a2f24be711ca7c5
```

回滚：把 `/Users/sparker/xiaoyi-dsv41-mlx/runtime/mlx-serve-xypd/mlx-serve.bak-20261110-pre-r593` 拷回 `mlx-serve`，再 kill 进程由 launchd 拉起。

召回记录内存：1280 MiB → 1792 MiB（`MLX_SERVE_D_RECALL_GIB=0.25`、density 1）。

## 5. 环境变量

mlx-serve 侧的变量在它自己的 `env.zig` 里登记：

| 变量 | 默认 | 作用 |
|---|---|---|
| `MLX_SERVE_D_RECALL` | `on` | DeepSeek V4.1 DSpark 复制草稿的跨请求召回开关（`=0` 关闭） |
| `MLX_SERVE_D_RECALL_GIB` | `0.0625` | 召回记录大小（GiB，token id）；索引还要再加约 4×密度 |
| `MLX_SERVE_D_RECALL_INDEX_DENSITY` | `0.25` | 索引里每个 id 分到的哈希槽数（钳在 1/64..4；槽多则记录满了以后旧的窗口还找得到） |

## 6. M5 实测

GSM8K 每档 16 道新题、串行、max_tokens 512，同批题第 1 遍 / 第 2 遍，按题配对的 decode tok/s 比值中位数：

| 温度 | 第 1 遍中位 | 第 2 遍中位 | 按题比值 | 聚合 tok/s | 服务端召回落地 token / 总 token |
|---|---|---|---|---|---|
| 0 | 103.4 | 157.5 | 1.47x | 91.5 → 132.2 | 1814 / 2893 |
| 0.6 | 92.4 | 90.0 | 0.93x | 85.8 → 85.8 | 905 / 3848 |
| 1.0 | 89.8 | 82.9 | 0.96x | 80.7 → 83.0 | 669 / 4220 |

第一版（Gate 不分贪心/采样）在 T=0.6 掉到 0.75x，原因是贪心请求的高落地率让采样请求也被放行；分开统计后采样档不再变慢。采样请求有 16%～24% 的 token 来自召回，但这些位置 DSpark 自己的草稿多半也能命中，所以 GSM8K 上净提速约为 0。模板化输出（格式、代码、固定话术）p(x) 高，预期收益更大，未单独测。

## 7. 回退

1. 从 `.yee-backup-<时间戳>/` 拷回原文件；
2. 重新 `zig build -Doptimize=ReleaseFast`；
3. 用 `mlx-serve.bak-20261110-pre-r593` 覆盖运行中的二进制，kill 进程由 launchd 拉起；
4. 或把 `MLX_SERVE_D_RECALL=0`，关掉召回功能但保留补丁。

通用回退方法见 [patches.md](patches.md)。
