# 热专家验证预算（expert_budget，4.1.0 引入，4.2.0 默认开）

`yee.integrations.expert_budget` 给 DeepSeek V4.1 在 Mia 镜像上的路由专家 MoE（适配器 `moe_b12x_next`，E=384，top-6）加一个按本步热度裁专家的预算。4.2.0 起默认启用。

## 1. 它做什么

GB10 上解码的 MoE 受带宽限制：一步的 MoE 耗时约等于把本步各行路由到的不同专家各读一遍。

启用后，每次 MoE 调用（每层每步一次）：

1. 按本步所有行的路由权重给每个专家记一个质量（权重量化到 1/4096 后求和）。
2. 只保留覆盖 `p` 比例质量的最热一批专家（并列整档保留），可再用 `kcap` 限制个数。
3. 被丢的 (行, 槽) 改指向本行保留的专家、权重置 0；本行保留的权重按比例放大，使行和不变。一行一个都没保留时，保留它权重最大的那个。

选哪些专家每一步跟着当前批次变，不需要预先统计热度，业务内容变了自动跟着变。草稿头（E=128）不动。

替换的是确定性路径（`DSV41_MOE_B12X_NEXT_DETERMINISTIC=1`，`_Geometry.direct` 为假）里 `ep_remap` 那一次启动：一个单 CTA 的 Triton 内核，直接写 `ids_buf`/`w_buf`，不加启动次数，可被 CUDA graph 捕获。超过 `YEE_EXPERT_BUDGET_MAXM` 行（默认 512）走原路径。

服务加载时每个 E=384 几何先跑一次预热（完成 JIT）、12 例与参考实现的逐例比对和一次定点切换检查；任何一项不通过就禁用这个几何（走原路径）并打印 `[yee-eb] geometry disabled: ...`。

## 2. 生效条件

热专家验证预算**只在 Mia 的 `moe_b12x_next` 确定性路径上起作用**，也就是同时满足：

- `DSV41_MOE_B12X_NEXT=1`
- `DSV41_MOE_B12X_NEXT_DETERMINISTIC=1`
- `EP_SIZE=1`

别的 MoE 路径上自动不介入，其余加速（召回、KAD 等）照常。

## 3. 默认值与开关

4.2.0 起默认打开，不用设环境变量：

| 环境变量 | 默认 | 作用 |
| --- | --- | --- |
| `YEE_EXPERT_BUDGET` | `1` | 总开关；设 0 关闭（装导入钩子这一步都不做） |
| `YEE_EXPERT_BUDGET_ARMED` | `1` | 启动初值：是否已 armed；设 0 启动即不启用 |
| `YEE_EXPERT_BUDGET_P` | `0.9` | 启动初值：保留覆盖多少比例的路由质量 |
| `YEE_EXPERT_BUDGET_E` | `384` | 打补丁的几何专家数 |
| `YEE_EXPERT_BUDGET_MAXM` | `512` | 行数上限，超过走原路径 |
| `YEE_EXPERT_BUDGET_CTRL` | `/state/yee_expert_budget.json` | 控制文件（每个节点各一份） |
| `YEE_EXPERT_BUDGET_STATS` | `/state/yee_expert_budget_stats.jsonl` | 统计文件（全局 rank 0 写） |
| `YEE_EXPERT_BUDGET_STATS_S` | `10` | 统计间隔（秒） |
| `YEE_EXPERT_BUDGET_KCAP` | `4096` | 启动初值：最多保留多少个专家 |

`/state` 不存在时，控制文件与统计文件落在 `~/.cache/yee/` 下。

4.2 默认启动即启用：各 rank 从第一步起就是同一档，不需要控制文件。要改档时把每个节点的控制文件写成同样的值（见第 4 节）。

## 4. 不重启改档

控制文件：`{"seq": int, "armed": 0|1, "p": (0,1], "kcap": int>=0, "at_calls": int（可选）}`，`seq` 变了才读。

- **带 `at_calls`（有流量时用）**：新值先放进待切换槽，内核在第 `at_calls` 次调用时自己切换。各 TP 卡执行的调用序列相同，所以 4 张卡在同一步的同一层切换。`at_calls` 取 rank 0 统计文件最新一行的 `calls`，加上几百个前向的余量（每个前向 40 次调用），例如 `calls + 12000`。4 台的控制文件必须写成同一个 `at_calls`。日志先打 `[yee-eb] ctrl seq=N ... scheduled at_calls=...`，生效时打 `... active ...`。
- **不带 `at_calls`**：立即写入当前槽。各节点轮询时刻不同（相差最多约 1 秒），有流量时这一秒里各卡的专家集合可能不一致，所以只能在空闲时用。

## 5. 统计

rank 0 每 `STATS_S` 秒追加一行：`{"t","seq","armed","p","kcap","calls","pending_at","stats"}`，`stats` 是 17×8 的累计计数，行号为桶 `min(ceil(M/8),16)`（静态验证每请求 8 行，桶号约等于并发请求数），8 列依次是调用次数、行数、裁前不同专家数之和、裁后实际读的不同专家数之和、总质量、保留质量、强制保留行数、启用状态下的调用次数。两行相减再除以调用次数就是该时段每层每步的均值。

## 6. 日志怎么看

| 日志 | 含义 |
|---|---|
| `[yee-eb] armed geometry E=... topk=... device=... ctrl=p:... kcap:... armed:...` | 这个几何已 armed，控制参数如后所示 |
| `[yee-eb] ctrl seq=N armed=... p=... kcap=... applied rank=... geoms=... (immediate)` | 控制文件的新值已立即应用 |
| `[yee-eb] ctrl seq=N armed=... p=... kcap=... scheduled at_calls=...` | 新值已排到第 `at_calls` 次调用切换 |
| `[yee-eb] ctrl seq=N active armed=... p=... kcap=... at_calls=... calls=...` | 定点切换已生效 |
| `[yee-eb] geometry disabled: ...` | 该几何的自检没过，走原路径 |
| `[yee-eb] rank check on: N ranks` | 各 TP rank 都装了同一版本 Yee，可以启用 |
| `[yee-eb] rank check off: ...` | 有 TP rank 没装 Yee 或版本不同，专家预算在所有 rank 一起关闭 |

跨 rank 一致性检查：各 TP rank 都装了同一版本 Yee 才启用专家预算（以及召回广播），否则所有 rank 一起关闭并打印上面那行 `rank check off`。

## 7. 怎么关

- **环境变量**：`YEE_EXPERT_BUDGET=0`，重启后整个模块不介入。
- **不重启**：把控制文件改成 `armed=0`（有流量时带 `at_calls`，让所有 rank 在同一步切换）。

## 8. 速度

4 台 DGX Spark 实测 p=0.90 合计吞吐 +17.3%（并发 8 时 new 类 +22.5%）。

## 9. 已知限制

- 只在 `moe_b12x_next` 确定性路径上起作用（见第 2 节），别的 MoE 路径不介入。
- 启动即启用（4.1.1 起）已在 4 台 DGX Spark 上运行；带 `at_calls` 的定点切换只在单卡机器上做过单测、图捕获与接线测试。
- 服务占满显存时容器里无法另起进程跑 `python expert_budget.py --selftest`，该几何的比对依赖服务加载时的自检。

## 10. 回退

设 `YEE_EXPERT_BUDGET=0` 后重启（或 `bash install.sh --uninstall --restart` 卸掉整个 Yee）；不重启的话把控制文件改成 `armed=0`（有流量时带 `at_calls`）。
