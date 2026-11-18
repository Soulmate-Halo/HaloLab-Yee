# 多分支树草稿（2.3.0，P4）

一轮验证里同时摆几条候选续写：草稿头自己的链（或它自己的 top-k 树），加上召回记录里找到的几条续写。它们合并成一棵挂在已确认 token `t1` 下面的 token 树，目标模型一次前向给每个节点打分，然后沿着和目标选择一致的分支往下走。

## 原理

- **候选**：`RecallEngine.candidates()` 把同一上下文能续写的地方全列出来：请求自己上下文里的重复、记录的热区命中、冷环命中、8/4 短窗口命中，按内容去重（一条是另一条的前缀时只留长的）。不动任何计数。
- **合并成树**：`tree.build_tree()` 把多条链合并成 trie。几条链共享的前缀算同一个事件；同一父节点下的兄弟节点互斥（目标在一个节点只选一个 token），概率和超过 1 时按比例缩到 1。按节点预算（验证行数）保留概率最高的节点，祖先一定一起保留。
- **验证**：
  - 贪心：每个节点取目标 argmax，跟着 token 相等的子节点走。
  - 采样、草稿是点质量（召回续写、贪心头）：每个节点从目标的过滤后分布独立抽一次，跟着相等的子节点走（`tree.verify_walk`）。这和逐 token 串行采样分布完全一样。
  - 采样、草稿头按 q 抽的子节点（EAGLE top-k 采样）：`tree.verify_specinfer`，子节点按 `min(1, p/q)` 接受，拒绝后残差 `norm(max(p - q, 0))` 交给下一个兄弟，全拒绝从残差抽。对任何 q 都无损。
- **只能验证一条链的基座**：`propose_select` 从全部候选里挑期望落地最多的一条填进链（头自己的上下文和召回一起比）。只有一个候选时和 `propose` 完全一样。

## 各基座支持情况（`yee bases --json` 的 `tree` 字段）

| 基座 | 草稿头 | tree | 说明 |
|---|---|---|---|
| mlx-serve（M5） | DSpark | native | r598：召回续写和 DSpark 块各是一个分支状态，一次批量验证（`dsparkTreeStep`），赢家状态接管 |
| SGLang | DSPARK / DFLASH | select | 一条块链，`propose_select` 选最好的候选 |
| SGLang | EAGLE / EAGLE3 / NEXTN / MTP / STANDALONE / UNO（topk=1） | select | 链 |
| SGLang | 同上（topk>1，真树） | none | 召回仍关闭；`tree.merge_chains` / `to_sglang` / `from_sglang` 已能在 `build_tree_kernel_efficient` 的数组和树之间转换，但原地改写这些数组的钩子没写（掩码布局要在 GPU 上核对） |
| SGLang | NGRAM | select | Yee 把整块换成链；SGLang 自己的 BFS trie 没有喂召回（`tree.ngram_insert` 是给原生 trie 钩子用的辅助函数） |
| vLLM V2 | DSpark / DFlash / EAGLE / EAGLE3 / MTP | select | V2 runner 的 propose 返回链；`speculative_token_tree` 静态树槽位不动 |
| TensorFold / cuteafd | — | none | 原生或补丁路径，2.3.0 不改 |

## 开关

| 变量 | 默认 | 作用 |
|---|---|---|
| `YEE_TREE`（别名 `RECALL_ACCEL_TREE`） | 1 | 0 = 回到 2.2.0：链式基座只用 `propose` |
| `YEE_TREE_K`（别名 `RECALL_ACCEL_TREE_K`） | 2 | 一轮最多加几个召回分支（0..3） |
| `YEE_TREE_NODES` | 头自己的 num_draft_tokens | 合并树保留的节点数（含根） |
| mlx-serve：`MLX_SERVE_D_RECALL_TREE` | on | =0 关闭分支状态树轮 |
| mlx-serve：`MLX_SERVE_D_RECALL_TREE_K` | 2 | 召回分支上限（0..3） |
| mlx-serve：`MLX_SERVE_D_RECALL_TREE_STATE_COST` | 0.08 | 每多一个分支状态的成本（相对一行验证） |
| mlx-serve：`MLX_SERVE_D_RECALL_TREE_MARGIN` | 0.05 | 树方案要比 DSpark 单链多出的期望收益 |
| mlx-serve：`MLX_SERVE_D_RECALL_TREE_MAX_MIB` | 512 | 解码状态超过这个大小就不分支 |
| mlx-serve：`MLX_SERVE_D_RECALL_TREE_TRACE` | off | =1 每个树轮打印各步日志（诊断用） |

## mlx-serve 的树轮怎么选方案

`d_recall.choosePlan`：分支 0 是 DSpark 块（可截任意长度，0 = 不要），其余是召回候选（各试几种截断），最多 K 个召回分支，总行数不超过批量验证的编译宽度 16。按"期望前进 token 数 /（验证行成本 + 额外分支状态成本 + DSpark 起草成本）"打分。含召回分支的最好方案要比 DSpark 单独的最好方案高出 `MARGIN` 才执行，否则这一轮就是原来的 DSpark 轮；召回分支长时间没被选中时，每 32 轮强制试一次，让它的接受率继续学习。单独一个召回续写就能赢 DSpark 时，仍走原来的复制轮（不起草 DSpark）。

## 已知限制

- SGLang EAGLE topk>1、vLLM `speculative_token_tree`：只有转换函数，没有运行时钩子。
- mlx-serve 的多槽并发（group round）不走树轮。
- 分支状态共享 GPU 行，第一次写入时各自复制一份（MLX 的写时复制），状态越大复制越多，所以有 `TREE_MAX_MIB` 上限。
