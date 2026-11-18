# 更新记录

这一页列出 Yee 各版本的变化。

## 4.3.0

### `yee doctor` 全适配诊断：装没装、能不能加速、改哪个参数，一眼看完

- `python -m yee doctor` 扫全部署：认得 15 个引擎（SGLang、vLLM、llama.cpp、TensorRT-LLM、LMDeploy、mlx-lm、mlx-serve、TensorFold、strata、Maya、cuteafd、TGI、Ollama、ktransformers、unknown），发现本机进程、docker 容器、ssh 远端（`--hosts`）、K8s（`--k8s`）、远端 URL（`--url`）、Maya 的 JSON 启动配置（`--config`）。
- 每个实例给一个结论：**可加速 / 要改参数才能加速 / 要打补丁 / 不支持 / 无需处理**，并列出功能状态（召回、KAD、专家预算、树形、记录共享）。
- 改动清单里的每一项都是**可以直接复制的**：参数片段、`KEY=VALUE`、补丁命令、路由配置。规则号 H0–H9（投机头）、P1–P4（补丁引擎）、C1–C11（并发与多流）、K1–K8（集群）逐条对应。
- 退出码：`0` 全部可用或无需处理；`1` 有要改参数 / 要打补丁 / 集群必改项；`3` 没找到实例或全部不支持。
- `--json` 输出 plan JSON，供脚本与远端解析。

### 运行中的 `[yee] plan` / `[yee] effect:` 提示

- 服务启动时（认得出引擎的进程，每进程一次）打印 `[yee] 4.3.0 plan: <engine> head=<头> -> <结论>`，有改动时跟几行 `[yee] 要改: <可复制内容>`，末尾提示 `python -m yee doctor`。
- 运行中在 `[recall] steps=` 之后按 `YEE_EFFECT_S`（默认 600 秒）打印 `[yee] effect:` 行：`E1` 表示钩子装上了但投机路径没跑到，`E2` 表示召回贡献占比很小，`E3` 表示链路有错，其余是正常工作的统计。
- 两个提示都可关：`YEE_NOTICE=0`。

### 安装套件 generic 模式（任何部署）

- `install.sh` 不再只认 Mia 方案：找不到方案目录时自动走 generic 模式，本机进程、docker 容器、`--hosts` 指定的每台机器都能装，装完打印诊断报告并写 `yee-plan.json`。
- 容器目标自动用容器里的解释器安装并核对，容器重建会丢的情况给出两种持久做法（装进镜像，或挂载 + `PYTHONPATH`）。
- 新参数：`--cmd`、`--url`、`--hosts`、`--config`、`--no-docker`、`--k8s`、`--restart-cmd`、`--plan-out`、`--pip`。

### 新增 `maya-recall` 补丁

- Project Maya v1.1.66 的跨请求召回补丁（`patch -p1`），重编命令 `cmake --build build --target maya`。
- 开关：`RECALL_ACCEL=1`、`RECALL_ACCEL_DIR=<目录>`、`MAYA_GLM_MULTI_MTP=3`、`MAYA_YEE_RECALL_ROWS`。
- 限制：`MAYA_GLM_MULTI_MTP_MAX`（默认 4）之外时召回只记账；8×3080 实测 8 路放开后 65.1 tok/s，默认配置 76.6 tok/s。

### 新增文档 `deploy.md`

全适配与集群部署一页纸：诊断命令与退出码、15 个引擎的适配矩阵、投机头规则 H0–H9、补丁引擎 P1–P4、并发与多流 C1–C11、集群与部署形态 K1–K8（多卡 TP/PP/EP、多副本同机与跨机、PD 分离、K8s、容器持久做法）、怎么看是否生效、收益预期。

## 4.2.0

### 所有加速开关默认打开，不依赖启动参数；跨 rank 一致性检查；一键安装

- 默认打开（只在没设时生效，显式设 0 仍可单独关闭；`YEE=0` 或 `RECALL_ACCEL=0` 时整个 Yee 不加载）：召回总开关 `RECALL_ACCEL`（`YEE`）、KAD `YEE_KAD`、热专家验证预算 `YEE_EXPERT_BUDGET`（启动即启用，`YEE_EXPERT_BUDGET_ARMED=1`、`YEE_EXPERT_BUDGET_P=0.9`）、vLLM 的 `VLLM_D_RECALL`、`VLLM_D_RECALL_NONGREEDY`、`VLLM_D_RECALL_KAD`。召回容量 `RECALL_ACCEL_GIB` 缺省 0.25，`RECALL_ACCEL_INDEX_DENSITY` 缺省 1。`install.sh` 放 `mia-dsv41-vcap` 时同时写控制文件 `{"mode": "adapt"}`（已存在则保留），逐请求 conf 默认打开。
- SGLang Mia DSV41 镜像里按启动参数自动补默认值：从 `--speculative-dspark-block-size N` 读块长，补 `DSPARK_BLOCK_SIZE=N`、`DSV41_VERIFY_CAP=conf:0.1`、`DSV41_VERIFY_CAP_STRIDE=N+1`、`SGLANG_DSPARK_FOLDED_SAMPLING=2`。块长、EP、MoE 后端等启动参数不改。
- 跨 rank 一致性检查：各 TP rank 都装了同一版本 Yee 才启用召回广播和热专家验证预算，否则所有 rank 一起关闭并打印 `[recall] rank check off: ...` / `[yee-eb] rank check off: ...`，服务按原样运行。以前只在部分机器上装了 Yee 时，TP rank 之间的集合通信会错位卡住，召回统计停在很小的 `steps` 不再增长。
- `.pth` 入口（`yee install-hook`）也会装热专家验证预算（4.1.x 只有 sitecustomize 会装）。
- 热专家验证预算的控制文件与统计文件缺省在 `/state`，没有 `/state` 时在 `~/.cache/yee/`。
- 新命令 `python -m yee config`：列出每个开关的生效值与来源（env / default / off）。
- 发布包新增 `install.sh`：Mia dsv41 方案上自动找方案目录、所有机器都装、在 env 文件末尾追加只含 `PYTHONPATH` 与 `YEE=1` 的标记块；`--check` 只检查，`--restart` 重启并做健康检查与冒烟，失败自动回退，`--uninstall` 卸载。其它机器走 `pip install` + `yee install-hook`。

## 4.1.1

### 修 GB10 上服务加载自检误判：默认 dtype 是 bfloat16 时 0.8 变成 0.80078125

- 问题：GB10 机器服务加载时热专家验证预算模块（`yee/integrations/expert_budget.py`）自检失败，日志 `[yee-eb] geometry disabled: RuntimeError('self-check vs reference failed: switch state ctrl=[0.80078125, 4096.0, 1.0, 0.0, 0.80078125, 4096.0, 1.0, 0.0] sched=[6, 4611686018427387904]')`，该几何的热专家验证预算被禁用。
- 原因：SGLang 构造模型时 torch 默认 dtype 被设成 bfloat16，`_check_switch` 里写新控制槽的 `torch.tensor([B[0], ...])` 没写 dtype，变成 bf16，0.8 变成 0.80078125，`round(c[0],4)` 对不上 0.8。本机 RTX 3090 独立跑自检时默认 dtype 是 float32，所以之前没发现。
- 修复：`_check_switch` 里写新控制槽的 `torch.tensor` 显式 `dtype=torch.float32`；`--selftest` 图重放计划里的 `torch.tensor` 同样显式 `dtype=torch.float32`。
- 另一处：`_arm` 里的 JIT 预热改用未启用控制 `new_ctrl(1.0, 4096, 0, dev)`。预热后的恒等检查只在未启用时成立，初值控制可能已启用（启动即启用）。
- 新增回归测试：`bfloat16` 默认 dtype 下 `_quick_check` 返回 `None`、初值已启用时 `_arm` 不抛异常、不需要 GPU 的 `new_ctrl` 返回 float32。
- 装 4.1.1 而不设 `YEE_EXPERT_BUDGET=1` 时行为与 4.1.0 相同；热专家验证预算本身（算法、开关、实测结论）没改。

## 4.1.0

### 热专家验证预算（默认关）：主模型路由专家每步只读覆盖 p 比例路由权重的最热一批

- 新模块 `yee/integrations/expert_budget.py`：DSV4.1 在 Mia 镜像上的路由专家（`moe_b12x_next`，E=384）每次 MoE 调用按本步所有行的路由权重给专家记质量，只保留覆盖 `p` 比例质量的最热专家（可加 `kcap` 上限）；被丢的槽指向本行保留的专家、权重置 0，本行保留权重按比例放大使行和不变。选择每步跟着当前批次变。草稿头（E=128）不动。
- 只替换确定性路径里 `ep_remap` 那一次启动（单 CTA Triton 内核，可被 CUDA graph 捕获）。未启用时与 `ep_remap` 逐位一致；服务加载时自检 12 例与一次定点切换，不过就禁用。
- 开关：`YEE_EXPERT_BUDGET=1`（`sitecustomize_template.py` 装导入钩子）。控制文件 `{"seq","armed","p","kcap","at_calls"}` 不重启改档：带 `at_calls` 时内核在同一调用序号切换，4 个 TP rank 同步；不带则立即生效，只在空闲时用。rank 0 每 10 秒写统计（每桶调用数、裁前/裁后不同专家数、丢掉的质量）。详见 `yee doc expert-budget`。
- 4 台 DGX Spark 实测（2026-11-18，块长 7 static + `conf:0.1`，每档 468 个请求，对比同日两次基线）：p=0.90 合计 +17.3%（new 并发 1/4/8 +7.0%/+15.7%/+22.5%，copy c8 +25.4%），每层每步读的专家数 c1 16.7→11.6、c8 75.6→40.9；p=0.95 合计 +8.1%；p=0.80 +13.0%（接受数降得更多）。未启用 +0.9%。
- 装 4.1.0 而不设 `YEE_EXPERT_BUDGET=1` 时行为与 4.0.0 相同。
## 4.0.0

### 召回感知验证预算与折叠提议下的 STS 采集（方案 C 实测）：compact 7 少验的主因是召回刚命中时的预算滞后

- 问题：r705 的 compact 7 合计比现网慢 3.5%，copy 并发 8 慢 13.8%、rewrite 并发 1 慢 12.3%。compact 的 `HostConfidenceBudgetPlanner.compute_budget` 用两步前的 confidence 算本步验证 token 总数，召回刚命中的请求这一步仍按低置信度分预算，召回草稿被截短。
- 召回感知预算（`yee/integrations/sglang_dspark.py`，默认开，`YEE_RECALL_BUDGET=0` 关）：rank0 在 `_on_propose` 里记下本步带召回计划的请求池行（`Req.req_pool_idx`，整对象替换的 frozenset），包装 `compute_budget`，把这些行的 confidence 当 1.0 后再算预算。预算本来就由 TP0 广播，4 个 rank 一致。日志 `[recall] ... budget_calls=… budget_hot_rows=…`。
- STS 采集钩子（`YEE_STS_COLLECT_PATH=<路径前缀>`）：引擎自带的采集在 DSV4 上采不到（static 不建置信度头；折叠提议时跳过；关掉折叠提议后块长 7 的 eager 路径在置信度头里崩）。4.0.0 在 `DsparkStepObservers.observe_verify_step` 上挂钩，rank0 每步写 logit(confidence) 与接受数，跳过召回行，格式与 `sglang.benchmark.dspark_sts_fit` 兼容。
- 4 台 DGX Spark 实测（2026-11-18，r653 矩阵 468 个请求，前后两次现网对照）：4.0（compact 7 + r705 SPS 表 + r713 STS 表）合计 −1.9%，按 2026-11-05 真实流量加权 −1.1%；new 并发 8 +6.0%；copy c8 −5.5%、rewrite c1 −5.6%、mix c4 −5.4%。和 r705 的 compact 7 相比合计好 1.7%（copy c8 −13.8% → −5.5%，rewrite c1 −12.3% → −5.6%）。带图回归 16/16，PIT PASS。
- STS：8105 个非召回行拟合出温度 `[1.00, 1.12, 1.26, 1.58, 1.78, 0.79, 0.28]`；校准前 ECE 已只有 0.001～0.025，期望接受长度只低估 5%。STS 不是主因。
- static 模式下两个钩子都不触发，装 4.0.0 不改变现网行为；现网仍用 static。
## 3.9.0

### 解码单步耗时拆分工具（方案 B 实测）：先量清跨机同步占多少，再决定改不改

- 新工具 `python -m yee.tools.step_breakdown <trace 文件或目录...> --out <目录>`（只用标准库）：读同一次 torch profiler 采集的各 TP rank chrome trace（`.json` / `.json.gz`，rank 从文件名里的 `TP-<n>` 识别），按 verify CUDA graph 切步，给出每 rank 每步的墙钟、GPU 忙与空闲、通信（RoCE 单发 all-reduce、RoCE all-gather、NCCL、其他）与计算分类（MoE、稠密 GEMM、注意力、草稿图、其他）、通信 kernel 的调用次数与单次均值、暴露在关键路径上的通信（GPU 忙但没有计算 kernel 在跑的时间），以及跨 rank 的最慢 rank 与等待估计。输出 `breakdown.json` 与 `breakdown.md`，末行 `STEP_BREAKDOWN: ranks=N steps=M wall_ms=… comm_ms=… idle_ms=…`。
- 4 台 DGX Spark 实测（2026-11-17，块长 7 static，每请求验 8 行）：
  - 并发 1：每步 32.4 ms，通信 5.4～6.0 ms，其中暴露在关键路径上的 2.4～2.6 ms（约 8%）。RoCE 单发 all-reduce 每步 90 次、单次均值 44 µs；RoCE all-gather 每步 51 次、34～37 µs；NCCL 每步 0.1～0.3 ms。
  - 并发 8：每步 86.6 ms，通信 17.2～18.2 ms，暴露 13.4～14.3 ms（约 14%）；RoCE 单发 all-reduce 单次 132～139 µs。
- 结论：解码里 2 MiB 以下的 TP 集合通信已经走 RoCE 单发内核，NCCL 每步只占 0.1～0.85 ms，调 NCCL 的算法和协议最多省约 1%；即使跨机通信降到 0，并发 1 也只快约 8%，达不到方案 B「并发 1 快 10% 以上」的通过线，所以 3.9.0 不改任何运行时参数。
- 3.9.0 的引擎侧文件（`verify_cap.py`、`vcap_policy.py`）与 3.8.0 逐字节相同，现网不用重新部署。
## 3.8.0

### vcap 选宽度器影子模式：先在真实流量上估"每步换验证宽度"能快多少

- 问题：SGLang Mia（DeepSeek-V4.1）的验证矩形宽度固定为 [并发, 块长+1]。vcap 截掉的草稿行仍在矩形里算，只省了路由专家的读取：r653 回归 `step_ms = 31.6 + 2.70·并发 + 1.33·活行 + 0.50·死行`，门槛只能在活行 1.33 ms 和死行 0.50 ms 之间取舍，怎么调都只有 ±1% 的空间。tideLM 在同一模型上的做法是每个验证宽度一套 CUDA graph，每步选"期望出 token ÷ 该宽度实测步时"最大的宽度。这要改引擎，所以先用影子模式估收益：合计不到 3% 就不改。
- 做法：`mia-dsv41-vcap` 的 `vcap_policy.py` 新增选宽度器，控制文件写 `{"mode": "width"}` 开启。verify_cap.py 把它当 legacy 走，引擎每步的 k 与 `conf:0.1` 相同，只观测。
  - 起草前定宽度：rank0 每步只用此前已观测的结果，给候选宽度 W（默认 4 / 8，即块长 3 / 7；可以在 `widths` 里加 13，`max_bs` 默认让 13 只在 bs ≤ 4 时可选）算 E(W) 和 T(W)。E(W) 是整批期望出 token 数，用每个请求最近 16 步的接受数，按 n : 2 混入全局最近 256 行；T(W) 是该 bs 档现网宽度最近 32 步实测步时的中位数，加上行数变化 × 每行成本。选 E/T 最大的宽度，比当前宽度好 3% 以上才换（`switch`）。另有一个只在不超过现网宽度里选的 `le0` 选择器。
  - 结算：每步结果到 CPU 后，按本步实际的接受数和验证数算"假如本步用 W"的出 token 数和步时。W 不超过现网宽度时是精确的（同一份草稿截短）；W 更宽时，越过现网块长的部分按延续率估：召回行用该请求的逐位接受率 q_r，模型行用全局 rho_acc × rho_k（只统计非召回行在最后一位的接受率和继续验证率）。
  - 每行成本：默认用 r650 块长 3 / 5 / 7 的实测斜率（每请求每行，bs 1 / 2 / 4 / 8：活行 2.8 / 2.2 / 1.67 / 1.29 ms，死行 0.58 / 0.63 / 0.55 / 0.52 ms，已含起草开销的变化），中间的 bs 按 log2 插值；某个 bs 档还没有实测步时时，T(W0) 用 r653 回归。W 更宽时新增行的成本可乘 `hi_scale`（默认 1.0；r653 的 copy 并发 1 数据显示块长 12 新增的行约 1.9 ms/行，默认值偏保守）。这些都能在控制文件的 `width` 子配置里改。
- 输出：`vcap_stats.jsonl` 每个窗口多一个 `width` 字段，按 bs 档给出 `act`（现网实际）、`sel`（选宽度器）、`sel_lo`（sel 不计估算部分的下限）、`le0`、`oracle`（每步事后最优）、`static`（每个固定宽度）的 token 数与毫秒数，以及 `pick`（各宽度被选次数）、`pred`（预测与结算的 token 数）。新工具 `python -m yee.tools.vcap_width_report vcap_stats.jsonl [--tag T] [--since 时间戳] [--json]` 汇总这些窗口，打印各方案相对现网的提速和 `VERDICT: GO / STOP`（`--threshold` 默认 0.03）。
- 安全：控制文件不存在时与 `conf:0.1` 逐位一致；宽度代码不碰门槛、不用随机数，`adapt` 模式的输出与关掉宽度时逐位相同（单测与 3.7.0 原版对拍）。宽度代码出错只关掉宽度影子，stats 里出现 `width_error`，门槛策略照常。每步 CPU 开销：bs=16 时约 +185 µs（本机实测），现网 D 上 bs=1 时约 43 µs。
- 实测对照（r703，现网 D，r653 的压测矩阵：new / copy / rewrite / mix × 并发 1 / 4 / 8 × 温度 0 / 0.7 / 混温，468 个请求，同时段没有外部流量）：
  - 只在 4 / 8 里选（`le0`）估计合计快 3.8%（并发 1 / 2 / 4 / 8：+1.2% / +4.3% / +4.8% / +5.3%），几乎全部来自 new 类（并发 1 / 4 / 8：+5.7% / +8.4% / +16.5%），copy、rewrite、mix 在 ±2% 以内。固定用 4 的估算与 r650 实测同方向（new 并发 8：估 +16.7%，实测 +16%）。
  - 宽度 13 的估算明显偏高：固定用 13 估计合计 +9.8%，r653 实测块长 12 合计 −5.4%；分格看高 5～30 个百分点（new 并发 8：估 −15.7%，实测 −26.8%；mix 并发 4：估 +11.2%，实测 −11.9%）。原因是召回行越过块长后的延续率（按 q_r ≈ 0.99 外推）和多出来的死行成本都估低了。所以默认候选只留 4 / 8，`sel` 与 `le0` 相同、全部基于观测；加了 13 时，`sel` 里 13 的那部分只作参考，`sel_lo` 是不计它的下限。
- 限度：每行成本来自块长 3～7 的实测；草稿按"同一份草稿截短"处理，块长 3 时 DSpark 实际起草的前 3 个 token 可能略有不同；上面的数来自压测矩阵，真实流量的内容比例不同，结论要看真实流量上的统计。
- 测试：新增 `tests/test_vcap_width.py` 18 项（与 3.7 原版对拍 6 组、选择、滞回、结算口径、配置与异常隔离、verify_cap 联调、性能）、`tests/test_vcap_width_report.py` 11 项。`verify_cap.py` 与 3.5.0～3.7.0 相同。

## 3.7.0

### 新增提问哈希索引（qhash）：同一个问题再问一次，直接从上次那份回答开始起草

- 问题：召回一直是在“这次请求的内容”里找与历史记录重复的片段，所以同一个问题被改写（换了日期、换了会话 id、换了系统提示的开头）之后，往往找不到上一次那份回答，只能从头生成。
- 做法：请求正常结束时，把这次的回答（最多 16384 个 token）按提示的哈希存起来。哈希分 4 级：整段提示、提示末尾 512 个 token、末尾 128 个、末尾 32 个（提示本身不长于该级时跳过该级）。新请求开始时依次查这 4 级，命中就从第 1 个输出 token 起，用那份旧回答当草稿；输出与旧回答分叉之后，用最近 4 个输出 token 在这份旧回答里重新对齐，对上就接着出草稿。
- 草稿照常由目标模型逐 token 验证，命中错了只是这份草稿被拒，输出不变。qhash 与原有的两个来源（`own`、`record`）并列，走同一套按来源和档位的期望值选择；统计里 `tiers` 多出 `qhash_*` 三档。
- 存储：disk 后端时每个键一个文件 `<召回目录>/qhash/<键>.i4`（int32 小端），原子替换写入，每次查找直接读文件——所以多个进程共用同一个目录都能看到，重启后也还在；最多保留 8192 个键文件（按修改时间删最旧的）。memory / cuda 后端时只存在进程内存里。`yee clear`（清库）时一并清空。
- 开关：默认开 1，设 0 关闭，关闭后行为与 3.6.0 完全相同。yee 是 `RECALL_ACCEL_QHASH`（旧名 `SGLANG_D_RECALL_QHASH`、`DSV41_RECALL_QHASH`），vllm_d_recall 是 `VLLM_D_RECALL_QHASH`。
- 统计新增：`qhash`（是否开启）、`qhash_hits`（新请求命中次数）、`qhash_realigns`（分叉后重新对齐次数）、`qhash_puts`（写入次数）、`qhash_entries`（当前键数）。
- 限制：只有输出和旧回答逐 token 相同的部分能被接受。温度大于 0 时每次采样不同，命中后也只在回答一致的片段上加速；温度 0 且提示相同时效果最好。memory 后端重启即清空，要跨重启必须用 disk 后端（`RECALL_BACKEND=disk` 或不设，缺省就是 disk）。
- 只改 Python 召回核心（`yee/core.py`、`vllm_d_recall/core.py`）：llama.cpp、TensorFold、cuteafd、mlx-serve、strata 这些补丁基座没有 qhash。
- 测试：新增 `tests/test_recall_qhash.py`（8 项：同提示命中、换头同尾命中、分叉后重新对齐、回答末尾与过短回答、关闭开关、磁盘后端跨重启与多进程共享与清空、容量上限与新键覆盖旧键、4 级键的生成）。

## 3.6.0

### 修 bug：多个进程共用一个磁盘记录目录时，后打开的进程写不进去、也看不到别人后写的记录

- 现象：多副本、多组 DP 等多个进程指向同一个记录目录（`RECALL_ACCEL_DIR`，vLLM 是 `VLLM_D_RECALL_DIR`，strata 是 `--recall-dir`）时，只有第一个打开目录的进程写入；其它进程的 `add()` 直接返回 0，而且只在启动时读一次目录，之后别的进程写的记录都用不上。rtx3080n1 上 strata 3 组副本共用目录，r0 的统计里 records=0 就是它。
- 原因在模块本身，不是哪个基座接错了：`yee.core` 和 `vllm_d_recall.core` 的磁盘后端打开目录时用 `flock(LOCK_EX|LOCK_NB)` 选出唯一的 writer 并一直持锁；`RecallEngine.tick()` 不重读磁盘。strata、llama.cpp 召回补丁里的 C++ 版是同一套逻辑的移植。只有一个进程用召回的部署（例如 4 台 DGX Spark的 SGLang，memory 后端，召回只在 rank0）碰不到。
- 修法：新开关 `RECALL_ACCEL_SHARED`（也认 `SGLANG_D_RECALL_SHARED`、`DSV41_RECALL_SHARED`；vLLM 是 `VLLM_D_RECALL_SHARED`），缺省 1。不再长期持锁：追加时取一小会儿排他锁，先把别的进程新写的 id 读进来，再接着写；`RecallEngine.tick()` 每 2 秒在共享锁下读一次别的进程新写的 id（在后台线程里做，不挡调度线程）。清空时改写目录里的 `generation` 文件，其它进程下次同步时整份重读。设 0 完全回到 3.5.2 的行为。
- 和旧进程混用：目录被一个一直持锁的进程（3.5.x，或设了 0）占着时，共享模式的进程等 10 秒后退回旧的只读行为，不会每次追加都卡住。共用一个目录的进程要一起升级，并用同样的容量（`RECALL_ACCEL_GIB`）。只用于同一台机器的本地文件系统，不要跨机器用 NFS 共享。
- strata：`strata-recall` 补丁里的 `src/spec/yee_recall.cpp` 换成修好的版本（开关 `STRATA_RECALL_SHARED`，缺省 1），并带双进程单测 `tests/yee_recall_shared_test.cpp`；补丁的其它部分不变。
- 没改：llama.cpp 召回补丁（`llama.cpp-recall`）的磁盘记录仍是单 writer，多个 llama-server 共用一个 `--recall-dir` 时只有第一个写入。
- 测试：新增 `test_recall_shared.py`，用真进程测两进程互相读到、单 writer 回退、4 进程并发写、清空、驱逐、`tick()` 触发同步、与旧版进程混用，`yee.core` 与 `vllm_d_recall.core` 各跑一遍（需要 fcntl，Windows 上跳过）。

## 3.5.2

### 新增 llama.cpp MTP 草稿头截断词表补丁与测速工具，见 [llama-mtp.md](llama-mtp.md)

- `llama.cpp-mtp-vocab`（git-patch，基线 ggml-org/llama.cpp 提交 `a3b9c23`；补丁的两处锚点在 v0.9.0 `cd959bae` 里也各出现一次）：只改 `src/models/qwen35.cpp` 的 MTP 图。`YEE_MTP_VOCAB_N=N`（0 < N < 词表大小）时输出头只取输出权重前 N 行做矩阵乘，logits 用 -1000 补回整个词表宽度，主机侧缓冲、拷贝、采样器不变；不设或 0 时与上游逐位一致；输出头带逐行缩放（`output_s`）时不截断。
- `yee/tools/mtp_bench`：`run_matrix.sh`（每个配置单独起 llama-server，15 条提示，出对比表与逐条输出比对）、`bench.py`、`compare.py`、`contention.sh`（CPU/GPU 内存争用）、`vocab_coverage.py`（生成 token 的 id 分布）、`mock_llama_server.py`。
- mx4 实测（Ryzen AI Max+ 395，Radeon 8060S，Vulkan，Qwen3.6-27B Q4_K_M）：无投机 12.45 tok/s；MTP 草稿在 GPU、k=3 是 24.29，加 N=131072 是 25.50（+5.0%）；草稿在 CPU、k=3 是 20.60，加 N=131072 是 20.70；GPU 解码时 CPU 解码从 6.30 掉到 1.79 tok/s。所有 k=3 配置的输出与原版逐字一致。
- 测试：`tests/test_mtp_vocab.py`（清单、补丁文本、在 a3b9c23 的 `qwen35.cpp` 上 check / apply / 已应用识别 / 回退）、`tests/test_mtp_bench.py`（文件齐全、编译、用 mock 服务跑两次并比对出 1 条不一致）。

## 3.5.1

### 新增 strata 引擎召回补丁，见 [patches.md](patches.md)

- `strata-recall`（git-patch，基线 Niko1221/Strata 提交 `712fec6b`，engine 0.1.59；也能干净打在 rtx3080n1 的 `strata-0.1.59-expert-server.patch` 之上）：新增 `include/strata/spec/yee_recall.hpp`、`src/spec/yee_recall.cpp`（召回库，llama.cpp 补丁 `common/recall.cpp` 的移植，命名空间 `yee`）、`include/strata/spec/yee_recall_strata.hpp`、`src/spec/yee_recall_strata.cpp`（引擎侧：每个进程一个记录，每个请求一个流，按槽号区分）、`third_party/nlohmann/include/nlohmann/json.hpp`（3.11.3，MIT）和两个单测；修改 `cmake/backends/cpu.cmake`（`strata_spec` 加两个源文件）与 `src/program/generate.cpp`（参数、单流窗口、批处理窗口）。
- 单流窗口里召回提议像 prompt-lookup 草稿一样进窗口（`from_sfx`）；`--batch-mtp` 的串行批处理窗口里替换该槽的 MTP 草稿，行数不超过 `--spec`（槽草稿器能追上的行数）与窗口剩下的行数。验证后按接受数回报（`feedback` / `observe_own`）。
- 引擎没有词表文本，记录身份用模型名加词表大小算，不与 Python 基座、llama.cpp 的记录共用。
- KAD 不移植：strata 自带逐窗口自适应（`--spec-min-p` 截断 MTP 草稿 + DraftPolicy）。
- 测试：召回库单测 18 项、适配层单测 25 项（本机 zig 0.17 / clang 22）；`generate.cpp` 以 Linux 目标、CUDA 12.9 头做 host 侧语法检查通过；CMake CPU 配置通过，`strata_spec` 的源文件全部编译通过。没有在 GPU 上编译和实测（构建时 rtx3080n1 连不上）。

## 3.5.0

### vLLM（`vllm_d_recall`）：门槛策略、非贪心召回、影子统计

- KAD 策略新增 `mode=threshold`：每个请求、每个位置的条件接受率按温度档（T0、T(0,0.5]、T(0.5,0.9]、T>0.9）的先验起步，再用滑动平均更新；整批用 Dinkelbach 法求每个请求的宽度，`budget` 可限制整批送验总数。字段与缺省：`c_ms` 0.49、`base_ms` 40、`p_ms` 0.30、`wmin` 1、`wmax` 12、`ema` 0.2、`budget` 0、`priors`；无效值用缺省并打印原因。缺省策略仍是 ladder。
- `VLLM_D_RECALL_NONGREEDY=1`（缺省 0）：温度大于 0 的行也接受召回草稿，被覆写的位置把草稿分布改成单点分布，修正失败时退回本步原草稿。只改 token 时输出分布的总变差是 0.39～0.90，改成单点分布后 8/8 个场景卡方检验通过。
- `VLLM_D_RECALL_SHADOW=1`（缺省 0）：影子统计，环形缓冲加后台线程写 JSONL（`VLLM_D_RECALL_SHADOW_DIR`，缺省临时目录下的 `vllm_d_recall_shadow`；`VLLM_D_RECALL_SHADOW_TOPK_RATE` 缺省 0.01），不改行为。
- 以上新增 34 项单元测试；没有在 vLLM 生产上实测。

### 两个 SGLang Mia 镜像补丁，见 [patches.md](patches.md)

- `mia-dsv41-vcap`（source-drop）：把 `verify_cap.py`、`vcap_policy.py` 放进容器 PYTHONPATH 最前面的目录，遮蔽镜像内的 `/opt/dsv41/adapter/verify_cap.py`。控制文件 `/state/vcap_ctl.json`（`DSV41_VCAP_CTL` 可改）不存在时与 `DSV41_VERIFY_CAP=conf:0.1` 逐位一致；`mode=adapt` 时 rank0 每步在 CPU 上给每个请求、每个位置算门槛（load / req / pos / recall / temp 五条逻辑加 3% 探索）。4 台 DGX Spark A/B：吞吐与固定 conf 持平（块长 7 是 0.998 倍，块长 12 是 1.004 倍）；块长 12 下 PIT 分布检验 12 组通过。
- `mia-dsv41-image-budget`（source-drop）：替换 `srt/multimodal/deepseek_v41_image_processing.py`，只接受镜像 `dsv41-4x-spark` 的原件（sha256 `dcd06c489aa1…`）。配 `XY_DSV41_IMAGE_BUDGET_RESERVE=1` 后 D 的图片 token 预算减 1，与 vLLM 的 P 一致：3024x588 的图两端都是 886 个 token（不打时 D 是 1024，PD 请求报 500）。带图回归 16/16。
- `yee patch` 的 source-drop 新增两个可选字段：文件条目的 `orig_sha256`（目标原件不是它就拒绝）、补丁的 `remove_after`（拷完后删掉的 glob，例如过期的 .pyc）。

### 生产实测结论写进 heads.md

- DeepSeek-V4.1-Flash + SGLang Mia DSPARK（4 台 DGX Spark，PD 分离）：块长 7 + `DSV41_VERIFY_CAP=conf:0.1` 比块长 12 快。块长 12 static 合计 −5.4%，compact 变长验证 −8.9%，灌库 8 轮后 −5.3%，长输入长输出 −12.1%。这条路径的验证矩形整批固定，KAD 是 native。

## 3.3.0

### 新增 llama.cpp 召回补丁，见 [patches.md](patches.md)

- 新增补丁基座 `llama.cpp-recall`（git-patch）：基线 ggml-org/llama.cpp v0.9.0 commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8` **加上已打的 `llama.cpp-kad/0001-kad.patch`**，**必须在 `llama.cpp-kad` 之后打**。
- 补丁内容：新增 `common/recall.h`、`common/recall.cpp`（`yee.core` 的 C++ 移植：跨请求复制草稿，记录可放内存也可落盘），另有 `common/CMakeLists.txt`、`common/common.h`、`common/arg.cpp`、`common/kad.h`、`common/kad.cpp`、`tools/server/server-context.cpp` 的锚点编辑。
- 作用：llama-server 读过的 token id 记下来，新请求与历史内容重复时把后续内容复制成草稿，交给 llama-server 正在用的加速头核对；带草稿模型时召回的提议替换草稿头的草稿，草稿头状态机照常走；不带草稿模型时召回有提议就投机核对，没有就普通解码。
- 开关：`--recall`、`LLAMA_ARG_RECALL=1`、`RECALL_ACCEL=1`、`YEE=1` 或 `YEE_RECALL=1`；参数 `--recall-dir`、`--recall-read-only`、`--recall-gib`、`--recall-n`。
- 各草稿头 × 召回的状态：draft-simple 与不带草稿头的无草稿模式 verified（WSL CPU 真跑小模型），eagle3 / mtp / dflash / dspark / ngram built（编译通过，没有对应模型，未实跑）。
- 3.2.0 的全部功能（KAD 覆盖全部基座、三个 KAD 补丁）行为不变。

## 3.2.0

### KAD 扩到全部基座，见 [kad.md](kad.md)

- 3.1.0 的 KAD（逐请求自适应草稿数，最大 12）只在 vLLM 适配器 `vllm_d_recall` 上；3.2.0 把同一套原理和同一份策略文件扩展到所有基座：`K_max` 统一为 12，默认 ladder 策略不变（base 5、kmin 5、kmax 12、全接受 +2、没全接受降到接受数 +1）。
- 新增 Python 核心 `yee.kad`：`parse_policy`、`ladder_update`、`decide_k` 与 vLLM 生产适配器 `vllm_d_recall.integrations.vllm_kad` 的 `parse_policy`、`ladder_update`、`decide_w` 逐项一致；`Kad` 类提供 `begin_step`、`width` / `widths` / `batch_width`、`observe`、`mark_hot`、`release`、`snapshot` 等接口，策略文件按 (mtime_ns, size) 热加载，最小间隔 1.0 秒。
- 新增适配器（`yee.integrations`）：`sglang_kad`、`trtllm_kad`、`lmdeploy_kad`、`mlx_lm_kad`、`hf_kad`，以及自动装配模块 `kad_auto`。
- 新增自动加载：`YEE_KAD=1` 时，`yee install-hook` 写的 `.pth`（`import yee.autoload`）在解释器启动时装配 `kad_auto`；它在各引擎模块被导入的那一刻装上对应适配器，装不上只打印一行原因，不抛异常。触发模块与适配器的对应：`sglang.srt.speculative.eagle_worker_v2` → `sglang_kad`，`tensorrt_llm._torch.pyexecutor.py_executor` → `trtllm_kad`，`lmdeploy.messages` → `lmdeploy_kad`，`mlx_lm.generate` → `mlx_lm_kad`，`transformers.generation.utils` → `hf_kad`。
- vLLM 侧不变：仍然用自己的开关 `VLLM_D_RECALL_KAD`（`vllm_d_recall`），不走 `YEE_KAD` 自动加载。

### 新增三个源码补丁，见 [patches.md](patches.md)

- `llama.cpp-kad`（git-patch）：基线 ggml-org/llama.cpp v0.9.0 commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8`；llama-server 按 slot 给草稿数 5～12。
- `tensorfold-kad`（git-patch）：基线是 ashhart/TensorFold commit `33ada77c3122fca844a6825316ce8bbabaca05be` 加上已打的 `tensorfold/0001-recall-after-priced.patch`，**必须在 `tensorfold` 之后打**。
- `mlx-serve-kad`（source-drop）：基线是已拷上 `mlx-serve-2.4` 的 M5 xypd 线上源码，**必须在 `mlx-serve-2.4` 之后拷**；替换 `src/engines/mlx/spec/` 下的 6 个文件。

### 各基座状态（照各路 NOTES）

| 基座 | 做法 | 状态 |
|---|---|---|
| vLLM（`vllm_d_recall`） | 全部草稿头的逐请求自适应 K，K_max 12 | 3.1.0 已有，本版不变 |
| SGLang（`sglang_kad`） | EAGLE / EAGLE3 / NEXTN / MTP 别名 / STANDALONE 复用 Mia 自带 adaptive 运行时，按批取 K_b = max k_i 并向上取到已捕获步数 | source-checked |
| SGLang 的 DSPARK / DFLASH / NGRAM / UNO / MULTI_LAYER_EAGLE | 块长或送验宽度整批固定，没有可切换的多套 graph | native |
| SGLang 上游 0.5.10.post1 | 没有按批或逐请求变草稿步数的机制 | unsupported（安装时打印原因，不装） |
| llama.cpp（`llama.cpp-kad`） | `draft-simple` 每 slot 草稿上限取 KAD 宽度；`eagle3`、`mtp` 的起草循环提前停止 | `draft-simple` verified（WSL CPU 真跑小模型）；`eagle3` / `mtp` built（编译通过，没有对应模型，未实跑）；`ngram-*` native |
| TensorRT-LLM（`trtllm_kad`） | PyTorch 后端且配置了 `draft_len_schedule` 的投机方法，每轮长度 = min(schedule 值, K_b) | source-checked |
| TensorRT-LLM 静态 / chain 草稿循环 | 引擎断言 `draft_len_schedule is None`，不能逐轮改长度 | native |
| TensorRT-LLM 一模型 EAGLE3 / MTP | 未配置 schedule 时长度固定 | 视配置而定 |
| LMDeploy（`lmdeploy_kad`） | 引擎只有一个固定的 `num_speculative_tokens` | native |
| mlx-lm（`mlx_lm_kad`） | `speculative_generate_step` 每轮草稿数 = min(KAD 宽度, 启动 K) | verified（WSL Linux CPU，mlx 0.32.4 真跑） |
| transformers（`hf_kad`） | 辅助生成（AssistedCandidateGenerator 及其子类）与 prompt lookup 每轮宽度取 KAD 宽度 | verified（CPU torch 真跑） |
| transformers 的 MTP、DFlash | 草稿数由模型结构定、一次前向出全部草稿 | native |
| TensorFold（`tensorfold-kad`） | lane 引擎的 head 草稿深度取 min(·, 该 stream 的 KAD 宽度)；copy / Recall 窗口长度不截 | built |
| mlx-serve（`mlx-serve-kad`） | DSpark stage 轮送验数、组轮 stage 槽可选行数取 KAD 宽度 | source-checked |
| mlx-serve 的 DSpark 树轮 | 不改 | native |
| cuteafd | 不加 KAD，原生跑 | native |

### 为什么 cuteafd 与 LMDeploy 是 native

- **cuteafd**：启动 K 到不了 12。`--dspark-draft-limit` 的取值范围是 1～7（默认 5），草稿器只有 5 宽和 7 宽两套训练权重，核对上限 `MAX_DSPARK_PROPOSALS = 7`，所以 KAD 的每请求上限只能是 min(12, K) ≤ 7；默认 K = 5 时 KAD 没有可调空间，K = 7 时 cuteafd 自己已经按每请求置信度和带宽成本选长度，KAD 的 ladder 只覆盖这套更细的定价，没有额外收益。
- **LMDeploy**：0.18.0 的每请求 token 数、输入形状、CUDA graph 和草稿循环都由常量 `num_speculative_tokens` 构建（`num_spec_tokens + 1`、`num_spec_tokens - 1` 次草稿前向等），逐请求改宽度要动 6 处以上的形状逻辑，KAD 省不了算力，所以只打印一行原因。

### 测试

- 包内 `pytest tests yee/selftests`：**238 passed**（3.1.0 是 172 passed）。
- `python -m yee selftest`：**PASS**。
- 各路基座自带验收：`VERIFY_SGLANG: PASS`、`VERIFY_TRT_LMDEPLOY: PASS`（29/29）、`VERIFY_LLAMACPP: PASS`、`VERIFY_MLXLM_HF: PASS`、`VERIFY_MLXSERVE: PASS`（10/10）；TensorFold 的 `verify_tensorfold.py` 在 WSL 里跑构建与单测。

## 3.1.0

### KAD 覆盖 vLLM 全部草稿头，见 [kad.md](kad.md)

- vLLM 适配器 `vllm_d_recall` 的 KAD（逐请求自适应草稿数）从只支持 DSpark 扩到 vLLM 的全部草稿头，`K_max` 统一为 12。
- 原理不变：启动时 `num_speculative_tokens = 12`，调度器按每个请求最近的核对接受情况在 5～12 之间决定本步送验数，并截短 `request.spec_token_ids`；默认 ladder 策略不变（base 5、kmin 5、kmax 12、全接受 +2、没全接受降到接受数 +1）。
- 草稿侧按头分三种路径：dspark 走变长块路径（每个请求只生成 `w_i` 个）；eagle、eagle3、mtp、draft_model 走 steps 路径（本步只跑 `max(w_i)` 步）；dflash（含 DFlash2、LiLiCorr）与多模块 MTP 原生出 12 个，由调度器按请求截短。
- ngram、ngram_gpu、suffix、medusa、mlp_speculator、custom_class 在 vLLM 0.31 的 V2 runner 下不支持（`init_speculator` 抛 `NotImplementedError`），要用 V1 runner（`VLLM_USE_V2_MODEL_RUNNER=0`），此时只有调度器侧截短生效。
- 结构化输出请求不再截短：3.1.0 起给它们发满宽 `w = K_max`，保证 worker 生成的草稿覆盖下一步送验的全部位置。
- 启动 K 必须是 12；MTP 系 `n_predict` 不整除 12 时 vLLM 报 `num_speculative_tokens:12 must be divisible by n_predict=...`，此时取不超过 12 的最大合法值；Qwen3-Omni DSpark 要求等于训练块长。
- 新日志：`[vllm-d-recall] kad(worker) head=<类名> method=<method> k=<K> path=<varlen|steps|native>`；K ≠ 12 时多一行启动告警；worker 统计行末尾新增 `steps=.. mean_n=.. steps_hist=..`；回退原因新增 `native_block`、`k1`、`cudagraph`、`dp`、`pcp`、`adaptive`、`watermark`。
- `vllm-d-recall check` 新增两项可选检查：`AutoRegressiveSpeculator.propose`、`PlainDraftModelSpeculator.propose`（缺失只打印 WARN）。

## 3.0.0

### 统一版本号

- 生产 vLLM 适配器 `vllm_d_recall` 原来是独立版本线（1.0.0 线上 → 1.1.0 HK → 1.2.0 KAD），从 3.0.0 起随 Yee 一起发，`vllm_d_recall.__version__` 也是 3.0.0。
- wheel 里新增 `vllm_d_recall`、`vllm_d_recall.integrations` 两个包。
- Yee 2.4（M5 冷层）以补丁基座 `mlx-serve-2.4` 入包，见下文。

### KAD：逐请求自适应草稿块宽，见 [kad.md](kad.md)

- 以 `num_speculative_tokens = K_max`（12）启动一次，之后每一步每个请求各自选块宽。
- 默认策略是 ladder（1.2.0 的默认是 fixed 5）：

```json
{"mode": "ladder", "base": 5, "kmin": 5, "kmax": 12, "up_after": 1, "up_step": 2, "down": "accepted", "down_slack": 1, "down_step": 2, "hot_k": 12, "hot_ttl": 8}
```

- 策略文件热加载，约 2 秒生效，不用重启。
- 代码与生产实测过的 1.2.0 相同，只改了版本号和默认策略。

### r629 生产对比（相对 K=5，折算后速度）

- 照抄型 greedy 请求：1.61～1.92 倍。
- 新内容和温度 0.7：0.950～1.035，95% 区间都包含 1。
- 固定块宽 5 时模块开销 0.974～1.075，errors 0。

### Yee 2.4 冷层：补丁基座 `mlx-serve-2.4`

- 4 个文件：`src/engines/mlx/spec/` 下的 `d_recall.zig`、`d_recall_live.zig`、`d_recall_cold.zig`（新），以及 `src/env.zig`。基线是 M5 上带 Yee 2.3 的线上源码。
- 靠 `MLX_SERVE_D_RECALL_COLD_DIR` 开启，不设或为空时行为与 2.3 相同。
- M5 A/B：聚合 decode 平均 −0.4852%，命中率 +4.79 / +10.13 / +4.36 个百分点，0 错误。见 [mlx-serve.md](mlx-serve.md) 第 8 节。

### 兼容

- 环境变量名和记录目录都不变。
- 自适应 K 只在 vLLM DSpark 适配器 `vllm_d_recall` 上有；SGLang、TensorFold、cuteafd、mlx-serve 在 3.0.0 里没有。
- 已有补丁基座 `tensorfold`、`cuteafd`、`mlx-serve` 的内容和行为不变。

## 2.3.0

### 多分支树草稿（P4），见 [tree.md](tree.md)

- 新模块 `yee.tree`：候选去重、多链合并成 token 树（共享前缀算一次，兄弟互斥，按节点预算剪枝且保留祖先）、SGLang EAGLE 树数组的双向转换（`to_sglang` / `from_sglang` / `sglang_full_mask`）、NGRAM trie 插入辅助、树验证参考实现（贪心与点质量采样 `verify_walk`，带 q 的 SpecInfer 多轮拒绝采样 `verify_specinfer`）、链式基座的选择器 `choose_chain`、批量分支验证的方案选择 `choose_plan`（与 mlx-serve 的 `d_recall.choosePlan` 同一个模型）。
- 召回核心：`RecallRecord.lookup_all()` 列出同一上下文的全部命中（热区、冷环、各短窗口），`RecallEngine.candidates()` 去重后返回全部候选，`RecallEngine.propose_select()` 从全部候选里选期望落地最多的一条。三者都不改 `lookup` / `propose` 原有的行为与计数。
- SGLang 全部链式草稿头（DSPARK、DFLASH、EAGLE 系 topk=1、NGRAM）和 vLLM V2（DSpark、DFlash、EAGLE、MTP）的计划阶段改走 `yee.tree.propose`：`YEE_TREE=1`（默认）时用 `propose_select`，`=0` 时与 2.2.0 相同。
- `yee bases --json` 每行新增 `tree`（native / select / none）和 `tree_note`。
- 新开关 `YEE_TREE`、`YEE_TREE_K`、`YEE_TREE_NODES`（`RECALL_ACCEL_*` 等别名同样生效）。
- mlx-serve（M5）r598 的分支状态树轮在 Zig 里实现，表里登记为 `native`；源码不在本包的补丁里（补丁仍是 r597 的 4 个文件）。

### 测试

- 新增 `test_tree.py`：去重、树合并与剪枝的祖先闭包、SGLang 数组转换的往返和手算对拍、贪心/点质量/SpecInfer 三种验证的一致性蒙特卡洛（TV < 0.02）、单候选时选择器退化、`propose_select` 与 `propose` 在单候选下逐项一致、`yee bases` 的 tree 字段。

## 2.2.0

### 改名为 Yee

- 发行包名 `yee`，wheel 为 `yee-2.2.0-py3-none-any.whl`。
- Python 包名 `yee`，命令 `yee`。

### 兼容层

- `recall-accel` 命令保留，与 `yee` 是同一个程序。
- `recall_accel` 包保留：`import recall_accel` 得到的就是 `yee`，`recall_accel.<子模块>` 与 `yee.<子模块>` 是同一个模块对象，旧部署里写死的 `import recall_accel.autoload` 照常工作。
- 旧环境变量（`RECALL_ACCEL_*`、`SGLANG_D_RECALL_*`、`VLLM_D_RECALL_*`、`DSV41_RECALL_*` 等）全部继续生效。
- 记录目录默认位置不变（`~/.cache/recall_accel/...`），已有记录直接可用。

### 新的环境变量前缀 YEE_

- `YEE_XXX` 等价于 `RECALL_ACCEL_XXX`，`YEE` 等价于 `RECALL_ACCEL`；两者都设时以 `RECALL_ACCEL_*` 为准。见 [env.md](env.md)。

### 钩子

- `yee install-hook` 写 `yee.pth`（内容 `import yee.autoload`）。
- `yee uninstall-hook` 同时删除 2.1 的 `recall_accel.pth`。

### TensorFold

- 补丁 `tensorfold/0001-recall-after-priced.patch`（基线 ashhart/TensorFold `33ada77c3122fca844a6825316ce8bbabaca05be`）：priced 提议器给不出草稿时改问 Recall，由 `TF_RECALL_AFTER_PRICED=1` 开启，不设时与上游一致。
- `yee.integrations.tensorfold.launch_env()` 返回 `{"TF_RECALL_AFTER_PRICED": "1"}`。
- `yee auto` 检测到 TensorFold 时多输出一行 `export TF_RECALL_AFTER_PRICED=1`；`yee tensorfold args` 的标准输出仍只有启动参数，这一行打印到标准错误，`$(yee tensorfold args)` 的用法不受影响。

### cuteafd 和 mlx-serve 补丁入包

- cuteafd（DeepSeek V4.1 native serve）：补丁 v1、v2（默认 v2）及说明 NOTES。
- mlx-serve（M5 xypd 版）：`src/engines/mlx/spec/` 下 4 个替换文件。
- 所有补丁文件的 sha256 写在 `yee/patches/MANIFEST.json`，应用前强制核对。

### 新命令

- `yee doc [topic]`：内置中文文档，不带参数列出全部 topic。
- `yee bases [--json]`：基座 × 加速头支持表。
- `yee patch list | show | path | apply`：见 [patches.md](patches.md)。

### 测试

- 新增 `test_yee_pkg.py`（兼容层模块同一性、YEE_ 变量别名、清单 sha256、git-patch 与 source-drop 应用流程、`yee doc`、`yee bases`、TensorFold 启动环境），已接入 `yee selftest`。
- pytest 只收集 `test_*` 函数，测试文件里的 `test(fn)` 装饰器不再被误当成测试。

## 2.1.0 及更早

见 2.1.0 的说明和验收记录。2.1.0 的召回核心、SGLang / vLLM 适配器、非贪心支持在 2.2.0 中没有行为变化。
