# Yee 4.2.0 文档目录

这一页是 Yee 内置文档的目录，`yee doc` 不带参数时列出的就是下面这些 topic。

## 0. 最短用法

4.2.0 起所有加速开关默认打开，不用设任何环境变量。显式设 0 可以单独关掉某一项；设 `YEE=0`（或 `RECALL_ACCEL=0`）整个 Yee 都不加载。

**Mia dsv41-4x-spark**：在 head 机上

```bash
bash install.sh --check      # 只检查
bash install.sh --restart    # 装到 4 台机器，装完重启并做健康检查和冒烟
```

回退：`bash install.sh --uninstall --restart`。

**其它 Python 推理服务**：

```bash
pip install yee-4.2.0-py3-none-any.whl && python -m yee install-hook
```

然后照常启动服务。

**怎么确认生效**：`python -m yee config`（打印每个开关的生效值与来源：env / default / off）；服务日志里看 `[yee] 4.2.0 defaults:`、`[recall] patched ...`、`[recall] rank check on`、`[recall] steps=...`（数字随请求增长）、`[yee-eb] armed geometry`。

**怎么关**：单项设 0（例如 `RECALL_ACCEL=0`、`YEE_KAD=0`、`VLLM_D_RECALL=0`、`YEE_EXPERT_BUDGET=0`）；整体设 `YEE=0`（或 `RECALL_ACCEL=0`）。

## 1. 目录

| Topic | 文件 | 一句话说明 |
|---|---|---|
| `readme` | [README.md](README.md) | 总览：Yee 是什么、4.2 最短用法、各基座快速开始、与 2.1 的兼容说明 |
| `index` | [index.md](index.md) | 本页：全部 topic 的目录 |
| `quickstart` | [quickstart.md](quickstart.md) | 从零到跑起来的完整步骤：安装、装钩子、验收 |
| `sglang` | [sglang.md](sglang.md) | SGLang 各加速头的接法、环境变量、Mia 镜像的 sitecustomize 模板 |
| `vllm` | [vllm.md](vllm.md) | vLLM V2 runner 的接法、投机配置、已知限制 |
| `tensorfold` | [tensorfold.md](tensorfold.md) | TensorFold 自带 Recall 的用法、Yee 补丁、哪些引擎不读 Recall |
| `cuteafd` | [cuteafd.md](cuteafd.md) | cuteafd（DeepSeek V4.1 native serve）补丁：开关、统计、限制 |
| `mlx-serve` | [mlx-serve.md](mlx-serve.md) | mlx-serve（M5 xypd 版）补丁：替换哪些文件、上线与回退；Yee 2.4 冷层补丁 mlx-serve-2.4；3.2.0 的 KAD 补丁 mlx-serve-kad |
| `heads` | [heads.md](heads.md) | 基座 × 加速头支持矩阵，含不支持项和原因 |
| `nongreedy` | [nongreedy.md](nongreedy.md) | 非贪心（采样）请求怎么用召回草稿，各基座怎么做到、怎么开关 |
| `sharing` | [sharing.md](sharing.md) | 全局共享的含义：谁能看到什么、隐私怎么管 |
| `env` | [env.md](env.md) | 全部环境变量表：名称、默认值、作用、回退顺序，含 4.2 的自动默认值与 `python -m yee config` |
| `patches` | [patches.md](patches.md) | `yee patch` 用法、各补丁的基线与回退方法（含 mlx-serve-2.4、3.2.0 的 tensorfold-kad / llama.cpp-kad / mlx-serve-kad、3.3.0 的 llama.cpp-recall、3.5.0 的 mia-dsv41-vcap / mia-dsv41-image-budget、3.5.1 的 strata-recall、3.5.2 的 llama.cpp-mtp-vocab） |
| `llama-mtp` | [llama-mtp.md](llama-mtp.md) | llama.cpp 的 MTP 草稿头：放 CPU 与放 GPU 的实测、截断词表补丁、测速与争用工具（3.5.2） |
| `troubleshooting` | [troubleshooting.md](troubleshooting.md) | 常见故障、定位步骤、恢复办法 |
| `changelog` | [changelog.md](changelog.md) | 各版本变化（4.2.0：所有加速开关默认打开、Mia 场景自动补默认值、跨 rank 一致性检查、`python -m yee config`、一键安装 `install.sh`；4.1.1：修 GB10 上 bfloat16 默认 dtype 导致的热专家验证预算自检误判；4.1.0：热专家验证预算 expert_budget；4.0.0：召回感知验证预算、STS 采集钩子；3.9.0：解码单步耗时拆分工具 step_breakdown；3.8.0：vcap 选宽度器影子模式；3.7.0：提问哈希索引 qhash；3.6.0：多个进程共用磁盘记录目录（修 bug）；3.5.2：llama.cpp MTP 草稿头截断词表与测速工具；3.5.1：strata 引擎召回补丁；3.5.0：vLLM 门槛策略与非贪心召回、两个 SGLang Mia 补丁；3.3.0：新增 llama.cpp 召回补丁；3.2.0：KAD 扩到全部基座、新增三个补丁；3.1.0：KAD 覆盖全部草稿头；3.0.0：统一版本号、KAD、2.4 冷层入包；2.3.0：多分支树草稿） |
| `tree` | [tree.md](tree.md) | 多分支树草稿（2.3.0）：原理、各基座支持表、开关、已知限制 |
| `kad` | [kad.md](kad.md) | 逐请求自适应草稿数（3.2.0 起覆盖全部基座，K_max 12）：开法、各基座路径、策略、统计、回退 |
| `expert-budget` | [expert-budget.md](expert-budget.md) | 热专家验证预算（4.1.0 引入、4.1.1 修自检、4.2.0 默认开）：原理、开关、不重启改档、统计 |

## 2. 怎么读

- 第一次用：先看 [quickstart.md](quickstart.md)，再按你的基座看 [sglang.md](sglang.md) / [vllm.md](vllm.md) / [tensorfold.md](tensorfold.md) / [cuteafd.md](cuteafd.md) / [mlx-serve.md](mlx-serve.md)。
- 想知道「我这个加速头支不支持」：看 [heads.md](heads.md)。
- 想开逐请求自适应草稿数（KAD）：看 [kad.md](kad.md)；各基座的开关与启动参数也在里面。
- 想在采样请求上也用：看 [nongreedy.md](nongreedy.md)。
- 想看每个开关当前生效值：跑 `python -m yee config`。
- 上线前评估隐私影响：看 [sharing.md](sharing.md)。
- 要打补丁或回退：看 [patches.md](patches.md)。
- 出问题了：看 [troubleshooting.md](troubleshooting.md)。
