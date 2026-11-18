# HaloLab Yee 加速模块 4.3.0

Yee 是一个面向大模型推理服务的投机解码加速模块。它把服务读过、写过的 token 记下来，新请求和历史内容重复时，直接把后续内容复制成草稿，交给服务正在用的加速头（草稿头）去验证。每个复制来的 token 都由目标模型逐个核对，命中得越多，解码步数越少。

除召回外还带了：逐请求自适应草稿数（KAD）、提问哈希索引（qhash）、热专家验证预算、逐请求 conf（vcap），**全部默认打开**，不需要设任何环境变量，也不改你的启动参数。

4.3.0 新增**部署诊断**：`python -m yee doctor` 认得 15 个推理引擎，能扫本机进程、docker 容器、ssh 远端、K8s 和服务 URL，告诉你每个实例现在能不能加速、要改哪个参数、要打哪个补丁，改法直接给可复制的命令。安装脚本也不再只认 Mia 方案，任何部署都能一键装。

- 下载：[Releases · v4.3.0](https://github.com/Soulmate-Halo/HaloLab-Yee/releases/tag/v4.3.0)
- 一键安装包 `Yee-4.3.zip`：1,140,415 字节，sha256 `673da46f721a4b51323e2e5e7bd9b96789ca15d69be0f3022f473fa25d1e4262`
- wheel `yee-4.3.0-py3-none-any.whl`：1,085,091 字节，sha256 `2d0895855e7258de81727f4849f464dbe4376c431ee7dd10fd1bcee337a5c9d0`

## 0. 4.3.0 更新了什么

### 0.1 `yee doctor` 部署诊断

- 识别 15 个引擎：SGLang（含 sglang-router）、vLLM、llama.cpp、TensorRT-LLM、LMDeploy、mlx-lm、mlx-serve、TensorFold、strata、Project Maya、cuteafd、TGI、Ollama、ktransformers，以及认不出的（unknown）。
- 发现方式：本机进程（/proc，跳过自身进程）、docker 容器、ssh 远端（`--hosts`，远端装了 Yee 时直接问远端的 `yee doctor --json --local`）、K8s（`--k8s`）、服务 URL（`--url`）、Maya 的 JSON 启动配置（`--config`）、直接给启动命令（`--cmd`）。
- 每个实例一个结论：**可加速 / 要改参数才能加速 / 要打补丁 / 不支持 / 无需处理**，并列出召回、KAD、专家预算、树形草稿、记录共享的状态。
- 每条改动都对应一条规则编号，可直接复制：H0–H9（投机头）、P1–P4（补丁引擎）、C1–C11（多流与高并发）、K1–K8（集群）、E1–E3（运行中效果）。
- 集群视角：同一模型多副本在同机时共享记录，跨机时提示按会话亲和路由，多机部署检查每台是否装了同一版本。
- 退出码：`0` 全部可加速或无需处理；`1` 有要改的参数、要打的补丁或集群必改项；`3` 没找到实例或全部不支持。`--json` 输出机器可读的 plan。
- 4.2 的符号检查保留在 `yee doctor --adapter`。

### 0.2 运行中提示

- 服务启动时打印 `[yee] 4.3.0 plan: <引擎> head=<加速头> -> <结论>`，有需要改的地方再打印 `[yee] 要改: <可复制命令>`。
- 运行中每 `YEE_EFFECT_S` 秒（默认 600）打印一行 `[yee] effect:`：`E1` 表示装上了但投机路径没跑到；`E2` 表示召回贡献占比很小（业务以新写内容为主），并列出投机头和并发这一侧可以改的参数；`E3` 表示多路并发时的召回统计。
- 设 `YEE_NOTICE=0` 关掉这些提示。

### 0.3 安装脚本 generic 模式

- 找不到 Mia 方案目录时，`install.sh` 自动走 generic 模式：按诊断结果装到本机进程、docker 容器或 `--hosts` 列出的每台机器。
- 装进服务**实际在用的解释器**（优先取启动命令里 venv 的 python），装完打印诊断并写 `yee-plan.json`。
- `--restart` 会 `docker restart` 容器目标；其它服务按原方式自己重启，再跑一次诊断确认。
- 新参数：`--cmd`、`--url`、`--hosts`、`--config`、`--no-docker`、`--k8s`、`--restart-cmd`、`--plan-out`、`--pip`。Mia 模式用法不变（安装目录改为 `~/.cache/yee43/`）。

### 0.4 新补丁 `maya-recall`

- 适用 Project Maya v1.1.66（Strata 分支），用 `patch -p1` 应用，然后 `cmake --build build --target maya`。
- 启动：`RECALL_ACCEL=1`、`RECALL_ACCEL_DIR=<目录>`、`MAYA_GLM_MULTI_MTP=3`；可选 `MAYA_YEE_RECALL_ROWS`。
- 限制：Maya 默认 `MAYA_GLM_MULTI_MTP_MAX=4`，并发超过 4 时 MTP 窗口关闭，召回只记账。8 卡 3080 实测把 8 路都放开是 65.1 tok/s，默认配置是 76.6 tok/s，所以放开前要先测。

### 0.5 新文档 `deploy.md`

一页讲清全适配与集群部署：诊断用法与退出码、15 个引擎的适配矩阵、全部规则、多卡 TP/PP/EP、多副本共享记录、PD 分离、K8s、怎么看是否生效、收益预期。装好后用 `yee doc deploy` 查看，源文件 [yee/docs/deploy.md](yee/docs/deploy.md)。

## 1. 支持的基座和加速头

状态说明：

| 状态 | 含义 |
|---|---|
| verified | 真机实测过，输出逐字一致，并且有提速 |
| container | 在真实镜像里检查过符号、签名和源码标记，钩子能装到真实类上，CPU 模拟测试通过，还没有起 GPU 服务实测 |
| source-checked / built | 对照引擎源码检查过（或补丁已编译），还没有上线实测 |
| native | 引擎自带这项能力，Yee 不挂钩，服务原样运行 |
| unsupported / no-op | 能识别，但没有可挂钩的点或组合不安全，Yee 不介入，服务原样运行 |

用 `python -m yee bases` 可以在本机看同一张表（`--json` 为机器可读格式）。

### 1.1 SGLang（`--speculative-algorithm`）

| 加速头 | 召回 | KAD | 说明 |
|---|---|---|---|
| `DSPARK` | verified | native | 4 台 DGX Spark（Mia 镜像，DeepSeek-V4.1-Flash）实测；块长 7 + `DSV41_VERIFY_CAP=conf:0.1` 最快 |
| `DFLASH`（含 `DFLASH2`） | container | native | 改写 `DFlashWorkerV2` 的块草稿 |
| `EAGLE` / `EAGLE3` / `NEXTN` | container | source-checked | 链式草稿（`--speculative-eagle-topk 1`） |
| `MTP` 及各类 `*_MTP`（`DEEPSEEK_MTP`、`QWEN3_NEXT_MTP`、`MIMO_MTP`、`ERNIE_MTP`、`GLM4_MOE_MTP`、`LONGCAT_FLASH_MTP`） | container | source-checked | 走 EAGLE worker |
| `STANDALONE` | container | source-checked | `StandaloneWorkerV2` |
| `UNO` | container | native | 经由 `build_eagle_verify_input` |
| `NGRAM`（含 `NGRAM_GPU`） | container | native | 不需要草稿模型；没有加速头的服务加 `--speculative-algorithm NGRAM` 就能用召回 |
| `EAGLE*` 且 topk > 1（树形草稿） | no-op | native | 不介入 |
| `FROZEN_KV_MTP` | unsupported | — | 没有可挂钩的点，doctor 会建议改成 EAGLE |

已验证版本：SGLang `0.0.0.dev0`（镜像 `dsv41-4x-spark`）。SGLang 上游正式版 0.5.10.post1 没有逐请求变草稿数的机制，KAD 不装，召回照常。

### 1.2 vLLM（V2 runner，`--speculative-config` 的 `method`）

| 加速头 | 召回 | KAD |
|---|---|---|
| `dspark` | container | verified（生产实测） |
| `dflash` / `dflash2` | container | 已适配 |
| `eagle` / `eagle3` | container | 已适配 |
| `mtp` / Gemma4 MTP / 多模块 MTP | container | 已适配 |
| `draft_model` | — | 已适配 |
| `ngram` / `suffix` | unsupported | V2 runner 下不支持 |

已验证版本：vLLM `0.28.1rc1.dev388+g8a728663c`。**必须设 `VLLM_USE_V2_MODEL_RUNNER=1`**，钩子装在 V2 runner 上。KAD 以 `num_speculative_tokens = 12` 启动，每个请求在 5～12 之间选草稿数。

### 1.3 全部 15 个引擎的接入方式（4.3.0 适配矩阵）

| 引擎 | 接入方式 | 支持的加速头 | 需要改的（可复制） |
|---|---|---|---|
| SGLang | 自动挂钩（装钩子即生效） | DSPARK / DFLASH / EAGLE / EAGLE3 / NEXTN / MTP / STANDALONE / UNO / NGRAM | EAGLE 家族要 `--speculative-eagle-topk 1`；FROZEN_KV_MTP 改 EAGLE |
| vLLM | 改参数后自动挂钩 | dspark / dflash / eagle / eagle3 / mtp | `VLLM_USE_V2_MODEL_RUNNER=1`；`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` |
| llama.cpp | 打补丁重编译 | draft-simple / eagle3 / mtp | `yee patch apply llama.cpp-recall --src <源码根>`；`cmake --build build --target llama-server`；启动加 `--recall` |
| TensorRT-LLM | 自动挂钩，只做 KAD，召回不接 | MTP / EAGLE3 / draft-target / ngram | yaml 里 `speculative_config: {decoding_type: MTP, num_nextn_predict_layers: 3}`，并加 `draft_len_schedule` |
| LMDeploy | Yee 不挂钩，原生投机照常运行 | deepseek_mtp / eagle3 / dflash / hy3_mtp / qwen3_5_mtp | 要用召回请换 vLLM 或 SGLang |
| mlx-lm | 自动挂钩，只做 KAD，召回不接 | draft model | `--draft-model <小模型> --num-draft-tokens 3` |
| mlx-serve | 打补丁重编译 | DSpark / copy / recall | `yee patch apply mlx-serve --src <源码根>`；`zig build -Doptimize=ReleaseFast` |
| TensorFold | 改参数后自动挂钩（引擎自带 Recall） | lane head / priced 提议器 | `tensorfold serve MODEL --recall`；priced 兜底再打 `yee patch apply tensorfold --src <仓库根>` |
| strata | 打补丁重编译 | MTP 单流窗口 / `--batch-mtp` 批处理槽 | `yee patch apply strata-recall --src <源码根>`；`RECALL_ACCEL=1`；多副本 `--recall-dir <同一目录>` |
| Project Maya（4.3 新增） | 打补丁重编译 | MTP（GLM 多槽调度器） | `yee patch apply maya-recall --src <源码根>`；`cmake --build build --target maya`；`RECALL_ACCEL=1` |
| cuteafd | 打补丁重编译 | DSpark | `yee patch apply cuteafd --src <源码根> --variant v2`；重编；控制目录里放 `enabled` 文件 |
| TGI | 不支持 | `--speculate` 不被 Yee 挂钩 | 换 vLLM 或 SGLang |
| Ollama | 不支持 | 无 | 换 llama.cpp 的 `llama-server` |
| ktransformers | 不支持 | 无 | 换 SGLang |
| 认不出（unknown） | — | — | 用 `--cmd "<启动命令>"` 告诉 doctor 它是什么 |

各基座的补充细节：

| 基座 | 加速头 / 路径 | 接入方式 | 状态 |
|---|---|---|---|
| llama.cpp | 召回：带草稿模型（`-md`）或不带草稿模型 | 补丁 `llama.cpp-recall`（先打 `llama.cpp-kad`） | 已提供补丁 |
| llama.cpp | KAD：`draft-simple`（`-md`） | 补丁 `llama.cpp-kad` | verified |
| llama.cpp | KAD：`eagle3`、`mtp` | 补丁 `llama.cpp-kad` | built |
| llama.cpp | `ngram-simple` / `ngram-mod` / `ngram-map` / `ngram-cache` | — | native |
| llama.cpp | qwen35 / qwen35moe 的 MTP 草稿头（`--spec-type draft-mtp`）截断词表 | 补丁 `llama.cpp-mtp-vocab`，`YEE_MTP_VOCAB_N=131072` | mx4 实测 k=3 快 5.0% |
| TensorRT-LLM | 配了 `draft_len_schedule` 的投机方法、一模型 EAGLE3 / MTP | Python 钩子 `trtllm_kad` | source-checked |
| mlx-lm | `--draft-model`（`speculative_generate_step`） | Python 钩子 `mlx_lm_kad` | verified |
| transformers | 辅助生成（AssistedCandidateGenerator 及子类）、prompt lookup | Python 钩子 `hf_kad` | verified |
| transformers | MTP（`use_mtp`）、DFlash | — | native |
| TensorFold | priced 提议器、lane 引擎 head 草稿 | 补丁 `tensorfold`、`tensorfold-kad` | built |
| cuteafd | DeepSeek V4.1 native serve 的 DSpark（贪心请求） | 补丁 `cuteafd`（v1 / v2） | 已提供补丁 |
| mlx-serve（M5 xypd 版） | DeepSeek V4.1 DSpark：贪心与采样请求 | 补丁 `mlx-serve`、`mlx-serve-2.4`、`mlx-serve-kad` | source-checked |
| strata 0.1.59 | MTP 草稿单流窗口、`--batch-mtp` 批处理槽 | 补丁 `strata-recall` | 补丁检查与单测通过，未上 GPU 实测 |
| Project Maya v1.1.66 | GLM 多槽 MTP 调度器 | 补丁 `maya-recall` | 8 卡 3080 上手工移植实跑过；入包后的补丁只做了检查与单测 |

### 1.4 SGLang Mia DeepSeek-V4.1 专项

| 项 | 默认 | 说明 |
|---|---|---|
| 逐请求 conf（`mia-dsv41-vcap`） | 打开（`{"mode": "adapt"}`） | 镜像里的 `verify_cap.py` 是补丁所基于的原件时才放入 |
| 热专家验证预算（`YEE_EXPERT_BUDGET`） | 打开，`YEE_EXPERT_BUDGET_P=0.9` | 每次 MoE 调用保留覆盖本步 90% 路由权重的热专家；4 台 DGX Spark 实测合计吞吐 +17.3%（并发 8 的 new 类 +22.5%） |
| 召回感知验证预算（`YEE_RECALL_BUDGET`） | 打开 | 只在 compact 变长验证下起作用 |
| 图片 token 预算对齐（`mia-dsv41-image-budget`） | 需显式打 | vLLM 做 P、SGLang Mia 做 D 时使用 |

## 2. 部署条件

### 2.1 通用

- **Python ≥ 3.8**，唯一依赖是 `numpy`（推理镜像里一般已有）。
- 必须装在**服务实际运行的那个 Python 环境**里（通常是镜像里的 python）。generic 模式会自动找到这个解释器。
- 多卡 / 多机（TP）部署时，**每个 TP rank 所在的机器都要装同一版本的 Yee**。版本不一致或有机器没装时，召回广播和专家预算在所有 rank 上一起关闭，服务照常运行，日志打印 `rank check off`。`yee doctor --hosts` 会逐台检查版本（规则 K1、K7）。
- 召回只认 token id：不同 tokenizer 的记录不能混用，记录目录按 tokenizer digest 自动隔离。
- 召回对和历史内容重复的部分起作用（同题重问、改写、代码编辑、RAG 摘抄、长文复述），同一段内容第二次出现时才明显变快。以新写文章、新写代码为主的业务提升很小，速度主要来自加速头本身，运行日志的 `[yee] effect: E2` 会写明。grammar / 结构化输出不参与复制。
- 默认召回记录容量 0.25 GiB（`RECALL_ACCEL_GIB`），索引密度 1，写满约占 1 GiB 内存。同机多进程可以共用一个磁盘记录目录；跨机多副本记录不共享，建议按会话亲和路由（规则 K3、K4）。
- 诊断本身只读，不装 Yee 也能跑；`--hosts` 需要能 ssh 到目标机器，`--k8s` 需要本机有 `kubectl`。

### 2.2 各基座

| 基座 | 条件 |
|---|---|
| SGLang | 已启用上表中的某个加速头；已验证镜像为 Mia `dsv41-4x-spark` |
| vLLM | `VLLM_USE_V2_MODEL_RUNNER=1`，并配了 `--speculative-config` |
| 热专家验证预算 | Mia 镜像的 `moe_b12x_next` 确定性路径：`DSV41_MOE_B12X_NEXT=1`、`DSV41_MOE_B12X_NEXT_DETERMINISTIC=1`、`EP_SIZE=1`；其它 MoE 路径上自动不介入，其余加速照常 |
| llama.cpp | 基线 v0.9.0 `cd959bae`（`llama.cpp-kad`、`llama.cpp-recall`）或 `a3b9c23e`（`llama.cpp-mtp-vocab`），打补丁后重新编译 llama-server |
| TensorFold | ≥ 1.0.12 直接用自带 Recall；补丁基线 `ashhart/TensorFold 33ada77c` |
| cuteafd | 源码 revision `9f53cc6c`，`yee patch apply cuteafd --src <源码根> --variant v2` 后重新编译 |
| mlx-serve | M5 xypd 版源码，替换源文件后重新编译 |
| strata | `Niko1221/Strata 712fec6b`（引擎 0.1.59），打补丁后重新编译 CUDA 版 |
| Project Maya | v1.1.66 源码，`maya-recall` 补丁后 `cmake --build build --target maya`；配置里要开 MTP（`MAYA_GLM_MULTI_MTP=3`，并用带 NextN 层的 GGUF） |

### 2.3 一键安装（MiaAI dsv41-4x-spark，多台 DGX Spark）

- 在 **head 宿主机**上，用平时运行 `./start-tp4.sh` 的那个用户操作。
- head 能用 `.env.tp4` 里的 `SSH_IDENTITY` 免密 ssh 到每台 worker（与 `start.sh` 的要求相同）。
- 机器上有 `bash`、`python3`、`docker`。安装脚本只用 Python 3.8+ 标准库。
- 方案目录里同时有 `start.sh` 和 `.env.tp4`（或 `.env`）。脚本会在当前目录、脚本上级目录、`/var/tmp/mia/*`、`~/*`、`/opt/*` 里自动找；找到多个时用 `--recipe <方案目录>` 指定。

### 2.4 一键安装（generic 模式，其它任何部署）

- 有 `bash` 和 Python 3.8+；装 docker 容器目标需要本机能执行 `docker`，装多机目标需要能 ssh 到 `--hosts` 里的每台机器。
- 补丁类引擎（llama.cpp、strata、Maya、mlx-serve、cuteafd）不会自动重编译，诊断报告里会给出补丁与构建命令。

## 3. 安装

### 3.1 任何部署：先诊断，再安装

```bash
unzip Yee-4.3.zip && cd Yee-4.3
bash install.sh --check                      # 只看诊断，不写任何东西
bash install.sh --check --hosts <节点列表>    # 逐台 ssh 过去查
bash install.sh --check --k8s                # 查 K8s 里的实例
bash install.sh --check --url http://<地址>:8000

bash install.sh                              # 按诊断结果装到各目标（容器、多机都算）
bash install.sh --hosts <节点列表>             # 多台机器都装
bash install.sh --restart                    # 装完重启 docker 目标并做健康检查
```

诊断也可以单独跑（不需要安装包里的脚本）：

```bash
python -m yee doctor                    # 本机进程 + docker 容器
python -m yee doctor --hosts h1,h2      # 逐台 ssh 过去查
python -m yee doctor --url http://h:8000
python -m yee doctor --k8s              # 读 kubectl get pods -A
python -m yee doctor --config maya.json # Maya 的 JSON 启动配置
python -m yee doctor --cmd "python3 -m sglang.launch_server --model-path /models/X"
python -m yee doctor --json             # 输出 plan JSON（给脚本读）
```

### 3.2 Mia dsv41-4x-spark

```bash
bash install.sh --check       # 最后一行是「检查通过，可以安装」再继续
bash install.sh --restart     # 4 台都装 + 重启 + 健康检查 + 冒烟；失败自动回退
```

脚本把 Yee 装到每台机器的 `~/.cache/yee43/` 并逐台核对 sha256，备份 env 文件后在末尾追加一段只写 `PYTHONPATH` 和 `YEE=1` 的标记块。块长、并发、EP、MoE 后端等启动参数都不改；Yee 从 `--speculative-dspark-block-size N` 读出块长，只给没设的变量补上配套值（`DSPARK_BLOCK_SIZE=N`、`DSV41_VERIFY_CAP=conf:0.1`、`DSV41_VERIFY_CAP_STRIDE=N+1`、`SGLANG_DSPARK_FOLDED_SAMPLING=2`）。

退出码：0 成功；1 检查或安装出错（env 文件没动）；2 服务已起来但 Yee 自己关掉了（日志里有 `rank check off`）；3 重启失败且已回退。

完整参数和常见问题见 [install/安装说明.md](install/安装说明.md)。

### 3.3 单个 Python 推理服务（手动）

SGLang、vLLM（V2 runner）、TensorRT-LLM、mlx-lm、transformers 等，在服务所用的 Python 环境里：

```bash
python3 -m pip install --upgrade yee-4.3.0-py3-none-any.whl && python3 -m yee install-hook
```

之后从这个 Python 启动的服务都会自动加载 Yee，照常启动即可。

### 3.4 源码基座（llama.cpp、TensorFold、cuteafd、mlx-serve、strata、Maya）

```bash
yee patch list                                # 列出全部补丁和基线
yee patch apply <补丁名> --src <源码根>        # 应用前核对 sha256，不匹配就拒绝
```

打完补丁重新编译。各补丁的开关和回退方法见 `yee doc patches`。安装包里的 `mtp_bench/`（llama.cpp MTP 测速工具）和 `strata/`（strata 召回补丁部署脚本）只在对应基座上用。

## 4. 确认生效

```bash
python -m yee doctor   # 部署诊断：能不能加速、要改什么；退出码 0 表示没有要改的
yee doctor --adapter   # 各适配器需要的符号、签名、源码标记（4.2 的检查）
yee selftest           # 自带测试组，不需要 pytest
python -m yee config   # 每个开关的生效值与来源（env / default / off）
```

服务日志里应能看到：

| 日志 | 含义 |
|---|---|
| `[yee] 4.3.0 plan: ...` | 部署诊断结论：能不能加速、要改什么 |
| `[yee] 要改: ...` | 需要改的参数或环境变量（可复制） |
| `[yee] effect: ...` | 运行中的效果判定（每 `YEE_EFFECT_S` 秒一行，默认 600） |
| `[yee] 4.3.0 defaults: ...` | 默认值已补上 |
| `[recall] patched ...` | 召回钩子已装上 |
| `[recall] rank check on: N ranks` | 各 rank 版本一致，召回广播启用 |
| `[recall] steps=... landed=... errors=0` | 召回统计，随请求增长 |
| `[yee-eb] armed geometry ...` | 热专家验证预算已启用 |

## 5. 开关与关闭

| 开关 | 默认 | 关闭 |
|---|---|---|
| `YEE`（`RECALL_ACCEL`）总开关 | 1 | `YEE=0`，整个 Yee 不加载 |
| `RECALL_ACCEL_NONGREEDY` 采样请求用召回 | 1 | 设 0 |
| `RECALL_ACCEL_QHASH` 提问哈希索引 | 1 | 设 0 |
| `YEE_KAD` 逐请求草稿数 | 1 | 设 0 |
| `YEE_EXPERT_BUDGET` 热专家验证预算 | 1，`_P=0.9` | 设 0；不重启时控制文件写 `armed=0` |
| 逐请求 conf（vcap） | adapt | 控制文件写 `{"mode": "legacy"}` 或删掉 |
| `VLLM_D_RECALL` / `_NONGREEDY` / `_KAD` | 1 | 设 0 |
| `YEE_NOTICE` 启动 plan 与运行中 effect 提示（4.3 新增） | 1 | 设 0 |
| `YEE_EFFECT_S` effect 提示间隔秒数（4.3 新增） | 600 | — |

- 只停复制：`yee pause on`，立即生效。
- 卸载：Mia 方案 `bash install.sh --uninstall --restart`（加 `--purge` 连安装目录一起删）；其它部署 `bash install.sh --uninstall`，或手动 `yee uninstall-hook && pip uninstall yee`。

全部环境变量见 `yee doc env`。

## 6. 仓库结构

| 路径 | 内容 |
|---|---|
| `yee/` | 主包：召回核心、各基座适配器、部署诊断（`yee/integrations/launch.py`、`discover.py`、`plan.py`）、补丁、内置文档（`yee/docs/`） |
| `vllm_d_recall/` | vLLM DSpark 生产适配器（含 KAD） |
| `recall_accel/` | 旧包名兼容层 |
| `install/` | 一键安装脚本 `install.sh`、`yee_install.py`、安装说明和脚本测试 |
| `tests/` | 包测试（`pytest`） |
| `tools/` | 补丁清单构建工具 |
| `dist/` | 4.3.0 wheel |

## 7. 文档

装好后用 `yee doc` 列出全部主题，源文件在 [yee/docs/](yee/docs/)：

[quickstart](yee/docs/quickstart.md) · [deploy（全适配与集群，4.3 新增）](yee/docs/deploy.md) · [heads（支持矩阵）](yee/docs/heads.md) · [sglang](yee/docs/sglang.md) · [vllm](yee/docs/vllm.md) · [kad](yee/docs/kad.md) · [expert-budget](yee/docs/expert-budget.md) · [tensorfold](yee/docs/tensorfold.md) · [cuteafd](yee/docs/cuteafd.md) · [mlx-serve](yee/docs/mlx-serve.md) · [llama-mtp](yee/docs/llama-mtp.md) · [patches](yee/docs/patches.md) · [env](yee/docs/env.md) · [troubleshooting](yee/docs/troubleshooting.md) · [changelog](yee/docs/changelog.md)

## 8. 验证

- 包内 `pytest`：728 通过、19 跳过、0 失败（新增 test_plan 28 项、test_launch 38 项、test_discover 21 项、test_doctor 8 项）；干净 venv 里装 wheel 后 `python -m yee selftest` 通过，`yee doctor --cmd "vllm serve /m"` 退出码 1 并给出 H0，`--json` 可解析。
- 安装脚本测试：Windows 28 通过 8 跳过，Linux（WSL）36 通过。
- WSL 里用发布包实跑 `bash install.sh --check`：SGLang 64 并发 / graph 批上限 32、vLLM MTP + `disable_by_batch_size 8` 两条命令行分别给出 C1、H4、C5。
- 4.3 还没在真实 GPU 集群上跑过；generic 模式的安装与 `--restart` 只在模拟的 ssh / docker 测试里跑过。
- 在仓库根目录跑测试请分两次：`python -m pytest tests yee`（包测试）和 `python -m pytest install`（安装脚本测试）。两组放进同一个 pytest 进程会互相影响，出现假失败。

## 9. 历史版本

- [v4.2.0](https://github.com/Soulmate-Halo/HaloLab-Yee/releases/tag/v4.2.0)：所有加速开关默认打开、不改启动参数、Mia 一键安装。
- [v2.2.0](https://github.com/Soulmate-Halo/HaloLab-Yee/releases/tag/v2.2.0)：改名 Yee，扩展到多个推理框架。
- [v1.2.0](https://github.com/Soulmate-Halo/HaloLab-Yee/releases/tag/v1.2.0)：首发 sglang-d-recall。

变更记录见 [CHANGELOG.md](CHANGELOG.md)。

## 许可

Apache-2.0，见 [LICENSE](LICENSE)。
