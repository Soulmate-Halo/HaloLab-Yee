# yee patch：源码补丁的用法、基线与回退

这一页说明 Yee 自带的源码补丁（TensorFold、cuteafd、mlx-serve，3.2.0 新增的 llama.cpp KAD、TensorFold KAD、mlx-serve KAD，以及 3.3.0 新增的 llama.cpp 召回、3.5.0 新增的两个 SGLang Mia 补丁、3.5.1 新增的 strata 召回、3.5.2 新增的 llama.cpp MTP 草稿头截断词表）怎么查看、检查、应用和回退。

## 命令

```
yee patch list                                         # 列出 base、类型、variants、基线、启动环境变量
yee patch show <base> [--variant v]                    # 打印补丁正文；source-drop 打印文件清单
yee patch path <base> [--variant v]                    # 打印补丁在已安装包内的绝对路径
yee patch apply <base> --src DIR [--variant v] [--check]
```

`apply` 的流程：

1. 先核对包内补丁文件的 sha256 与 `MANIFEST.json` 一致，不一致就拒绝（文件被改过或损坏）。
2. git-patch：先 `git apply --reverse --check` 判断是否已经打过，已打过就提示 `already applied` 并返回 0；再 `git apply --check`，通过后才 `git apply`。机器上没有 git 时改用 `patch -p1 --dry-run`，通过后 `patch -p1`。
3. source-drop：检查 `DIR/src/engines/mlx/spec` 存在；内容与补丁相同的文件跳过；其余文件先备份到 `DIR/.yee-backup-<时间戳>/`（保留原相对路径）再覆盖。
4. `--check` 只做检查，不写任何文件。

退出码：成功 `0`，检查失败 `1`，用法错误（未知 base / variant、`--src` 不是目录）`2`。

## 各补丁

| base | 类型 | 基线 | 文件 | 应用位置 |
|---|---|---|---|---|
| `tensorfold` | git-patch | ashhart/TensorFold commit `33ada77c3122fca844a6825316ce8bbabaca05be` | `tensorfold/0001-recall-after-priced.patch` | 仓库根目录 |
| `tensorfold-kad` | git-patch | 同上，**并已打好 `tensorfold/0001-recall-after-priced.patch`**（先打 `tensorfold`） | `tensorfold/0002-kad.patch` | 仓库根目录 |
| `cuteafd` | git-patch，variants `v1`、`v2`（默认 `v2`） | cuteafd revision `9f53cc6cc0f3ff56de7cb81d587b3fad77137f4c` | `cuteafd/0001-recall-v1.patch`、`cuteafd/0001-recall-v2.patch`，说明 `cuteafd/NOTES-s839y-r572.md` | 源码根目录 |
| `mlx-serve` | source-drop | M5 xypd mlx-serve 源码（2026-11-10 r593 版） | `dspark.zig`、`dspark_group.zig`、`d_recall.zig`、`d_recall_live.zig` | `src/engines/mlx/spec/` |
| `mlx-serve-2.4` | source-drop | M5 上带 Yee 2.3 的 mlx-serve 线上源码（r608 src_base） | `d_recall.zig`、`d_recall_live.zig`、`d_recall_cold.zig`（新文件）、`env.zig` | 前三个在 `src/engines/mlx/spec/`，`env.zig` 在 `src/` |
| `mlx-serve-kad` | source-drop | 已拷上 `mlx-serve-2.4` 的 M5 xypd 线上源码（r608 src_base；`dspark.zig` sha256 `30412c8bd5a1e1b2...`、`dspark_group.zig` `c37b03e31a08b678...`），**先拷 `mlx-serve-2.4`** | `d_recall.zig`、`dspark.zig`、`dspark_group.zig`、`kad.zig`、`kad_live.zig`、`kad_stdio.zig` | `src/engines/mlx/spec/` |
| `llama.cpp-kad` | git-patch | ggml-org/llama.cpp v0.9.0 commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8` | `llama.cpp-kad/0001-kad.patch` | 仓库根目录 |
| `llama.cpp-recall` | git-patch | 同上，**并已打好 `llama.cpp-kad/0001-kad.patch`**（先打 `llama.cpp-kad`） | `llama.cpp-recall/0002-recall.patch` | 仓库根目录 |
| `maya-recall` | patch（`patch -p1`） | Project Maya v1.1.66（Strata 分支） | `maya-recall/0001-maya-yee-recall.patch` | Maya 源码根目录 |

### tensorfold

```
yee patch apply tensorfold --src <TensorFold 仓库根> --check
yee patch apply tensorfold --src <TensorFold 仓库根>
# 按 TensorFold 自己的构建方式重新编译（Zig 版本见仓库 .zig-version，当前 0.17.0）
TF_RECALL_AFTER_PRICED=1 tensorfold serve MODEL --recall
```

改动 `zig/src/core/lanes/windows.zig`、`config.zig`、`recall_engine_test.zig`，共增 74 行、删 7 行。不设 `TF_RECALL_AFTER_PRICED` 时行为与上游完全一致。详见 [tensorfold.md](tensorfold.md)。

### tensorfold-kad（3.2.0 新增）

必须在 `tensorfold` 之后打（基线要求已含 `0001-recall-after-priced.patch`）。

```
yee patch apply tensorfold --src <TensorFold 仓库根> --check
yee patch apply tensorfold --src <TensorFold 仓库根>
yee patch apply tensorfold-kad --src <TensorFold 仓库根> --check
yee patch apply tensorfold-kad --src <TensorFold 仓库根>
# 按仓库 .zig-version（当前 0.17.0）重新编译
# DeepSeek V4.1 的启动宽度 12：模型 config.json 里写 "dspark_block_size": 12
TF_KAD=1 tensorfold serve MODEL --recall
```

- 补丁内容：新增 `zig/src/core/lanes/kad.zig`（KAD 核心）、`kad_stdio.zig`、`kad_host.zig`（TensorFold 宿主），另有 10 处锚点编辑（6 个已有文件）。
- 开关：`TF_KAD=1` 或 `YEE_KAD=1`；策略文件用 `TF_KAD_POLICY` 或 `YEE_KAD_POLICY`（默认 `$YEE_KAD_DIR/kad_policy.json`，目录默认 `~/.cache/yee/kad`）；统计用 `TF_KAD_STATS_FILE` 或 `YEE_KAD_STATS_FILE`（默认 `kad_stats_tensorfold.json`）。
- 日志：`[yee-kad] tensorfold enabled: launch K=12, policy <路径>, stats <路径>`；启动宽度不是 12 时多一行 `[yee-kad] tensorfold WARNING launch K=<K>, KAD needs 12 (per-request width is capped at min(12, K))`。
- 启动宽度 = `cfg.most_drafts` = min(`Model.drafts`, 窗口宽度 - 1, max_draft)；DeepSeek V4.1 的 `Model.drafts` 来自模型 `config.json` 的 `"dspark_block_size"`（默认 5），上限 16。

### cuteafd

```
yee patch apply cuteafd --src <cuteafd 源码根> --check
yee patch apply cuteafd --src <cuteafd 源码根> --variant v2
```

推荐 v2。仅贪心请求生效，记录只在进程内存里，重启清空；开关是控制目录里的 `enabled` 文件。详见 [cuteafd.md](cuteafd.md) 和 `yee patch show cuteafd` 旁的 NOTES。

### mlx-serve

```
yee patch apply mlx-serve --src <mlx-serve 源码根> --check
yee patch apply mlx-serve --src <mlx-serve 源码根>
# 重新编译 mlx-serve 后重启
```

详见 [mlx-serve.md](mlx-serve.md)。

### mlx-serve-2.4（Yee 2.4 冷层）

```
yee patch apply mlx-serve-2.4 --src <mlx-serve 源码根> --check
yee patch apply mlx-serve-2.4 --src <mlx-serve 源码根>
# 重新编译 mlx-serve；设 MLX_SERVE_D_RECALL_COLD_DIR 后重启才开冷层
```

基线是已经带 Yee 2.3 的源码，不是 r593 原版。详见 [mlx-serve.md](mlx-serve.md) 第 8 节。

### mlx-serve-kad（3.2.0 新增）

必须先拷 `mlx-serve-2.4`，再拷本补丁（基线要求已含 2.4）。

```
yee patch apply mlx-serve-2.4 --src <mlx-serve 源码根> --check
yee patch apply mlx-serve-2.4 --src <mlx-serve 源码根>
yee patch apply mlx-serve-kad --src <mlx-serve 源码根> --check
yee patch apply mlx-serve-kad --src <mlx-serve 源码根>
# 重新编译并重启
```

- 补丁内容：替换 `src/engines/mlx/spec/` 下的 `d_recall.zig`、`dspark.zig`、`dspark_group.zig`，新增 `kad.zig`、`kad_live.zig`、`kad_stdio.zig`（与 TensorFold 共用的 Zig 核心）。
- 重编命令：`zig build check`；`zig build test -Dtest-filter=d_recall`、`-Dtest-filter=dspark`、`-Dtest-filter=kad`；`zig build -Doptimize=ReleaseFast`。
- 开关：`MLX_SERVE_KAD=1` 或 `YEE_KAD=1`；策略文件用 `MLX_SERVE_KAD_POLICY` 或 `YEE_KAD_POLICY`（默认 `$YEE_KAD_DIR/kad_policy.json`）；统计用 `MLX_SERVE_KAD_STATS_FILE` 或 `YEE_KAD_STATS_FILE`（默认 `kad_stats_mlx-serve.json`）。
- mlx-serve 没有草稿数的命令行参数：DSpark 块长来自模型 `config.json` 的 `"dspark_block_size"`，写成 12；核对上限 15，12 合法。
- 日志：`[yee-kad] mlx-serve enabled: launch K=12, policy <路径>, stats <路径>`；启动宽度不是 12 时多一行 `[yee-kad] mlx-serve WARNING launch K=<K>, KAD needs 12 (per-request width is capped at min(12, K))`。

### llama.cpp-kad（3.2.0 新增）

```
yee patch apply llama.cpp-kad --src <llama.cpp 仓库根> --check
yee patch apply llama.cpp-kad --src <llama.cpp 仓库根>
# 重编 llama-server
cmake --build build --target llama-server
```

- 补丁内容：新增 `common/kad.h`、`common/kad.cpp`，另有 13 处锚点编辑，共 7 个文件。
- 开关：`--kad`、`LLAMA_ARG_KAD=1` 或 `YEE_KAD=1`；策略文件用 `--kad-policy FNAME` / `LLAMA_ARG_KAD_POLICY`，否则 `YEE_KAD_POLICY`，否则 `$YEE_KAD_DIR/kad_policy.json`；统计用 `YEE_KAD_STATS_FILE`（默认 `kad_stats_llama.cpp.json`）。
- 启动宽度 12 的参数：`llama-server -m <target.gguf> -md <draft.gguf> --spec-type draft-simple --spec-draft-n-max 12 --spec-draft-n-min 0 --kad`（这个版本必须显式写 `--spec-type draft-simple`，否则投机和 KAD 都不会启动）。
- 日志：`[yee-kad] llama.cpp enabled: launch K=12, policy <路径>, stats <路径>`；启动宽度不是 12 时多一行 `[yee-kad] llama.cpp WARNING launch K=<K>, KAD needs 12 (per-request width is capped at min(12, K))`。

### llama.cpp-recall（3.3.0 新增）

必须在 `llama.cpp-kad` 之后打（基线要求已含 `0001-kad.patch`）。

```
yee patch apply llama.cpp-kad --src <llama.cpp 仓库根> --check
yee patch apply llama.cpp-kad --src <llama.cpp 仓库根>
yee patch apply llama.cpp-recall --src <llama.cpp 仓库根> --check
yee patch apply llama.cpp-recall --src <llama.cpp 仓库根>
# 重编 llama-server
cmake --build build --target llama-server
```

- 补丁内容：新增 `common/recall.h`、`common/recall.cpp`（`yee.core` 的 C++ 移植：跨请求复制草稿，记录可以放内存也可以落盘），另有 `common/CMakeLists.txt`、`common/common.h`、`common/arg.cpp`、`common/kad.h`、`common/kad.cpp`、`tools/server/server-context.cpp` 的锚点编辑。
- 作用：把服务读过、写过的 token id 记下来，新请求与历史内容重复时把后续内容复制成草稿，交给 llama-server 正在用的加速头核对；带草稿模型时召回的提议替换草稿头的草稿，草稿头状态机照常走；不带草稿模型时召回有提议就投机核对，没有就普通解码。
- 开关：`--recall`、`LLAMA_ARG_RECALL=1`、`RECALL_ACCEL=1`、`YEE=1` 或 `YEE_RECALL=1`；参数 `--recall-dir DIR`（记录落盘目录，重启后还在，不给就是内存记录）、`--recall-read-only`（只读用已有记录）、`--recall-gib F`（容量，默认 0.0625）、`--recall-n N`（每步最多召回草稿数，默认 12）。
- 环境变量：`LLAMA_ARG_RECALL_DIR`、`LLAMA_ARG_RECALL_READ_ONLY`、`LLAMA_ARG_RECALL_GIB`、`LLAMA_ARG_RECALL_N`；统计文件 `YEE_RECALL_STATS_FILE`，周期日志秒数 `YEE_RECALL_LOG_S`（默认 30）。
- 启动参数（三种用法）：

```
# 只开召回（没有草稿模型）
llama-server -m <target.gguf> --recall
# 召回 + draft-simple + KAD
llama-server -m <target.gguf> -md <draft.gguf> --spec-type draft-simple --spec-draft-n-max 12 --spec-draft-n-min 0 --kad --recall
# 记录落盘，重启后继续用
llama-server -m <target.gguf> --recall --recall-dir ~/.cache/yee/recall/llama
```

- 生效日志（启动时两行）：

```
[yee-recall] llama.cpp record: backend=memory dir=- writer=1 ids=0 cap_ids=16777216 index_slots=4194304 short=2 tree=1 k=4 stats=<路径>
[yee-recall] llama.cpp enabled: backend=memory dir=- read_only=0 gamma=12 head=none
```

运行中每 `YEE_RECALL_LOG_S` 秒一行 `[yee-recall] llama.cpp ticks= streams= proposals= used= landed= src=own:,record: tier=16-31:,32-63:,64+:,<16: store_ids= records= evicted= errors=`。
- 已知限制：只认纯文本 slot，多模态（带 mmproj）、TTS、realtime 的 slot 不用召回；`--spec-synth-*` 合成接受率下召回整体关闭；召回记录目录不与 Python 基座共用（两边身份摘要不同会被拒绝）；没有在 GPU 上实测过。

### mia-dsv41-vcap（3.5.0 新增）

SGLang Mia 镜像（DeepSeek-V4.1，`dsv41-4x-spark`）的逐请求 conf。`--src` 是容器 PYTHONPATH 最前面的目录（生产是宿主机的 `/root/.cache/yee42`，挂进容器后排在 `/opt/dsv41/adapter` 前面）。

```
yee patch apply mia-dsv41-vcap --src /root/.cache/yee42 --check
yee patch apply mia-dsv41-vcap --src /root/.cache/yee42
# TP 的每台宿主机都要放，然后重启 D
```

- 补丁内容：`verify_cap.py`（在镜像内置版本 sha256 `949c3fb72d72…` 上加运行时控制文件）、`vcap_policy.py`（rank0 CPU 上的门槛策略，只依赖 numpy）。
- 控制文件 `/state/vcap_ctl.json`（`DSV41_VCAP_CTL` 可改路径；head 宿主机在 Mia 目录的 `state-tp4/` 下），rank0 最多每 0.5 秒看一次：不存在或 `{"mode": "legacy"}` 时与 `DSV41_VERIFY_CAP=conf:0.1` 逐位一致；`{"mode": "adapt"}` 时用 CPU 算的每请求门槛。生效后写 `/state/vcap_ack.json`，出错时写 `mode=failed` 并退回 `conf:T`；统计写 `/state/vcap_stats.jsonl`、`/state/vcap_req.jsonl`。
- 其它字段（`thr`、`thr_min` / `thr_max`、`kmin` / `kmax`、`explore`、`logics` 等）见 `vcap_policy.py` 文件头。
- 3.8.0 选宽度器影子模式（只改了 `vcap_policy.py`，`verify_cap.py` 与 3.5.0～3.7.0 相同）：控制文件写 `{"mode": "width", "tag": "<标签>"}` 开启，verify_cap 当 legacy 走（每步 k 与 `conf:0.1` 相同，ack 里的 mode 是 `legacy`）。rank0 每步给候选验证宽度（默认 4 / 8；可以在 `width.widths` 里加 13，但 r703 对照实测它的估算偏高 5～30 个百分点，只作参考）估期望出 token 数和步时，记下会选哪个宽度，并按本步实际接受数结算；`vcap_stats.jsonl` 每个窗口多一个 `width` 字段（按 bs 档：`act`、`sel`、`sel_lo`、`le0`、`oracle`、`static`、`pick`、`pred`）。汇总用 `python -m yee.tools.vcap_width_report /state/vcap_stats.jsonl`。可调字段写在控制文件的 `width` 子配置里（`widths`、`max_bs`、`switch`、`hist`、`cost`、`hi_scale`、`base` 等，见 `vcap_policy.py` 文件头）。比现网宽度更宽的那部分收益是估算值，`sel_lo` 是不计这部分的下限。

### mia-dsv41-image-budget（3.5.0 新增）

vLLM 做 P、SGLang Mia 镜像做 D 的 PD 部署：让 D 的图片 token 预算与 P 一致。`--src` 是 SGLang 的 Python 包根。

```
yee patch apply mia-dsv41-image-budget --src /sgl-workspace/sglang/python/sglang --check
yee patch apply mia-dsv41-image-budget --src /sgl-workspace/sglang/python/sglang
# 容器环境里加 XY_DSV41_IMAGE_BUDGET_RESERVE=1，重启 D
```

- 补丁内容：替换 `srt/multimodal/deepseek_v41_image_processing.py`。目标必须是镜像原件（sha256 `dcd06c489aa1…`），不是就拒绝；拷完删掉 `srt/multimodal/__pycache__/deepseek_v41_image_processing.*.pyc`。
- 作用：`XY_DSV41_IMAGE_BUDGET_RESERVE=1` 时 `safe_resize` 的 token 预算减 1，与 vLLM 的预处理一致，3024x588 的图两端都是 886 个 token。缺省 0 与原版一致。
- 镜像只读时，在原镜像上加一层做这一步：Dockerfile 里装上本 wheel 后运行上面两条命令，或者直接 `COPY` 补丁目录里的这个文件。

### strata-recall（3.5.1 新增）

给 C++ MoE 引擎 strata（Niko1221/Strata engine 0.1.59，提交 `712fec6b`）加跨请求召回。也能打在 rtx3080n1 的专家服务器版源码上。

```
yee patch apply strata-recall --src <strata 仓库根> --check
yee patch apply strata-recall --src <strata 仓库根>
# 重编 CUDA 版 strata；专家服务器版先备份原来的 engine/strata，再把 build-*/strata 拷过去
cmake --build <构建目录> --target strata
# 启动引擎时加 RECALL_ACCEL=1（serve/server.py 起的引擎子进程继承环境变量），或引擎参数 --recall
```

- 补丁内容：新增召回库 `include/strata/spec/yee_recall.hpp`、`src/spec/yee_recall.cpp`（与 llama.cpp-recall 的 `common/recall.cpp` 同一份移植，命名空间 `yee`），引擎侧 `include/strata/spec/yee_recall_strata.hpp`、`src/spec/yee_recall_strata.cpp`，`third_party/nlohmann/include/nlohmann/json.hpp`（3.11.3，MIT），单测 `tests/yee_recall_test.cpp`、`tests/yee_recall_strata_test.cpp`；锚点编辑 `cmake/backends/cpu.cmake`、`src/program/generate.cpp`。
- 作用：每个引擎进程一个记录，每个请求一个流（批处理槽 0～batch-1，单流用槽 batch）。单流窗口里召回提议像 prompt-lookup 草稿一样进窗口；`--batch-mtp` 的批处理窗口里替换该槽这一窗口的 MTP 草稿，最多 `--spec` 减 1 个，且不挤掉后面的槽。验证窗口核对每个草稿，输出不变；温度大于 0 时按单点草稿接受，分布不变。
- 开关与参数：`--recall` 或 `RECALL_ACCEL=1` / `YEE_RECALL=1` 开；`--no-recall` 强制关；`--recall-dir DIR`（记录目录，多个进程可共用；默认 `RECALL_ACCEL_DIR`，都没有时是内存记录，进程退出即清空）、`--recall-gib G`（容量，默认 0.0625）、`--recall-read-only`（只用旧记录）。`RECALL_ACCEL_MIN_MATCH`、`YEE_RECALL_STATS_FILE`（默认 `~/.cache/yee/recall/recall_stats_strata.json`）、`YEE_RECALL_LOG_S`（默认 30）照召回库。
- 生效日志（stderr，启动时两行）：

```
[yee-recall] strata record: backend=memory dir=- writer=1 ids=0 cap_ids=16777216 index_slots=4194304 short=2 tree=1 k=4 stats=<路径>
[yee-recall] strata: on, backend memory, dir -, gamma 7, read_only 0
```

- 已知限制：只认纯文本请求（带图的请求不用召回）；`STRATA_BATCH_DF2`、`STRATA_PIPED_MTP`、不带 `--batch-mtp` 的批处理窗口不接召回（启动时打印一行说明）；gated 窗口与合并首窗口不接；记录身份按模型名加词表大小，不与 Python 基座、llama.cpp 的记录共用；记录写入进行中的那一窗口不问召回（召回库的设计：解码线程不等锁）。没有在 GPU 上编译和实测过。

### strata-recall（3.6.0 更新）

3.6.0 只换了补丁里新增的 `src/spec/yee_recall.cpp`（多副本共享磁盘记录，`STRATA_RECALL_SHARED`，缺省 1），并新增双进程单测 `tests/yee_recall_shared_test.cpp`；其它部分与 3.5.1 相同，基线不变。已经打过 3.5.x `strata-recall` 的树，先 `git apply -R` 旧补丁（`yee patch path strata-recall` 在 3.5.x 环境里给出路径），再打 3.6.0 的。

### llama.cpp-mtp-vocab（3.5.2 新增）

llama.cpp 的 qwen35 / qwen35moe MTP 草稿头（`--spec-type draft-mtp`）只算词表前 N 个 token。基线 ggml-org/llama.cpp 提交 `a3b9c23`（2026-08-21）。

```
yee patch apply llama.cpp-mtp-vocab --src <llama.cpp 仓库根> --check
yee patch apply llama.cpp-mtp-vocab --src <llama.cpp 仓库根>
cmake --build build --target llama-server
YEE_MTP_VOCAB_N=131072 llama-server -m <带 MTP 的 gguf> -ngl 99 --spec-type draft-mtp --spec-draft-n-max 3
```

- 补丁内容：`src/models/qwen35.cpp` 新增 `yee_mtp_vocab_n()`（只读一次 `YEE_MTP_VOCAB_N`）；MTP 图的输出头先用 `ggml_view_2d` 只留前 N 行，矩阵乘后 `ggml_scale_bias(+1000)`、`ggml_pad` 补到词表宽度、`ggml_scale_bias(-1000)`。
- 生效日志：`[yee-mtp-vocab] MTP draft head limited to first 131072 of 248320 tokens`。
- 已知限制：只对 qwen35 / qwen35moe 的 MTP 图生效；N 要按模型量（`yee/tools/mtp_bench/vocab_coverage.py`），Qwen3.6 用 131072；输出头带逐行缩放时不截断。

### maya-recall（4.3.0 新增）

给 Project Maya v1.1.66（Strata 分支）加跨请求召回。Maya 的 token 循环在 C++ 进程里，Python 钩子碰不到，所以以源码补丁接入：召回库是 Yee 的 C++ 移植（与 strata、llama.cpp 的补丁同一份），召回提议替换 MTP 验证窗口里的草稿，窗口逐个核对草稿。

```
yee patch apply maya-recall --src <Maya 源码根> --check
yee patch apply maya-recall --src <Maya 源码根>
# 重编并重启
cmake --build build --target maya
```

- 补丁内容：改 `CMakeLists.txt`（`strata_spec` 加两个源文件、链接 `Threads::Threads`），改 `src/program/generate.cpp`（GLM 多槽调度器，`--parallel N`），新增召回库与引擎适配层、三个单测文件，以及 `third_party/nlohmann/include/nlohmann/json.hpp`（3.11.3，MIT）。
- 开关：

| 变量 | 默认 | 作用 |
|---|---|---|
| `RECALL_ACCEL` 或 `YEE_RECALL` | 关 | `1` 打开召回；不设时新代码被短路，行为与原版一致 |
| `RECALL_ACCEL_DIR` | 空（内存记录） | 磁盘记录目录；多个进程可共用（`STRATA_RECALL_SHARED=1` 默认，每 2 秒读入别人写的） |
| `RECALL_ACCEL_GIB` | 0.0625 | 记录容量 |
| `RECALL_ACCEL_READ_ONLY` | 0 | 只读记录 |
| `RECALL_ACCEL_MIN_MATCH` | 16 | 最短匹配长度（token） |
| `YEE_RECALL_LOG_S` | 30 | 统计行间隔（秒） |
| `MAYA_YEE_RECALL_ROWS` | 不设（= 1 + `MAYA_GLM_MULTI_MTP`） | 验证窗口至少这么多行（上限 16）；设 8 时召回一槽最多给 7 个草稿，MTP 草稿链仍只做 `MAYA_GLM_MULTI_MTP` 个 |
| `MAYA_GLM_MULTI_MTP` | 不设 | MTP 草稿数；H7 要求它是 3 |

- 生效日志：`[yee-recall] maya: on, backend <memory|disk>, dir <dir|->, gamma <n>, read_only <0|1>`。
- 已知限制：只在 MTP 验证窗口里起作用；`MAYA_GLM_MULTI_MTP_MAX`（默认 4）决定窗口只在运行中的请求不超过 4 个时启用，超过时**召回只记账、不出草稿**；带图请求不用召回；单进程 `--parallel 1` 不走多槽调度器，不接召回；温度大于 0 的请求草稿只按 token 验证，输出分布不变。
- 8×3080 实测：8 路放开（`MAYA_GLM_MULTI_MTP_MAX=0`）后 65.1 tok/s，默认配置是 76.6 tok/s，放开反而更慢，只在 GPU 不是瓶颈时考虑。

## 回退

| base | 回退方法 |
|---|---|
| `tensorfold` | 仓库根目录执行 `git apply -R "$(yee patch path tensorfold)"`，重新编译；或者不重编、只去掉 `TF_RECALL_AFTER_PRICED`，行为即与上游一致 |
| `tensorfold-kad` | 仓库根目录执行 `git apply -R "$(yee patch path tensorfold-kad)"`，重新编译；不重编时去掉 `TF_KAD`（或 `YEE_KAD`）即回到 0001 的行为 |
| `cuteafd` | 源码根目录执行 `git apply -R "$(yee patch path cuteafd --variant v2)"`（没有 git 用 `patch -p1 -R -i <同一路径>`），重新编译；不重编时删掉控制目录里的 `enabled` 文件即关闭召回 |
| `mlx-serve` | 把 `DIR/.yee-backup-<时间戳>/src/engines/mlx/spec/` 下的文件拷回 `DIR/src/engines/mlx/spec/`，重新编译 |
| `mlx-serve-2.4` | 不重编时去掉 `MLX_SERVE_D_RECALL_COLD_DIR` 即回到 2.3 行为；要还原源码，把 `DIR/.yee-backup-<时间戳>/` 下的文件按原路径拷回，删掉新增的 `src/engines/mlx/spec/d_recall_cold.zig`，重新编译 |
| `mlx-serve-kad` | 把 `DIR/.yee-backup-<时间戳>/src/engines/mlx/spec/` 下的文件拷回 `DIR/src/engines/mlx/spec/`，删掉新增的 `kad.zig`、`kad_live.zig`、`kad_stdio.zig`，重新编译；不重编时去掉 `MLX_SERVE_KAD`（或 `YEE_KAD`）即回到 2.4 的行为 |
| `llama.cpp-kad` | 仓库根目录执行 `git apply -R "$(yee patch path llama.cpp-kad)"`（没有 git 用 `patch -p1 -R -i <同一路径>`），重新编译；不重编时去掉 `--kad`（或 `LLAMA_ARG_KAD`、`YEE_KAD`）即回到上游行为 |
| `llama.cpp-recall` | 先回退本补丁：仓库根目录执行 `git apply -R "$(yee patch path llama.cpp-recall)"`（没有 git 用 `patch -p1 -R -i <同一路径>`），再回退 `llama.cpp-kad`，重新编译；不重编时去掉 `--recall`（或 `LLAMA_ARG_RECALL`、`RECALL_ACCEL`、`YEE_RECALL`）即回到只打 KAD 的行为 |
| `mia-dsv41-vcap` | 删掉 `DIR` 里的 `verify_cap.py`、`vcap_policy.py`（`DIR` 原来就有同名文件时，从 `DIR/.yee-backup-<时间戳>/` 拷回），重启 D；不重启时把控制文件改成 `{"mode": "legacy"}` 或删掉，即与 `conf:0.1` 逐位一致 |
| `mia-dsv41-image-budget` | 把 `DIR/.yee-backup-<时间戳>/srt/multimodal/deepseek_v41_image_processing.py` 拷回原处，重启 D；不还原文件时去掉 `XY_DSV41_IMAGE_BUDGET_RESERVE`（缺省 0）即与原版一致 |
| `strata-recall` | 仓库根目录执行 `git apply -R "$(yee patch path strata-recall)"`（没有 git 用 `patch -p1 -R -i <同一路径>`），重新编译；不重编时去掉 `RECALL_ACCEL`（或 `--recall`、`YEE_RECALL`）即与上游逐位一致；rtx3080n1 也可直接换回备份的 `engine/strata` |
| `llama.cpp-mtp-vocab` | 仓库根目录执行 `git apply -R "$(yee patch path llama.cpp-mtp-vocab)"`，重新编译；不重编时去掉 `YEE_MTP_VOCAB_N` 即与上游逐位一致 |
| `maya-recall` | Maya 源码根目录执行 `patch -p1 -R < maya-yee-recall.patch` 后重编；或换回备份的 `build/maya`；不重编时去掉 `RECALL_ACCEL` 即与原版逐位一致 |

回退前先确认补丁版本：`yee patch list` 显示的 variant 要和当初应用的一致。

## 补丁清单的完整性

`yee/patches/MANIFEST.json` 里每个文件的 sha256 是构建时由 `tools/build_manifest.py` 算出写死的；`yee selftest` 里的 `test_yee_pkg.py` 会逐个核对，并检查包内没有清单之外的文件。
