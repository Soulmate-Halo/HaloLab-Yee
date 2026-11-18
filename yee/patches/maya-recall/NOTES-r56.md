# Yee 4.2 召回（strata-recall）移植到 Project Maya v1.1.66

补丁：`maya-yee-recall.patch`，在 Maya 源码根目录 `patch -p1 < maya-yee-recall.patch`。共 10 个文件：

- 修改 `CMakeLists.txt`：`strata_spec` 加 `src/spec/yee_recall.cpp`、`src/spec/yee_recall_strata.cpp`，私有 include `third_party/nlohmann/include`，链接 `Threads::Threads`。
- 修改 `src/program/generate.cpp`（GLM 多槽调度器，`--parallel N`，N>1）。
- 新增（从 Yee 4.2 `yee/patches/strata-recall/0001-yee-recall.patch` 原样取出）：`include/strata/spec/yee_recall.hpp`、`include/strata/spec/yee_recall_strata.hpp`、`src/spec/yee_recall.cpp`、`src/spec/yee_recall_strata.cpp`、`tests/yee_recall_test.cpp`、`tests/yee_recall_strata_test.cpp`、`tests/yee_recall_shared_test.cpp`、`third_party/nlohmann/include/nlohmann/json.hpp`（3.11.3，MIT）。

## 开关

| 变量 | 默认 | 作用 |
|---|---|---|
| `RECALL_ACCEL` 或 `YEE_RECALL` | 关 | `1` 打开召回；不设时所有新代码被 `if (yee_rec)` 短路，行为与原版相同 |
| `RECALL_ACCEL_DIR` | 空（内存记录） | 磁盘记录目录；多个进程可共用（`STRATA_RECALL_SHARED=1` 默认，每 2 s 读入别人写的） |
| `RECALL_ACCEL_GIB` | 0.0625 | 记录容量 |
| `RECALL_ACCEL_READ_ONLY` | 0 | 只读记录 |
| `RECALL_ACCEL_MIN_MATCH` | 16 | 最短匹配长度 |
| `YEE_RECALL_LOG_S` | 30 | 统计行间隔（秒） |
| `MAYA_YEE_RECALL_ROWS` | 不设（= 1 + `MAYA_GLM_MULTI_MTP`） | 验证窗口至少这么多行（上限 16 与组行数上限）。设 8 时召回一槽最多给 7 个草稿，MTP 草稿链仍只做 `MAYA_GLM_MULTI_MTP` 个，只有召回填的窗口才这么宽；按窗口宽度分配的缓冲每卡多几 MB |

打开后引擎日志有一行 `[yee-recall] maya: on, backend <memory|disk>, dir <dir|->, gamma <n>, read_only <0|1>`。

## 接入点（generate.cpp，补丁后行号）

| 行 | 位置 | 做什么 |
|---|---|---|
| 2609–2628 | 调度器初始化 | `YeeRecall::create`（只在多槽调度器里建），gamma = 窗口宽度 M->W 减 1 |
| 2686–2689 | `struct Gen` | 新增 `yee_on`、`yee_k` |
| 2753–2756 | `gen_token` | 每个写出的 token `extend` |
| 2902–2903 | `gen_finish` | `end(ctx, finish != cancel)` |
| 2992–2994 | `win_take` | 窗口验证后 `verified(ctx, yee_k, k-1)`（只在窗口有草稿时） |
| 3087 | `multi_fail` | `end(ctx, false)` |
| 3325–3330 | `gen_admit` | 纯文本请求 `begin(ctx, prompt, reuse)` |
| 3379 | `multi_step` | 每轮 `tick()` |
| 3521–3549 | 窗口组装 | MTP 草稿之后、`mtp_rows_count` 之前 `propose`；召回更有把握时用它覆写本槽草稿，并同步 `draft_at`/`draft_tok` |

## 限制

- 只在 MTP 验证窗口里起作用。`MAYA_GLM_MULTI_MTP_MAX`（默认 4）决定窗口只在运行中的请求不超过 4 个时启用；超过时召回只记账、不出草稿。
- 每步召回草稿不超过窗口宽度 M->W 减 1。默认 M->W = 1 + `MAYA_GLM_MULTI_MTP`（线上 3 个草稿）；`MAYA_YEE_RECALL_ROWS=N` 把窗口放宽到 N 行（`src/core/glm_mtp_rows.cu` 的 `yee_recall_rows()`），同时把 NextN 草稿链的草稿数与窗口宽度分开（`mtpw_drafts_cap()`，替换原来的 `M->W - 1`，共 5 处）。不设时 M->W - 1 与 `mtpw_drafts_cap()` 相等，行为不变。组内总行数仍受 `rows_max()`（16）限制，`mtp_rows_fit` 会裁。
- 带图请求不用召回。单进程 `--parallel 1` 不走多槽调度器，不接召回。
- 温度大于 0 的请求：草稿只按 token 验证（draft == pick），输出分布不变。

## 构建与回滚

宿主：`cd /var/lib/libvirt/images/maya/project-maya-1.1.66 && patch -p1 < maya-yee-recall.patch && .venv/bin/cmake --build build --target maya`。
回滚：`patch -p1 -R < maya-yee-recall.patch` 后重编，或直接换回备份的 `build/maya.r55`；不重编时去掉 `RECALL_ACCEL` 即可。

## 本地验证

- WSL g++ 13：`yee_recall_test` 18/18、`yee_recall_strata_test` 25/25、`yee_recall_shared_test` 2/2 通过。
- 补丁在干净基线副本上 `patch -p1 --dry-run` 与实际应用都成功，应用结果与修改后的源码逐字相同。
- generate.cpp 的完整编译只能在宿主（CUDA 13.0）上做。
