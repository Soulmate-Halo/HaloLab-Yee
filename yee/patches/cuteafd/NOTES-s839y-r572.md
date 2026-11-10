# cuteafd 跨请求历史复制草稿（recall）补丁说明（s839y-r572）

源码基线：ysy101:/home/ysy101/cuteafd（10-18 发布包，镜像 revision 9f53cc6cc0f3ff56de7cb81d587b3fad77137f4c）。
`orig/` 是原文件，`new/` 是当前（v2）改后文件，路径从 `rust/` 开始。

| 版本 | 补丁 | 二进制 sha256 | 镜像 |
|---|---|---|---|
| v1 | `recall.patch`（md5 bed0372f9a4876ca0acfbe4032f3904e，556 行） | 5ba977861b9cbbf3dac55673275bdb26b4bd0a49fe25a289d63ce332bcbda4c2 | `cuteafd-coordinator-recall:v3.0.0-r572`（be48be7f7657） |
| v2 | `recall_v2.patch`（md5 802a0a66470ef9cf6d15fa044ae3d98e，657 行） | 15dedc5e076084b2affe173285f0a9c0819f52576dbfbda796e9a500b07cc0dc | `cuteafd-coordinator-recall:v3.0.0-r572b`（02f30431a01a） |

两个补丁都在源码根目录 `patch -p1` 应用，都已在 ysy101 上对原样副本复验（PATCH-VERIFY PASS / PATCH2-VERIFY PASS）。

## 改动（v2）

| 文件 | 改动 |
|---|---|
| `rust/crates/cuteafd-daemon/src/shared/draft/recall.rs` | 新文件：环形 token 存储 `Store`、直接映射哈希索引、后台控制线程、`init/enabled/prior/lookup/record/note_proposal`，8 个单元测试 |
| `rust/crates/cuteafd-daemon/src/shared/draft.rs:23` | `pub(crate) mod recall;` |
| `rust/crates/cuteafd-daemon/src/shared/draft/copy.rs` | 新增 `sourced`（返回续写和"是否来自 recall"）；`continuation` 改为调用它；本地复制不合格时走私有函数 `recalled` |
| `.../deepseek_v41/v41_native_serve/scheduler.rs:138` | `retire_request` 里 `if request.cacheable && !request.failed { recall::record(&request.tokens) }` |
| `.../deepseek_v41/v41_native_serve/scheduler.rs:242` | `serve` 开头 `recall::init()` |
| `.../deepseek_v41/v41_native_serve/scheduler.rs` `extend_inputs` | 改调 `copy::sourced`；recall 续写调用 `extend_recalled_confidence(..., recall::prior())`，本地复制仍调用原来的 `extend_confidence` |
| `.../deepseek_v41/v41_native_serve/speculative.rs` | 新增 `DraftRuntime::extend_recalled_confidence`：recall 续写的每个位置（神经前缀与复制部分）置信度至少为 prior |
| `.../deepseek_v41/v41_native_serve/scheduler/independent.rs:324` | 独立双 lane 的 `retire` 里同样的 `record` 调用 |

Cargo.toml 未改，没有新依赖。其它 family 的代码未改。

v1 与 v2 的区别只在草稿长度策略看到的置信度：v1 的 recall 续写沿用本地复制的尾部先验（0.95、0.9、0.65、0.3，之后 0.2），simple 策略按累计存活率 > 0.5 截断，实际只多验 1–2 个复制 token；v2 让 recall 续写的每个位置至少是 prior（默认 0.98），累计存活率可以撑到二三十个位置。

## 机制

- 记录：请求正常结束（EOS 或到 max_tokens，`cacheable=true` 且 `failed=false`；冷探针请求不记）后，把 `request.tokens`（prompt + 生成内容）经 mpsc 交给后台线程 `cuteafd-recall` 写入环形缓冲，段尾写分隔符 `u32::MAX`。每写 8192 个 token 释放一次写锁。出错、取消、客户端中途断开的请求不记录。
- 索引：对每个以位置 i 结尾的 16 个 token（`RECALL_MATCH=16`，不跨分隔符）算 64 位哈希（乘法混合 + fmix64 终混），槽里存"后继 token 的全局位置 + 1"。槽数 = 2×容量向上取 2 的幂。同一序列再出现会覆盖旧槽，所以命中的是最近一次出现。
- 查询：`copy::sourced` 在本地复制不合格时调用 `recall::lookup(history, limit)`。拿不到读锁（`try_read`）直接返回 None，不阻塞 decode。取 history 尾部 16 个 token 查槽，校验位置仍在环内、逐个比对 16 个 token（防哈希碰撞和环形覆盖），再从后继位置起最多取 `limit` 个 token，遇分隔符或写位置截断。
- 合格条件与本地复制相同：神经草稿前 `neural` 个（置信度 ≥0.6 的前缀，至少 3 个）必须与 recall 续写一致，且续写长度 > neural。只对 greedy 请求生效（与本地复制同一个前置条件）。

## 开关与统计（不重启切换）

控制目录：容器内 `CUTEAFD_RECALL_DIR`，默认 `/run/cuteafd/bench/recall`，对应宿主机 `/home/ysy101/.cache/cuteafd/bench/recall`（服务启动后自动创建）。

| 操作 | 命令（宿主机） | 生效 |
|---|---|---|
| 开 | `touch /home/ysy101/.cache/cuteafd/bench/recall/enabled` | ≤1 秒，首次开启时分配内存 |
| 关 | `rm /home/ysy101/.cache/cuteafd/bench/recall/enabled` | ≤1 秒，释放内存、历史清空 |
| 清空历史 | `touch /home/ysy101/.cache/cuteafd/bench/recall/clear.request` | ≤1 秒，文件自动删除 |
| 调 prior（v2） | `echo 0.95 > /home/ysy101/.cache/cuteafd/bench/recall/prior`（0.5–0.999，非法值用默认 0.98） | ≤1 秒 |
| 容量 | 开启前写 `capacity` 文件（token 数，1048576–134217728），默认 16777216 | 下次开启时读 |
| 统计 | `cat /home/ysy101/.cache/cuteafd/bench/recall/stats.json` | 每秒刷新 |

stats.json 字段：enabled、capacity、stored_tokens、write_pos、recorded_requests、recorded_tokens、lookups、hits、busy（读锁忙跳过次数）、proposals（合格并送去长度策略的 recall 续写数）、proposed_tokens（其中复制部分的 token 数，截断前）、clears、prior（v2）、uptime_s。

内存：默认容量 16M token 时，环 64 MiB + 索引 256 MiB（calloc，按触及页计入 RSS）；关闭时不占内存。

## 关闭时与原版等价

1. `sourced` 的三个前置条件（greedy、drafted 长度、calibrated 长度）原样直接返回 None。
2. 本地复制的判断式、`window` 调用和参数（`limit` 就是原来的 `room.min(drafted.len() + COPY_EXTENSION)`）不变；合格时返回原来的续写并标记"非 recall"，`extend_inputs` 走原来的 `extend_confidence`。
3. 原来 `return None` 的分支现在走 `recalled(...)`，它第一行是 `if !recall::enabled() { return None; }`。没有 `enabled` 文件时 `ENABLED` 永远是 false，所以 `extend_recalled_confidence` 不会被调用，返回值与原版逐分支相同。
4. `record` 在关闭时第一行就返回；`init` 只启动一个每秒醒一次的后台线程，不碰 decode 路径。
5. 原有 copy 的 6 个单元测试一个未改，v1、v2 都全部通过（`cargo test ... -- shared::draft::`：v1 68 passed，v2 69 passed，0 failed）。

## 构建与部署

- 构建：容器 `r572-build`（同线上镜像），Rust 1.98.1，`RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --locked --release -p cuteafd-daemon --bin cuteafd -j 8 --target x86_64-unknown-linux-gnu --target-dir /build/target/x86-64-v4`（与 scripts/build/host-isa.sh 的 release 命令一致）。源码包缺 `third_party/transformers` 子模块内容，只补了 `include_bytes!` 用到的两个文件（pinned revision 62d7ebd7 的 raw 原文）。
- 原样构建与线上官方二进制：同一 rustc 1.98.1 / LLD 22.1.8，.text 相差 64 字节，AVX-512 指令数相同（9422），差异只在依赖源码路径字符串。增量重编约 90–100 秒。
- 镜像 FROM 线上同一镜像，只替换 `/opt/cuteafd/bin/x86-64-v4/cuteafd`，revision 标签继承，配置指纹不变。
- 上线：`./run.sh --config deploy-ours.config --set COORDINATOR_DOCKER_INFERENCE=cuteafd-coordinator-recall:v3.0.0-r572b --restart`（deploy-ours.config 未改）。
- 回退：`./run.sh --config deploy-ours.config --restart`（不带 --set 即用回官方 v3.0.0 镜像）；只想关功能不重启就删 `enabled` 文件。

## 已知限制

- 只对 greedy（temperature=0）请求生效，采样请求走不到复制草稿。
- 历史只在进程内存里，D 端重启后清空，需要重新积累。
- 神经草稿前 3 个以上 token 必须与历史一致才会用。
- prior 调高后，recall 续写会占用更多验证行（单步全批上限 64 行），续写错时浪费验证、并挤占同批其它请求的草稿行。
- 图像占位 token 也会被记录，不同图片的占位序列可能误配；验证会拒掉。
- 正常重启（不带 --set）会回到官方镜像，补丁不会自动保留。
