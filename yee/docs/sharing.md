# 全局共享与隐私

这一页说明召回记录是怎么共享的、会带来什么隐私影响，以及上线前该怎么管。

## 共享的含义

召回记录保存的是服务读过和写过的 token id。它不区分用户：

- 同一份记录服务这台机器（或这个记录目录）上的所有用户、所有请求；
- A 用户生成过的内容，B 用户再生成相同或相近内容时就能抄到草稿，速度变快；
- 用磁盘后端时记录在重启后还在。

| 基座 | 记录在哪 | 范围 | 重启后 |
|---|---|---|---|
| SGLang / vLLM（Yee 适配器） | `RECALL_ACCEL_DIR`，默认 `~/.cache/recall_accel/<分词器摘要前 16 位>`；`RECALL_ACCEL_BACKEND=memory` 或 `cuda` 时只在进程内 | 同一记录目录下的全部请求 | disk 后端保留，memory / cuda 清空 |
| TensorFold | `~/.cache/tensorfold/recall/<分词器身份>`，可用 `--recall-dir` 改 | 同一台机器上所有用户、所有请求 | 保留 |
| cuteafd | 进程内存 | 这个进程的全部请求 | 清空 |
| mlx-serve（M5） | 见 [mlx-serve.md](mlx-serve.md) | 这个服务的全部请求 | 见 mlx-serve.md |

## 隐私影响

1. **速度侧信道。** 后来的用户能从生成速度推测别人是否生成过相同内容：同样的请求明显变快，说明记录里有它。
2. **记录可还原成原文。** 记录目录里的 token id 用同一个分词器就能解码回文字。记录目录要按对话数据的级别保管：限制读权限、纳入数据保留和删除流程、备份时同样加密。
3. **输出本身不受影响。** 召回草稿都经过目标模型验证（见 [nongreedy.md](nongreedy.md)），一个用户的内容不会直接出现在另一个用户的回复里；泄露风险只来自上面两条。

## 怎么管

- **隔离：** 不同租户或不同保密级别的服务用不同的 `RECALL_ACCEL_DIR`（TensorFold 用不同的 `--recall-dir`），不要共用记录目录。
- **只读：** `RECALL_ACCEL_READ_ONLY=1`（TensorFold 用 `--recall-read-only`）只用已有记录、不再写入，适合用审核过的语料预热后上线。
- **不落盘：** `RECALL_ACCEL_BACKEND=memory`，进程退出即清空。
- **暂停与清空：** `yee pause on|off` 暂停或恢复记录；`yee clear` 请求清空；TensorFold 用 `yee tensorfold clear --url <地址>`。
- **目录权限：** Yee 创建记录目录时用 0700 权限；迁移或手工创建时保持同样的权限。
- **完全关闭：** 不设 `RECALL_ACCEL`（或设为 `0`），TensorFold 启动时不加 `--recall`。

## 多个进程共用一个记录目录（3.6.0）

- 同一台机器上的多个进程（多副本、多组 DP、strata 的几组副本）可以指向同一个记录目录：每个进程都写，别的进程写的记录最多 2 秒后也能复制。`RECALL_ACCEL_SHARED=0`（vLLM 是 `VLLM_D_RECALL_SHARED=0`，strata 是 `STRATA_RECALL_SHARED=0`）回到 3.5.x：第一个打开目录的进程一直持锁、只有它写入，其它进程只读，而且只在启动时读一次目录。
- 实现：追加时在 `lock` 文件上取一小会儿排他锁，先读进别的进程新写的 id 再接着写；`RecallEngine.tick()` 每 2 秒在共享锁下读一次。清空改写 `generation` 文件，其它进程下次同步时整份重读。读到的统计：`stats()` 的 `shared`、`synced_ids`、`shared_resets`、`lock_timeouts`。
- 只用于同一台机器的本地文件系统（依赖 flock），不要跨机器用 NFS 共享目录。
- 共用一个目录的进程要一起升级到 3.6.0，并用同样的容量（`RECALL_ACCEL_GIB`）。目录被 3.5.x（或设了 0）的进程占着时，3.6.0 的进程等 10 秒后退回只读。
- `stats.json` 由各进程轮流改写，看到的是最后写的那个进程。

## 提问哈希索引目录 `qhash`（3.7.0）

- disk 后端时，提问哈希索引（qhash）在记录目录下的 `qhash/` 子目录里，每个键一个文件 `<键>.i4`（int32 小端的 token id）。键是提示的哈希，分 4 级：整段提示、提示末尾 512 / 128 / 32 个 token。
- **多进程共用**：每次查找都是直接读文件，不缓存、不需要同步，所以指向同一个记录目录的多个进程立刻就能看到彼此写入的 qhash 条目（不像记录本体那样有最多 2 秒的同步延迟）。
- **跨重启**：qhash 条目是文件，重启后仍在，只要记录目录不换。
- **memory / cuda 后端**没有这个目录，索引只在进程内存里，重启即清空；要跨重启必须用 disk 后端（`RECALL_BACKEND=disk` 或不设，缺省就是 disk）。
- **容量**：最多保留 8192 个键文件，超出时按修改时间删最旧的。同一个键有了新回答，旧回答被覆盖。
- **清空**：`yee clear`（清库）时 `qhash/` 一并清空。
- **隐私**：与记录目录里的其它内容同级别——qhash 文件里是回答的 token id，用同一个分词器就能解码回文字；按记录目录的级别保管（限权限、纳入删除流程、备份加密）。另外，命中与否本身就是一个速度侧信道：同一个问题再问一次明显变快，说明这份索引里有它。
- 关闭：`RECALL_ACCEL_QHASH=0`（vLLM 是 `VLLM_D_RECALL_QHASH=0`），关闭后行为与 3.6.0 完全相同。
- 只在 Python 召回核心里：llama.cpp、TensorFold、cuteafd、mlx-serve、strata 这些补丁基座没有 qhash。
