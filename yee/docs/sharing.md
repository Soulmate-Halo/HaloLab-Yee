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
