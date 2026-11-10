# yee patch：源码补丁的用法、基线与回退

这一页说明 Yee 自带的三份源码补丁（TensorFold、cuteafd、mlx-serve）怎么查看、检查、应用和回退。

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
| `cuteafd` | git-patch，variants `v1`、`v2`（默认 `v2`） | cuteafd revision `9f53cc6cc0f3ff56de7cb81d587b3fad77137f4c` | `cuteafd/0001-recall-v1.patch`、`cuteafd/0001-recall-v2.patch`，说明 `cuteafd/NOTES-s839y-r572.md` | 源码根目录 |
| `mlx-serve` | source-drop | M5 xypd mlx-serve 源码（2026-11-10 r593 版） | `dspark.zig`、`dspark_group.zig`、`d_recall.zig`、`d_recall_live.zig` | `src/engines/mlx/spec/` |

### tensorfold

```
yee patch apply tensorfold --src <TensorFold 仓库根> --check
yee patch apply tensorfold --src <TensorFold 仓库根>
# 按 TensorFold 自己的构建方式重新编译（Zig 版本见仓库 .zig-version，当前 0.17.0）
TF_RECALL_AFTER_PRICED=1 tensorfold serve MODEL --recall
```

改动 `zig/src/core/lanes/windows.zig`、`config.zig`、`recall_engine_test.zig`，共增 74 行、删 7 行。不设 `TF_RECALL_AFTER_PRICED` 时行为与上游完全一致。详见 [tensorfold.md](tensorfold.md)。

### cuteafd

```
yee patch apply cuteafd --src <cuteafd 源码根> --variant v2 --check
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

## 回退

| base | 回退方法 |
|---|---|
| `tensorfold` | 仓库根目录执行 `git apply -R "$(yee patch path tensorfold)"`，重新编译；或者不重编、只去掉 `TF_RECALL_AFTER_PRICED`，行为即与上游一致 |
| `cuteafd` | 源码根目录执行 `git apply -R "$(yee patch path cuteafd --variant v2)"`（没有 git 用 `patch -p1 -R -i <同一路径>`），重新编译；不重编时删掉控制目录里的 `enabled` 文件即关闭召回 |
| `mlx-serve` | 把 `DIR/.yee-backup-<时间戳>/src/engines/mlx/spec/` 下的文件拷回 `DIR/src/engines/mlx/spec/`，重新编译 |

回退前先确认补丁版本：`yee patch list` 显示的 variant 要和当初应用的一致。

## 补丁清单的完整性

`yee/patches/MANIFEST.json` 里每个文件的 sha256 是构建时由 `tools/build_manifest.py` 算出写死的；`yee selftest` 里的 `test_yee_pkg.py` 会逐个核对，并检查包内没有清单之外的文件。
