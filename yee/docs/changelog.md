# 更新记录

这一页列出 Yee 2.2.0 相对 recall-accel 2.1.0 的变化。

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
