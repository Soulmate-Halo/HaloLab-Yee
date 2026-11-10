# Yee 2.2.0 使用文档

本文档是 Yee 2.2.0 的总览：它是什么、怎么装、各个基座怎么用、去哪一页看细节。

Yee 是一个「跨请求召回」加速模块：把服务读过、写过的 token id 记下来，当新请求和历史内容重复时，直接把后续内容复制成草稿，交给服务正在用的加速头去验证。复制来的每一个 token 都由目标模型逐个验证，所以贪心（temperature=0）输出与不装本模块时逐字一致，只是步数变少。

## 1. 名字与版本

| 项目 | 2.2.0 | 说明 |
|---|---|---|
| Python 包名 | `yee` | 2.2.0 的新包名 |
| 命令 | `yee` | 2.2.0 的新命令 |
| 旧包名 | `recall_accel` | 仍可导入，仍可用 |
| 旧命令 | `recall-accel` | 仍可执行，与 `yee` 等价 |
| 旧环境变量 | `RECALL_ACCEL_*` | 仍可读，优先级低于新的 `YEE_*` |

## 2. 安装

在**服务实际使用的 Python 环境**里执行（通常是镜像里的那个 python）：

```bash
pip install --no-deps yee-2.2.0-py3-none-any.whl
```

只依赖 numpy，镜像里一般已经有；`--no-deps` 可以避免它去动服务环境里的其它包。

装完先自检：

```bash
yee doctor          # 版本，以及各适配器需要的符号、签名、源码标记；全部 PASS 才算通过
yee selftest        # 自带的测试组，不需要 pytest
```

## 3. 各基座一句话快速开始

- **SGLang**：`yee auto --apply` 识别引擎和加速头并写入 `.pth` 钩子，按它打印的环境变量重启服务即可（镜像自带 sitecustomize 时改用模板，见 [sglang.md](sglang.md)）。
- **vLLM**：`yee auto --apply` 同上，另外必须加 `VLLM_USE_V2_MODEL_RUNNER=1`（钩子装在 vLLM V2 runner 上），见 [vllm.md](vllm.md)。
- **TensorFold**：TensorFold 1.0.12 起自带 Recall，不需要打补丁，直接 `tensorfold serve MODEL --recall`；需要 priced 提议器兜底时再打 Yee 的 TensorFold 补丁，见 [tensorfold.md](tensorfold.md)。
- **cuteafd**（DeepSeek V4.1 native serve，Rust）：`yee patch apply cuteafd --src <源码根> --variant v2`，重新编译后用控制目录里的 `enabled` 文件开关，见 [cuteafd.md](cuteafd.md)。
- **mlx-serve**（M5 上的 xypd 版，Zig）：`yee patch apply mlx-serve --src <源码根>`，替换四个源文件后重新编译，见 [mlx-serve.md](mlx-serve.md)。

## 4. 文档目录

| 文档 | 内容 |
|---|---|
| [index.md](index.md) | 全部 topic 的一页目录（`yee doc` 不带参数时列的就是它） |
| [quickstart.md](quickstart.md) | 从零到跑起来的完整步骤与验收方法 |
| [sglang.md](sglang.md) | SGLang 各加速头的接法、环境变量、sitecustomize 模板 |
| [vllm.md](vllm.md) | vLLM V2 runner 的接法与限制 |
| [tensorfold.md](tensorfold.md) | TensorFold 自带 Recall、Yee 补丁、引擎缺口 |
| [cuteafd.md](cuteafd.md) | cuteafd 补丁、开关、统计与回退 |
| [mlx-serve.md](mlx-serve.md) | mlx-serve 补丁、上线与回退 |
| [heads.md](heads.md) | 基座 × 加速头支持矩阵，含不支持项和原因 |
| [nongreedy.md](nongreedy.md) | 非贪心（采样）请求为什么无损，各基座怎么做到 |
| [sharing.md](sharing.md) | 全局共享的含义与隐私风险 |
| [env.md](env.md) | 全部环境变量表：名称、默认值、作用 |
| [patches.md](patches.md) | `yee patch` 用法、各补丁基线与回退方法 |
| [troubleshooting.md](troubleshooting.md) | 常见故障与定位步骤 |
| [changelog.md](changelog.md) | 2.2.0 相对 2.1.0 的变化 |

包内也可以直接看：`yee doc`（不带参数列出目录）、`yee doc <topic>`（看某一篇）、`yee bases`（看基座 × 加速头表）。

## 5. 与 2.1 的兼容说明

2.2.0 是 2.1.0 的改名升级版，**升级不需要改任何部署脚本**：

1. **旧命令还在**：`recall-accel auto --apply`、`recall-accel doctor`、`recall-accel selftest` 等与 `yee` 的同名子命令完全等价。
2. **旧包名还在**：`import recall_accel` 仍然可用；`recall_accel.core`、`recall_accel.integrations` 等模块路径不变。
3. **旧环境变量还在**：`RECALL_ACCEL_*` 全部继续生效，语义和 2.1.0 完全一致。
4. **新增 `YEE_*` 前缀**：`YEE_*` 与同名的 `RECALL_ACCEL_*` 等价（`YEE_GIB` = `RECALL_ACCEL_GIB`，`YEE_NONGREEDY` = `RECALL_ACCEL_NONGREEDY`，以此类推），并且优先于 `RECALL_ACCEL_*` 被读取。
5. **更老的名字仍在更底层**：2.1.0 支持的 `SGLANG_D_RECALL_*` / `VLLM_D_RECALL_*` / `DSV41_RECALL_*` / `RECALL_*` 回退链在 2.2.0 里原样保留，顺序见 [env.md](env.md)。
6. **新增能力不改变旧行为**：TensorFold 补丁、cuteafd 补丁、mlx-serve 补丁都要**显式打**才会生效；不设 `TF_RECALL_AFTER_PRICED` 时 TensorFold 的行为与上游完全一致。
7. **记录目录默认没变**：仍按 tokenizer digest 分目录，`RECALL_ACCEL_DIR` 不设时是 `~/.cache/recall_accel/<digest16>`；要沿用 2.0/2.1 的旧记录目录，显式把 `RECALL_ACCEL_DIR` 指过去。

## 6. 回滚

按影响从小到大：

1. **只停复制**：`yee pause on`（写 `paused` 文件），立即生效，不用重启。
2. **关闭钩子**：设 `YEE=0`（或 `RECALL_ACCEL=0`），或 `YEE_ADAPTER=off`，然后重启。
3. **彻底移除**：`yee uninstall-hook && pip uninstall yee`（用 sitecustomize 模板的，还要从 PYTHONPATH 里去掉模板目录），然后重启。
4. **回退补丁**：见 [patches.md](patches.md) 里每个补丁各自的回退方法。
