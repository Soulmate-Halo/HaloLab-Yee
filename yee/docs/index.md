# Yee 2.2.0 文档目录

这一页是 Yee 内置文档的目录，`yee doc` 不带参数时列出的就是下面这些 topic。

## 目录

| Topic | 文件 | 一句话说明 |
|---|---|---|
| `readme` | [README.md](README.md) | 总览：Yee 是什么、怎么装、各基座快速开始、与 2.1 的兼容说明 |
| `index` | [index.md](index.md) | 本页：全部 topic 的目录 |
| `quickstart` | [quickstart.md](quickstart.md) | 从零到跑起来的完整步骤：安装、装钩子、配环境变量、验收 |
| `sglang` | [sglang.md](sglang.md) | SGLang 各加速头的接法、环境变量、Mia 镜像的 sitecustomize 模板 |
| `vllm` | [vllm.md](vllm.md) | vLLM V2 runner 的接法、投机配置、已知限制 |
| `tensorfold` | [tensorfold.md](tensorfold.md) | TensorFold 自带 Recall 的用法、Yee 补丁、哪些引擎不读 Recall |
| `cuteafd` | [cuteafd.md](cuteafd.md) | cuteafd（DeepSeek V4.1 native serve）补丁：开关、统计、限制 |
| `mlx-serve` | [mlx-serve.md](mlx-serve.md) | mlx-serve（M5 xypd 版）补丁：替换哪些文件、上线与回退 |
| `heads` | [heads.md](heads.md) | 基座 × 加速头支持矩阵，含不支持项和原因 |
| `nongreedy` | [nongreedy.md](nongreedy.md) | 非贪心（采样）请求为什么无损，各基座怎么做到、怎么开关 |
| `sharing` | [sharing.md](sharing.md) | 全局共享的含义：谁能看到什么、隐私怎么管 |
| `env` | [env.md](env.md) | 全部环境变量表：名称、默认值、作用、回退顺序 |
| `patches` | [patches.md](patches.md) | `yee patch` 用法、各补丁的基线与回退方法 |
| `troubleshooting` | [troubleshooting.md](troubleshooting.md) | 常见故障、定位步骤、恢复办法 |
| `changelog` | [changelog.md](changelog.md) | 2.2.0 相对 2.1.0 的变化清单 |

## 怎么读

- 第一次用：先看 [quickstart.md](quickstart.md)，再按你的基座看 [sglang.md](sglang.md) / [vllm.md](vllm.md) / [tensorfold.md](tensorfold.md) / [cuteafd.md](cuteafd.md) / [mlx-serve.md](mlx-serve.md)。
- 想知道「我这个加速头支不支持」：看 [heads.md](heads.md)。
- 想在采样请求上也用：看 [nongreedy.md](nongreedy.md)。
- 上线前评估隐私影响：看 [sharing.md](sharing.md)。
- 要打补丁或回退：看 [patches.md](patches.md)。
- 出问题了：看 [troubleshooting.md](troubleshooting.md)。
