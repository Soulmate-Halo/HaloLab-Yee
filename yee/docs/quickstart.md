# 快速开始

这一页给出从零到跑起来的完整步骤：装包、装钩子、配环境变量、验收、出问题怎么退。

## 0. 前置

- 服务已经在跑，并且有一个投机加速头（speculative head）。没有加速头的话，SGLang 可以加 `--speculative-algorithm NGRAM`，这样不需要草稿模型也能用上复制草稿。
- 你有服务所在机器的 shell 权限，以及服务启动脚本的修改权限。
- 知道服务用的是哪个 Python 解释器（`ps` 看启动命令，或进容器 `which python3`）。

## 1. 装包

在**服务实际使用的 Python 环境**里执行：

```bash
pip install --no-deps yee-2.2.0-py3-none-any.whl
```

只依赖 numpy。`--no-deps` 避免动服务环境里的其它包。

装完立刻自检：

```bash
yee doctor
yee selftest
```

`doctor` 会列出各适配器需要的符号、签名、源码标记；有任何 FAIL 就**停下**，不要改服务源码，把输出交给人工处理。`selftest` 跑自带的测试组，不需要 pytest。

## 2. 让 Yee 认出你的服务

两种方式，二选一：

```bash
# 方式 A：把启动命令直接给它
yee auto --cmdline "python3 -m sglang.launch_server --model-path /models/DeepSeek-V4.1-Flash --speculative-algorithm EAGLE3 --speculative-eagle-topk 1"

# 方式 B：让它从进程表里找正在跑的服务
yee auto
```

`auto` 会输出：识别到的引擎和加速头、选中的适配器和它的状态、每个符号的检查结果（OK/FAIL）、以及需要给服务进程加的环境变量。

如果加速头不在支持矩阵里，或者符号检查 FAIL，`auto` 会写明「not accelerated」，不写钩子，退出码为 3，服务保持原样运行。

支持矩阵见 [heads.md](heads.md)。

## 3. 装钩子

```bash
yee auto --apply
```

这会把 `yee.pth` 写进当前解释器的 site-packages，解释器启动时自动导入钩子。

**例外：SGLang 的 Mia 镜像**（启动时会先执行它自己的 sitecustomize）不要用 `.pth` 方式，改用模板：

```bash
cp $(python3 -c "import yee,os;print(os.path.dirname(yee.__file__))")/integrations/sitecustomize_template.py /opt/yee/sitecustomize.py
export PYTHONPATH=/opt/yee:$PYTHONPATH   # 必须排在最前面；它会先执行 Mia 原来的 sitecustomize
```

## 4. 加环境变量

按 `auto` 的输出给服务进程加环境变量。典型的一套：

```bash
export YEE=1
export YEE_BACKEND=memory        # 想跨重启保留记录就用 disk
export YEE_GIB=0.25              # 记录池大小，详见 env.md
export YEE_TOKENIZER_DIR=/models/DeepSeek-V4.1-Flash   # 记录按 tokenizer 隔离
# vLLM 还要加：
export VLLM_USE_V2_MODEL_RUNNER=1
```

`RECALL_ACCEL_*` 和 `YEE_*` 等价，写哪个都行，`YEE_*` 优先。全部变量见 [env.md](env.md)。

## 5. 重启并确认

重启服务，确认日志里出现 `[recall] patched ...`（SGLang / vLLM）或 `[recall-accel] patched`。

每 30 秒还会打印一行统计：

```
[recall] steps=... used=... landed=...
```

## 6. 验收

```bash
yee bench --url http://127.0.0.1:8000 --model <served-model-name> --rounds 2
```

`bench` 把同一条贪心请求连发两次。验收标准有两条：

- 必须输出 `replies identical: True`（逐字一致）；
- 第 2 次比第 1 次快（`last vs first: x>1.1`），说明复制草稿生效了。

任何一条不满足，就按 [troubleshooting.md](troubleshooting.md) 定位。

## 7. 运行中的控制

```bash
yee status      # 查看记录状态
yee pause on    # 暂停复制，立即生效，不用重启
yee pause off   # 恢复
yee clear       # 请求清空记录
```

## 8. 回滚

1. 只停复制：`yee pause on`。
2. 关钩子：设 `YEE=0` 或 `YEE_ADAPTER=off`，重启。
3. 彻底移除：`yee uninstall-hook && pip uninstall yee`（用 sitecustomize 模板的还要从 PYTHONPATH 里去掉模板目录），重启。

## 9. 各基座的入口

- SGLang：[sglang.md](sglang.md)
- vLLM：[vllm.md](vllm.md)
- TensorFold：[tensorfold.md](tensorfold.md)（自带 Recall，通常不需要装钩子）
- cuteafd：[cuteafd.md](cuteafd.md)
- mlx-serve：[mlx-serve.md](mlx-serve.md)
