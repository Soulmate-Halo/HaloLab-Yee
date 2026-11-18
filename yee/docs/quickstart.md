# 快速开始

这一页给出从零到跑起来的完整步骤：装包、装钩子、验收、出问题怎么退。

## 0. 前置

- 服务已经在跑，并且有一个投机加速头（speculative head）。没有加速头的话，SGLang 可以加 `--speculative-algorithm NGRAM`，这样不需要草稿模型也能用上复制草稿。
- 你有服务所在机器的 shell 权限，以及服务启动脚本的修改权限。
- 知道服务用的是哪个 Python 解释器（`ps` 看启动命令，或进容器 `which python3`）。
- 不确定服务能不能加速时，先跑 `python -m yee doctor`，见 [deploy.md](deploy.md)。

## 1. 4.3.0 最短用法

**第一步：跑一次部署诊断。**

```bash
python -m yee doctor          # 本机进程 + docker 容器；多机加 --hosts，集群加 --k8s
bash install.sh --check        # 或者用安装脚本的检查模式
```

报告告诉你每个实例现在能不能加速、要改哪个参数、要打哪个补丁，每条改动都能直接复制。全适配与集群部署的完整说明见 [deploy.md](deploy.md)。

4.2.0 起所有加速开关默认打开，不用设任何环境变量。显式设 0 可以单独关掉某一项；设 `YEE=0`（或 `RECALL_ACCEL=0`）整个 Yee 都不加载。

### Mia dsv41-4x-spark

在 head 机上：

```bash
bash install.sh --check      # 只检查，不改任何东西
bash install.sh --restart    # 装到 4 台机器，装完重启并做健康检查和冒烟
```

回退：

```bash
bash install.sh --uninstall --restart
```

`install.sh` 会自动找方案目录、4 台机器都装、在 env 文件末尾追加一段带标记的块（只写 `PYTHONPATH` 与 `YEE=1`）。其它机器上它自动走 `pip install` + `yee install-hook`。

### 其它 Python 推理服务

```bash
pip install yee-4.3.0-py3-none-any.whl && python -m yee install-hook
```

然后照常启动服务，不需要再手动加环境变量。

## 2. 装包

在**服务实际使用的 Python 环境**里执行：

```bash
pip install --no-deps yee-4.3.0-py3-none-any.whl
```

只依赖 numpy。`--no-deps` 避免动服务环境里的其它包。

装完立刻自检：

```bash
yee doctor
yee selftest
python -m yee config
```

`doctor` 会列出各适配器需要的符号、签名、源码标记；有任何 FAIL 就**停下**，不要改服务源码，把输出交给人工处理。`selftest` 跑自带的测试组，不需要 pytest。`config` 打印每个开关的生效值与来源（env / default / off）。

想看整个部署（多机、容器、集群）的结论，跑 `python -m yee doctor`；用法与退出码见 [deploy.md](deploy.md)。

## 3. 让 Yee 认出你的服务

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

## 4. 装钩子

```bash
yee auto --apply
```

这会把 `yee.pth` 写进当前解释器的 site-packages，解释器启动时自动导入钩子。

**例外：SGLang 的 Mia 镜像**（启动时会先执行它自己的 sitecustomize）不要用 `.pth` 方式，改用模板：

```bash
cp $(python3 -c "import yee,os;print(os.path.dirname(yee.__file__))")/integrations/sitecustomize_template.py /opt/yee/sitecustomize.py
export PYTHONPATH=/opt/yee:$PYTHONPATH   # 必须排在最前面；它会先执行 Mia 原来的 sitecustomize
```

## 5. 环境变量

4.2.0 起不用设环境变量就能用。要改的时候看 [env.md](env.md) 的「4.2 自动默认值」一节，`python -m yee config` 会告诉你每个开关当前是 env 设的还是默认的。

常用的一小批：

```bash
export YEE=1                      # 整体开关；设 0 则整个 Yee 不加载
export YEE_BACKEND=memory        # 想跨重启保留记录就用 disk
export YEE_GIB=0.25              # 记录池大小，详见 env.md
export YEE_TOKENIZER_DIR=/models/DeepSeek-V4.1-Flash   # 记录按 tokenizer 隔离
# vLLM 还要加：
export VLLM_USE_V2_MODEL_RUNNER=1
```

`RECALL_ACCEL_*` 和 `YEE_*` 等价，写哪个都行，`YEE_*` 优先。全部变量见 [env.md](env.md)。

## 6. 重启并确认

重启服务，确认日志里出现：

| 日志 | 含义 |
|---|---|
| `[yee] 4.3.0 plan: ...` | 4.3 启动诊断：能不能加速、要改什么 |
| `[yee] 4.3.0 defaults:` | 4.2 起的默认值已生效 |
| `[recall] patched ...`（SGLang / vLLM）或 `[recall-accel] patched` | 召回钩子已装上 |
| `[recall] rank check on` | 各 TP rank 装的是同一版本 Yee，召回广播与专家预算启用 |
| `[recall] steps=... used=... landed=...` | 召回统计，数字随请求增长 |
| `[yee-eb] armed geometry ...` | 热专家验证预算已就绪 |

每 30 秒还会打印一行召回统计：

```
[recall] steps=... used=... landed=...
```

各 TP rank 版本不一致时会打印 `[recall] rank check off: ...` / `[yee-eb] rank check off: ...`，此时所有 rank 一起关闭相关功能，服务照常运行。

## 7. 验收

```bash
yee bench --url http://127.0.0.1:8000 --model <served-model-name> --rounds 2
```

`bench` 把同一条贪心请求连发两次。验收标准有两条：

- 必须输出 `replies identical: True`（逐字一致）；
- 第 2 次比第 1 次快（`last vs first: x>1.1`），说明复制草稿生效了。

任何一条不满足，就按 [troubleshooting.md](troubleshooting.md) 定位。

## 8. 运行中的控制

```bash
yee status      # 查看记录状态
yee pause on    # 暂停复制，立即生效，不用重启
yee pause off   # 恢复
yee clear       # 请求清空记录
```

## 9. 回滚

1. 只停复制：`yee pause on`。
2. 关钩子：设 `YEE=0` 或 `YEE_ADAPTER=off`，重启。
3. 彻底移除：`yee uninstall-hook && pip uninstall yee`（用 sitecustomize 模板的还要从 PYTHONPATH 里去掉模板目录），重启。

Mia dsv41-4x-spark 用一键脚本回退：`bash install.sh --uninstall --restart`。

## 10. 各基座的入口

- 全适配与集群部署：[deploy.md](deploy.md)
- SGLang：[sglang.md](sglang.md)
- vLLM：[vllm.md](vllm.md)
- TensorFold：[tensorfold.md](tensorfold.md)（自带 Recall，通常不需要装钩子）
- cuteafd：[cuteafd.md](cuteafd.md)
- mlx-serve：[mlx-serve.md](mlx-serve.md)
