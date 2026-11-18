# 故障排查

这一页按现象列出常见问题、怎么定位、怎么恢复。

## 先跑这三条

```
yee doctor            # 版本、各适配器需要的符号/签名/源码标记是否都在，钩子是否已装
yee detect            # 识别到的引擎和加速头（JSON）
yee status            # 记录目录和 stats.json
```

`doctor` 必须在服务进程所用的 Python 解释器（镜像）里跑，否则会显示 sglang / vllm 不可导入。

## 没有加速

| 现象 | 原因 | 处理 |
|---|---|---|
| 日志没有 `[recall-accel]` 安装信息 | 钩子没装，或被关掉 | Mia 方案用 `bash install.sh --check` 再 `bash install.sh --restart`；其它机器在服务的解释器里 `yee install-hook` 后重启服务。4.2 起开关默认打开，确认没有显式的 `YEE=0` / `RECALL_ACCEL=0` |
| 召回统计 `[recall] steps=...` 停在很小的数不增长，或服务卡住 | 只有部分机器装了 Yee，或各台版本不同，TP rank 之间不一致 | 每台机器都装同一版本（`install.sh` 会做）；4.2 起会打印 `[recall] rank check on: N ranks` 或 `rank check off: <原因>`，off 时召回在所有 rank 上一起关闭，服务照常 |
| 显式设置的旧环境变量覆盖了 4.2 的默认值 | env 文件或启动脚本里还留着以前设的 `YEE_*` / `RECALL_ACCEL_*` / `DSV41_RECALL*` 等 | 在容器里 `python3 -m yee config` 看每个开关的值与来源（env / default / off），删掉不需要的显式设置 |
| 日志里有 `[yee-eb] rank check off` 或没有 `[yee-eb] armed geometry` | 热专家验证预算没起作用：部分 rank 没装，或 MoE 不在 `moe_b12x_next` 确定性路径上 | 见 [expert-budget.md](expert-budget.md) 的生效条件；其它加速不受影响 |
| 日志里有 `RECALL_ACCEL_ADAPTER=off: no hooks` | 适配器被关掉 | 去掉 `RECALL_ACCEL_ADAPTER` 或设为 `auto` |
| `doctor` 某一项 FAIL | 引擎版本改了符号或签名，钩子保持不生效以免出错 | 把 `doctor` 输出连同引擎版本一起反馈；服务照常运行，只是没加速 |
| `auto` 显示 `status=unsupported` / `no-op` | 这个加速头不支持（如 SGLang FROZEN_KV_MTP、vLLM NGRAM / SUFFIX、EAGLE topk>1） | 见 [heads.md](heads.md)；SGLang 没有加速头时可改用 `--speculative-algorithm NGRAM` |
| 首次请求不快、重复请求才快 | 正常：召回只对记录里已有的内容有效 | 用 `yee bench --url URL --model M` 看第二轮相对第一轮的倍数，>1.1 说明召回在工作 |
| 采样请求不加速 | `RECALL_ACCEL_NONGREEDY=0`，或草稿器保存了草稿分布，或是 cuteafd（只支持贪心） | 见 [nongreedy.md](nongreedy.md) |
| vLLM 完全不参与 | 没走 V2 runner | 设 `VLLM_USE_V2_MODEL_RUNNER=1` |
| TensorFold 不加速 | 启动没加 `--recall`，或版本低于 1.0.12，或引擎本身不读 Recall（Qwen4、Kimi Metal、Drama、Mac 上的 Flash Next 等） | `yee tensorfold detect` 看是否支持；DeepSeek-V4.1-Flash 4x DGX Spark 配方默认没加 `--recall`，需手动加 |
| TensorFold 设了 lookup 计划后召回不起作用 | priced 提议器给空时上游不问 Recall | 打 Yee 补丁 0001，重编，`TF_RECALL_AFTER_PRICED=1` 启动，见 [tensorfold.md](tensorfold.md) |

## 记录问题

| 现象 | 处理 |
|---|---|
| 换了模型或分词器后记录不生效 | 记录按分词器身份区分，身份不同就不会混用；这是预期行为。必要时用 `RECALL_ACCEL_TOKENIZER_DIR` 指定正确的分词器目录 |
| 磁盘空间不够时不再记录 | 剩余空间低于 `RECALL_ACCEL_MIN_FREE_GIB`（默认 4 GiB）时停止写入；清理磁盘或调小这个值 |
| 想清空记录 | `yee clear`（写入 `clear.request`，服务下次检查时清空）；TensorFold 用 `yee tensorfold clear --url <地址>` |
| 想暂停记录 | `yee pause on`，恢复用 `yee pause off` |
| `status` 显示的目录不对 | 用 `yee status --dir <目录>` 明确指定，或检查 `RECALL_ACCEL_DIR` |

## 补丁问题

| 现象 | 处理 |
|---|---|
| `refusing to apply ... do not match MANIFEST.json` | 包内补丁文件被改过或损坏；重装 wheel |
| `git apply --check failed` | 源码不是补丁的基线版本，见 [patches.md](patches.md) 的基线表；先切到基线 commit 再打 |
| `already applied` | 已经打过，返回 0，无需操作 |
| mlx-serve 报 `src/engines/mlx/spec is missing` | `--src` 要指向 mlx-serve 源码根目录 |
| 打完补丁行为没变 | 补丁改的是源码，需要重新编译并重启服务；TensorFold 还要设 `TF_RECALL_AFTER_PRICED=1` |

## 升级自 2.1

- 旧命令 `recall-accel`、旧包名 `recall_accel`、旧环境变量都继续可用。
- 2.1 写的 `recall_accel.pth` 钩子仍能工作；想换成新钩子：`yee uninstall-hook`（会一并删掉旧的 `recall_accel.pth`）再 `yee install-hook`。
- 记录目录默认位置没变（`~/.cache/recall_accel/...`），升级后已有记录直接可用。

## 自检

```
yee selftest          # 跑包内全部测试，逐文件给 PASS / FAIL
yee selftest -v       # 同上，打印全部输出
```
