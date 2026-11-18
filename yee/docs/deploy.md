# 全适配与集群部署（Yee 4.3.0）

这一页回答三个问题：**我这个部署现在能不能加速、要改哪个参数、集群层面还差什么**。

先跑一次诊断，再照着报告里的可复制命令改。

## 1. 第一步：跑诊断

```bash
python -m yee doctor                    # 本机进程 + docker 容器
python -m yee doctor --hosts h1,h2      # 逐台 ssh 过去查
python -m yee doctor --url http://h:8000 --url http://h2:8000
python -m yee doctor --k8s              # 读 kubectl get pods -A
python -m yee doctor --config maya.json # Maya 的 JSON 启动配置
python -m yee doctor --cmd "python3 -m sglang.launch_server --model-path /models/X"
python -m yee doctor --json             # 输出 plan JSON（给脚本读）
```

常用开关：`--no-local` 只看容器，`--no-docker` 只看本机进程，`--local` 只看本机（远程调用时用）。

报告按实例给结论：**可加速 / 要改参数才能加速 / 要打补丁 / 不支持 / 无需处理**，每一行改动都能直接复制：

```
[i1] sglang（Mia dsv41）pid 1234 @ host-a · 模型 /models/X · TP4 EP1 · 投机头 DSPARK
     结论：可加速
     召回 开 · KAD 开 · 专家预算 开 · 树形 select · 记录共享 同机共享
[i2] vllm @ host-b 容器 vllm1 · 投机头 NONE
     结论：要改参数才能加速
     - [参数] --speculative-config '{"method":"mtp","num_speculative_tokens":3}'   原因：没开投机头，Yee 没有可以加速的路径（H0）
     - [环境变量] VLLM_USE_V2_MODEL_RUNNER=1   原因：Yee 的 vLLM 挂钩在 V2 runner 上（H4）
集群
  模型 /models/X：2 个副本，跨 2 台机器 → 记录不共享
     - [路由] sglang-router 用 --policy cache_aware   原因：跨机副本要按会话亲和路由（K4）
合计：可加速 1 · 要改参数 1 · 要打补丁 0 · 不支持 0 · 无需处理 0。退出码 1
```

退出码：

| 退出码 | 含义 | 该做什么 |
|---|---|---|
| `0` | 所有实例都已可加速或无需处理，集群层面也没有必改项 | 不用改，看运行中的 `[yee] effect:` 确认效果 |
| `1` | 有实例要改参数 / 要打补丁，或集群层面有必改项 | 按报告里每行的可复制命令改，改完重跑一次 |
| `3` | 一个实例都没找到，或全部实例都不支持 | 先确认服务怎么起的（`--cmd`），或按「不支持」那几行换引擎 |

不装任何东西也能跑诊断，`doctor` 只读。

## 2. 适配矩阵：15 个引擎分别怎么做

| 引擎 | 做法 | 支持的投机头 | 需要改的（可复制） |
|---|---|---|---|
| SGLang | 自动挂钩（Python 进程，装钩子即生效） | DSPARK / DFLASH / EAGLE / EAGLE3 / NEXTN / MTP / STANDALONE / UNO / NGRAM | EAGLE 家族要 `--speculative-eagle-topk 1`；FROZEN_KV_MTP 不支持，改 EAGLE |
| vLLM | 改参数后自动挂钩 | dspark / dflash / eagle / eagle3 / mtp | `VLLM_USE_V2_MODEL_RUNNER=1`；`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` |
| llama.cpp | 打补丁重编译 | draft-simple / eagle3 / mtp | `yee patch apply llama.cpp-recall --src <llama.cpp 源码根>`；`cmake --build build --target llama-server`；启动加 `--recall` |
| TensorRT-LLM | 自动挂钩，只做 KAD（逐请求自适应草稿数），召回不接 | MTP / EAGLE3 / draft-target / ngram | yaml 里 `speculative_config: {decoding_type: MTP, num_nextn_predict_layers: 3}`，并加 `draft_len_schedule`（C10） |
| LMDeploy | 不支持并换引擎（原生投机照常，Yee 只能统计） | deepseek_mtp / eagle3 / dflash / hy3_mtp / qwen3_5_mtp | 要用召回换 vLLM 或 SGLang |
| mlx-lm | 自动挂钩，只做 KAD，召回不接 | draft model | `--draft-model <小模型> --num-draft-tokens 3` |
| mlx-serve | 打补丁重编译 | DSpark / copy / recall | `yee patch apply mlx-serve --src <mlx-serve 源码根>`；`zig build -Doptimize=ReleaseFast` |
| TensorFold | 改参数后自动挂钩（引擎自带 Recall） | lane head / priced 提议器 | `tensorfold serve MODEL --recall`；priced 兜底再打 `yee patch apply tensorfold --src <仓库根>` |
| strata | 打补丁重编译 | MTP 单流窗口 / `--batch-mtp` 批处理槽 | `yee patch apply strata-recall --src <strata 源码根>`；`RECALL_ACCEL=1`；多副本 `--recall-dir <同一目录>` |
| Maya | 打补丁重编译 | MTP（GLM 多槽调度器） | `yee patch apply maya-recall --src <Maya 源码根>`；`cmake --build build --target maya`；`RECALL_ACCEL=1` |
| cuteafd | 打补丁重编译 | DSpark | `yee patch apply cuteafd --src <cuteafd 源码根> --variant v2`；重编；控制目录里放 `enabled` 文件 |
| TGI | 不支持并换引擎 | `--speculate` 不被 Yee 挂钩 | 换 vLLM 或 SGLang |
| Ollama | 不支持并换引擎 | 无 | 换 llama.cpp 的 `llama-server` |
| ktransformers | 不支持并换引擎 | 无 | 换 SGLang |
| 认不出（unknown） | 不支持并换引擎 | — | 用 `--cmd "<启动命令>"` 告诉 doctor 它是什么 |

各引擎的投机头细节见 [heads.md](heads.md)；补丁的基线与回退见 [patches.md](patches.md)。

## 3. 投机头规则 H0–H9

诊断报告里的每条改动都带一个规则号，对照下面这张表就能自己看懂。

| 规则 | 条件（怎么看出来） | 可复制的参数 | 原因 |
|---|---|---|---|
| H0 | 自动挂钩的引擎（sglang / vllm / trtllm / mlx_lm）没开投机头（`head=NONE`） | 模型自带 MTP/NextN 层时：SGLang `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`；vLLM `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`；TRT-LLM yaml `speculative_config: {decoding_type: MTP, num_nextn_predict_layers: 3}`。没有 MTP 层时：SGLang `--speculative-algorithm NGRAM`（不需要草稿模型）；vLLM `--speculative-config '{"method":"eagle3","model":"<EAGLE3 草稿目录>","num_speculative_tokens":3}'`；mlx-lm `--draft-model <小模型> --num-draft-tokens 3` | 没有投机头就没有可以加速的路径：召回只提供草稿，要有人核对 |
| H1 | SGLang 的 EAGLE 家族且 `--speculative-eagle-topk` 大于 1 | `--speculative-eagle-topk 1 --speculative-num-draft-tokens <num_steps+1>` | 树形草稿没有稳定的挂钩点，复制草稿只在 topk=1 的链上跑 |
| H2 | SGLang 用了 `FROZEN_KV_MTP` | `--speculative-algorithm EAGLE`（MTP 模型用 NEXTN 同一条路） | 它自己构造验证输入，没有可挂钩的点 |
| H3 | vLLM 的 `method` 是 ngram / suffix 等 Yee 不挂的头 | 改 `mtp`（模型有 MTP 层时）或 `eagle3`：`--speculative-config '{"method":"eagle3","model":"<EAGLE3 草稿目录>","num_speculative_tokens":3}'` | vLLM 的原生 ngram 能用，但 Yee 不参与，召回拿不到机会 |
| H4 | vLLM 用的是 Yee 的 `vllm_spec` 适配器（dspark / dflash / eagle / eagle3 / mtp） | `VLLM_USE_V2_MODEL_RUNNER=1` | Yee 的 vLLM 挂钩在 V2 runner 上，V1 上没有这些投机器类 |
| H5 | 投机头在支持矩阵里是 unsupported / no-op | 改到 [heads.md](heads.md) 里状态为 verified / container 的头 | 这类组合 Yee 不打钩子，服务原样运行 |
| H6 | strata 的 `--spec 0` 或没写 `--spec` | `--spec 4` | `--spec` 是 MTP 草稿数，0 等于关掉投机 |
| H7 | Maya 配置里没有 MTP（`installer.draft` 不是 mtp，环境里也没有 `MAYA_GLM_MULTI_MTP`） | `MAYA_GLM_MULTI_MTP=3`，并用带 NextN 层的 GGUF | 召回草稿要挂在 MTP 验证窗口里，没有 MTP 就没有窗口 |
| H8 | llama.cpp 既没有 `-md` 也没有 MTP 参数 | `-md <草稿 GGUF> --draft-max 8`，或按 [llama-mtp.md](llama-mtp.md) 用 MTP | 召回提议要交给草稿头核对 |
| H9 | TGI / Ollama / ktransformers / LMDeploy / 认不出 | Ollama → 换 `llama-server`；TGI、LMDeploy → 换 vLLM 或 SGLang；ktransformers → 换 SGLang | 这些引擎没有可挂钩的投机路径，Yee 不参与 |

## 4. 补丁引擎 P1–P4

C++ 引擎的 token 循环在 Python 之外，Yee 只能以源码补丁接入：打完补丁必须重编、重启。

### P1 llama.cpp

```bash
yee patch apply llama.cpp-kad --src <llama.cpp 源码根>        # 基线补丁
yee patch apply llama.cpp-recall --src <llama.cpp 源码根>     # 召回补丁（在 KAD 之后）
yee patch apply llama.cpp-mtp-vocab --src <llama.cpp 源码根>  # 用 MTP 草稿头时再打
cmake --build build --target llama-server
```

启动：`--recall --recall-dir <目录>`；KAD：`--kad` 或 `LLAMA_ARG_KAD=1`；MTP 词表：`YEE_MTP_VOCAB_N=131072`。

### P2 strata

```bash
yee patch apply strata-recall --src <strata 源码根>
cmake --build <构建目录> --target strata
```

启动：`RECALL_ACCEL=1` 或 `--recall`；多副本共用记录加 `--recall-dir <同一目录>`。

### P3 Maya

```bash
yee patch apply maya-recall --src <Maya 源码根>
cmake --build build --target maya
```

启动：`RECALL_ACCEL=1`、`RECALL_ACCEL_DIR=<目录>`、`MAYA_GLM_MULTI_MTP=3`。适用 Project Maya v1.1.66，详见 [patches.md](patches.md) 的 maya-recall 一节。

### P4 mlx-serve / cuteafd / tensorfold

```bash
# mlx-serve
yee patch apply mlx-serve --src <mlx-serve 源码根>
zig build -Doptimize=ReleaseFast

# cuteafd
yee patch apply cuteafd --src <cuteafd 源码根> --variant v2
# 按仓库自己的方式重编

# tensorfold：1.0.12 起引擎自带 Recall，不用补丁
tensorfold serve MODEL --recall
# priced 提议器给不出草稿时再打补丁
yee patch apply tensorfold --src <TensorFold 仓库根>
```

mlx-serve 的冷层变量是 `MLX_SERVE_D_RECALL_COLD_DIR`；cuteafd 由控制目录里的 `enabled` 文件开关。

## 5. 多流与高并发 C1–C11

| 规则 | 怎么看出来 | 改什么 | 要不要先测 |
|---|---|---|---|
| C1 | SGLang 开了投机，`--max-running-requests` 大于 `--cuda-graph-max-bs` | `--cuda-graph-max-bs <max_running>`（不超过 512） | 不用测：超出 graph 批上限的批会退出 CUDA graph，投机步明显变慢 |
| C2 | SGLang 开了投机，CUDA graph 批上限未知 | 确认 `--cuda-graph-max-bs` 不低于峰值并发 | 不用测，先确认现状 |
| C3 | SGLang 带 `--disable-cuda-graph`，或 vLLM 带 `--enforce-eager` | 去掉这两个参数 | 不用测：关了 graph 就没有投机加速 |
| C4 | SGLang 的 `--max-running-requests` 大于 64 | 不用改，知道就行：召回对单批超过 64 行的批整批跳过，64 行以内照常 | 不用测 |
| C5 | vLLM 的 `--speculative-config` 里有 `disable_by_batch_size=N`，而 `--max-num-seqs` 未知或大于 N | 把 N 提到峰值并发，或去掉这一项 | **要测**：GPU 有余量时才可能更快，先 `yee bench` 做 A/B |
| C6 | Maya 的 `--parallel N` 大于 `MAYA_GLM_MULTI_MTP_MAX`（默认 4） | `MAYA_GLM_MULTI_MTP_MAX=0`（总开）或 `=N` | **要测**：8×3080 实测，8 路放开后 65.1 tok/s，默认配置是 76.6 tok/s，放开反而更慢；只在 GPU 不是瓶颈（专家全在显存）时考虑 |
| C7 | strata 的 `--batch N` 大于 1 | 不用改，知道就行：多行批与 MTP 草稿同时生效，召回在每行窗口里起作用；`--batch` 不要超过显存允许的行数 | 不用测 |
| C8 | 峰值并发（`--max-running-requests` / 运行中请求数 / `--parallel` / `--batch`）达到 32 或更多 | 不用改，知道就行：高并发时投机本身的收益变小，召回只在业务有重复内容时才有可见收益；新内容为主的业务要靠投机头和自适应草稿数（KAD） | 不用测 |
| C9 | 用 KAD，且是上游 SGLang（非 Mia 镜像） | 不用改，知道就行：上游 SGLang 没有按批变草稿步数的机制，KAD 不生效，其余照常；Mia 镜像要 `--speculative-adaptive` | 不用测 |
| C10 | 用 KAD，且是 TensorRT-LLM，yaml 里没配 `draft_len_schedule` | yaml 里加 `draft_len_schedule`，按批大小缩短草稿长度 | 不配时草稿长度固定，Yee 在 TensorRT-LLM 上没有别的可做；配了以后高并发时自动缩短草稿 |
| C11 | 用 KAD，启动时的草稿数 K 不是 12 | 不用改，知道就行：宽度上限变成 min(12, 启动 K)；要满宽度就按 [kad.md](kad.md) 用 K=12 启动 | 不用测 |

## 6. 集群与部署形态 K1–K8

### K1 多机多节点：每台都装同一个版本

多机 TP / PP / Ray 多副本 / Mia worker，只要有一个 rank 所在机器没装 Yee，或各台版本不同，跨 rank 一致性检查会把召回和专家预算一起关掉（日志 `rank check off`）。

```bash
bash install.sh --hosts <节点列表>     # 逐台装同一版本
```

装完各台版本必须是 4.3.0。诊断命令：`python -m yee doctor --hosts <节点列表>`。

多卡并行参数本身不用改：`--tp-size` / `--tp`（张量并行）、`--pp-size`（流水并行）、`--ep-size` / `--ep`（专家并行）按你原来的配置走。唯一要留意的是 EP：专家预算只在 `EP_SIZE=1` 的确定性路径上起作用（见 K8）。

### K2 容器里的实例

Yee 要装到**容器里那个解释器**上，宿主机上的解释器没有用。容器重建后临时安装会丢，两种持久做法二选一：

```dockerfile
# 做法一：做进镜像
RUN pip install --no-deps yee-4.3.0-py3-none-any.whl && python -m yee install-hook
```

```bash
# 做法二：挂载 + 环境变量
docker run -v /opt/yee:/opt/yee -e PYTHONPATH=/opt/yee <镜像> ...
```

### K3 同一模型多副本在同一台机器：共享记录

所有副本用**同一个** `RECALL_ACCEL_DIR`，副本之间就共享记录（每 2 秒同步一次）：

```bash
export RECALL_ACCEL_DIR=/var/lib/yee/recall
export RECALL_ACCEL_BACKEND=disk      # 缺省就是 disk
```

容器里要把**同一个宿主目录挂到同一个路径**：

```bash
docker run -v /var/lib/yee/recall:/var/lib/yee/recall -e RECALL_ACCEL_DIR=/var/lib/yee/recall <镜像> ...
```

### K4 同一模型多副本跨机器：记录不共享，按会话路由

记录只在本机共享（依赖文件锁，不能放 NFS 共享目录）。跨机的副本之间没有共同的记录，所以要按会话或前缀做亲和路由，让相关请求落到同一台：

```bash
# SGLang 自带路由
sglang_router.launch_router --policy cache_aware --worker-urls <各副本地址>
```

```nginx
# nginx：按会话 ID 一致性哈希
upstream inference {
    hash $http_x_session_id consistent;
    server <副本地址>;
}
```

没有亲和路由时，同一个会话的两次请求落到不同机器，就都拿不到前一次的记录。

### K5 PD 分离（prefill / decode）

召回在 decode 节点起作用，**prefill 节点不用装**（装了也不影响）。要装的是 decode 那一侧。

### K6 路由节点本身

router / 负载均衡节点不用装 Yee。只有当路由策略是 round_robin / random、且后面有跨机副本时，才需要按 K4 改成亲和路由。

### K7 各台版本不一致

统一到 4.3.0。诊断里出现「多个实例 yee.version 不一致」时，先把落后的那几台升上来。

### K8 专家预算

热专家验证预算只在 Mia 的 `moe_b12x_next` 确定性路径上起作用（`DSV41_MOE_B12X_NEXT=1`、`DSV41_MOE_B12X_NEXT_DETERMINISTIC=1`、`EP_SIZE=1`）。其它路径上这一项不介入，其余加速照常；Mia 且 `EP_SIZE` 不等于 1 时不生效。

### K8s：装进集群

做法一，把 wheel 装进镜像：

```dockerfile
FROM <你的推理镜像>
RUN pip install --no-deps yee-4.3.0-py3-none-any.whl && python -m yee install-hook
```

做法二，用 initContainer 装到共享 volume，再用 `PYTHONPATH` 指过去：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm
spec:
  template:
    spec:
      initContainers:
        - name: yee-init
          image: <含 wheel 的镜像>
          command: ["python3", "-m", "pip", "install", "--no-deps", "--target", "/opt/yee", "/wheels/yee-4.3.0-py3-none-any.whl"]
          volumeMounts:
            - name: yee
              mountPath: /opt/yee
      containers:
        - name: vllm
          image: <推理镜像>
          env:
            - name: PYTHONPATH
              value: "/opt/yee"
            - name: VLLM_USE_V2_MODEL_RUNNER
              value: "1"
            - name: RECALL_ACCEL_DIR
              value: "/recall"
          volumeMounts:
            - name: yee
              mountPath: /opt/yee
            - name: recall
              mountPath: /recall"
      volumes:
        - name: yee
          emptyDir: {}
        - name: recall
          persistentVolumeClaim:
            claimName: recall-pvc
```

多副本共享记录时，`recall` 这个 volume 要让所有副本挂到**同一个** PVC（同 K3）。跨节点的 PVC 只能当普通文件目录用，不要指望跨机共享记录（见 K4）。

集群里查状态：`python -m yee doctor --k8s`。

## 7. 怎么看是否生效

### 启动时

服务启动后，认得出引擎的进程会打印（每个进程只打一次）：

```
[yee] 4.3.0 plan: sglang head=DSPARK -> 可加速
[yee] 要改: --speculative-eagle-topk 1  (H1 EAGLE 家族 topk>1)
[yee] 全部建议: python -m yee doctor
```

看到「要改」就照那几行改；不想看这些行就设 `YEE_NOTICE=0`。

### 运行中

```
[recall] steps=... used=... landed=... errors=0
```

`steps` 随请求增长就是钩子在跑。召回统计后面会跟一行 `[yee] effect:`（默认每 600 秒一次，间隔由 `YEE_EFFECT_S` 控制）：

| 行 | 含义 | 对策 |
|---|---|---|
| `[yee] effect: 召回在工作 used=… landed=… 每行约 x token` | 正常，召回在贡献 token | 无 |
| `E1`（`rows==0` 且 `steps>0`） | 钩子装上了，但投机路径没跑到 | 按报告里 H / C 类的改动改：多半是没开投机头、topk>1、vLLM 没开 V2 runner |
| `E2`（`landed/rows` 小于 0.02） | 召回有贡献但占比很小 | 业务重复内容少时正常；要么加投机头的参数，要么看 C 类改动；没有可改的就说明这类业务的收益主要来自投机头 |
| `E3`（`errors>0`） | 召回链路有错 | 看这一行前面的报错内容 |

### 各基座自己的行

| 基座 | 行 |
|---|---|
| SGLang / vLLM | `[recall] patched ...`、`[recall] rank check on: N ranks` |
| llama.cpp / strata / Maya / cuteafd / mlx-serve | `[yee-recall] <引擎>: on, backend …` |
| 全部 | `[recall] steps=...`、`[yee] 4.3.0 plan: …`、`[yee] effect: …` |

## 8. 收益预期：召回在哪类业务上有用

| 业务类型 | 收益主要来自 | 说明 |
|---|---|---|
| 同题重问 | 召回 | 提问哈希命中，从第 1 个输出 token 起复制上次那份回答 |
| 改写、润色 | 召回 | 与旧回答重叠的片段被复制，新写的部分照常算 |
| 代码编辑 | 召回 | 上下文和没改动的代码段重复出现，命中率高 |
| RAG 摘抄 | 召回 | 摘出来的大段原文会在回答里再出现一遍 |
| 长文复述 | 召回 | 复述时大量句子与原文相同 |
| 新写文章 | 投机头 | 内容不重复，召回几乎没有命中，速度来自投机头与自适应草稿数 |
| 新写代码 | 投机头 | 同上 |

**命中率高不等于收益大。** 某集群的提问哈希命中率是 29.1%，看着不错；但每次命中平均只接受约 17 token，而回答平均约 2762 token，召回只占总输出的 0.18%，整体提升 0%–1%——对方的结论是「没用」，这个结论是对的：召回贡献的 token 占比太小，撑不起整体速度。

所以：

- 业务里有大段重复内容（同题重问、改写、编辑、摘抄、复述）时，召回的收益会直接反映在整体速度上；
- 业务里没有重复内容（新写文章、新写代码）时，收益主要来自投机头本身，召回只是个加成；
- 先看 `[yee] effect:` 里 `landed` 与总输出的比例，再决定要不要继续调召回；比例长期很小就把精力放在投机头参数和 KAD 上。

## 9. 相关文档

- 诊断与参数总表：本页
- 投机头支持矩阵：[heads.md](heads.md)
- 补丁基线与回退：[patches.md](patches.md)
- 环境变量：[env.md](env.md)
- 故障排查：[troubleshooting.md](troubleshooting.md)
- KAD：[kad.md](kad.md)
- 记录共享与隐私：[sharing.md](sharing.md)
