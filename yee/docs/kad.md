# KAD：逐请求自适应草稿数（Yee 3.2.0，全部基座）

KAD（per-request adaptive draft width）给每个请求单独决定本步的草稿数：引擎以草稿数上限 K = 12 启动，之后每一步每个请求各自决定本步送验的草稿数 k，按它最近的核对接受情况升降。改宽度不用重启，策略文件运行时热加载。KAD 内部任何异常都回退原生行为（满启动 K），只打印前 5 次，绝不让服务崩；开关默认关，关着时各分支与原版逐分支相同。
3.1.0 的 KAD 只在 vLLM 适配器 `vllm_d_recall` 上，覆盖 vLLM 的全部草稿头。3.2.0 把它扩到所有基座：vLLM、SGLang、llama.cpp、TensorRT-LLM、LMDeploy、mlx-lm、transformers、TensorFold、mlx-serve、cuteafd。各基座的开关、启动参数、限制见下面各节。
「启动 K」是引擎启动时定的草稿数上限，KAD 只在它以内给每个请求定宽。默认策略 kmax = 12，所以启动 K 要正好是 12；启动 K ≠ 12 时会打一行 WARNING，宽度上限变成 min(12, 启动 K)。

## 1. 支持表

状态含义：**verified** = 真机实测；**built** = 真实源码编译 + 单测通过；**source-checked** = 对真实源码做了结构核对 + CPU 模拟测试；**native** = 引擎侧省不了算力，原生跑满并只做统计；**unsupported** = 识别得出但没有可挂钩的点，不装、服务原样运行。

| 基座 | 头 / 路径 | 做法 | 状态 |
|---|---|---|---|
| vLLM | 全部草稿头（dspark、eagle、eagle3、mtp、draft_model、dflash、多模块 MTP） | 调度器按策略给每个请求定宽并截短送验数，worker 分 varlen / steps / native 三条路径 | DSpark：verified（生产实测 r629、r633）；其余头：3.1.0 起适配，未在生产实测 |
| SGLang | EAGLE、EAGLE3、NEXTN、MTP 别名、STANDALONE（topk=1 链式） | 复用 Mia 自带 adaptive 运行时，每批按 K_b = max k_i 选已捕获步数，草稿步数和核对长度真的按 K_b 跑 | source-checked |
| SGLang | topk > 1 树形草稿 | adaptive 运行时不支持（`adaptive_unsupported_reason`），原生跑 | native |
| SGLang | DSPARK、DFLASH、NGRAM、UNO、MULTI_LAYER_EAGLE | 块长或送验宽度整批固定，没有可切换的多套 graph，原生跑 | native |
| SGLang | 上游正式版 0.5.10.post1 | 没有任何按批或逐请求变草稿步数的机制，install 时打印原因不装 | unsupported |
| llama.cpp | `draft-simple`（草稿模型 `-md`） | 每个 slot 每步的草稿上限取 KAD 宽度，核对行数随之变少 | verified |
| llama.cpp | `eagle3`、`mtp` | 补丁让两个起草循环按每 slot 的 `dp.n_max` 提前停止 | built |
| llama.cpp | `ngram-simple`、`ngram-mod`、`ngram-map`、`ngram-cache` | ngram 一次产出整段候选，KAD 只在事后截断、统计也不记，原样运行 | native |
| TensorRT-LLM | 配置了 `draft_len_schedule` 的投机方法（Draft-Target、NGram、非静态循环的两模型 EAGLE3 等） | 每轮长度 = min(schedule 值, K_b)，走引擎已有的动态草稿长度路径 | source-checked |
| TensorRT-LLM | 静态 / chain 草稿循环（TRTLLM 注意力后端 + 贪心草稿的两模型 EAGLE3） | 引擎断言 `draft_len_schedule is None`，不能逐轮改长度，原生跑 | native |
| TensorRT-LLM | 一模型 EAGLE3 / MTP | 未配置 schedule 时长度固定；配置了则同上一行 | source-checked（未配置时原生跑） |
| LMDeploy | 0.18.0（deepseek_mtp、eagle3、dflash、hy3_mtp、qwen3_5_mtp） | 所有形状按常量 `num_speculative_tokens` 构建，改宽度要动 6 处以上形状逻辑 | native |
| mlx-lm | `speculative_generate_step`（`mlx_lm.generate --draft-model`、`mlx_lm.server --draft-model` 都走它） | 每轮草稿数 = min(KAD 宽度, 启动 K)，草稿模型真的只跑 k 步、目标模型只核对 k+1 个 | verified |
| transformers | 辅助生成：AssistedCandidateGenerator 及其子类（early exit、不同 tokenizer、USD）、SinglePositionMultiToken | 每轮 `num_assistant_tokens` = KAD 宽度，替代内置 heuristic 调度 | verified |
| transformers | prompt lookup（`prompt_lookup_num_tokens`） | 每轮 `num_output_tokens` = KAD 宽度 | verified |
| transformers | MTP（`use_mtp`）、DFlash（`speculation_type="dflash"`） | 草稿数由模型结构定、一次前向出全部草稿 | native |
| TensorFold | lane 引擎的 head 草稿（`Rule.headDepth`） | 返回的深度再取 min(·, 该 stream 的 KAD 宽度)，核对行数随之变少 | built |
| TensorFold | copy 窗口（上下文复制、Recall 复制、priced 提议器） | 长度不截，只用接受情况更新 ladder；Recall 行被接受时该 stream hot | built |
| mlx-serve | DSpark stage 轮，单请求 | 置信度策略选出的送验草稿数再取 min(·, KAD 宽度)，核对后按送验数和接受数升降 | source-checked |
| mlx-serve | DSpark 组轮的 stage 槽（`moduleGroupRound`） | 每个 stage 槽的可选行数 `avail` 再取 min(·, 该请求 KAD 宽度)，各槽本来就不等宽 | source-checked |
| mlx-serve | copy 轮与 d_recall 召回轮 | 长度不截，只用接受情况更新 ladder；召回轮接受数 > 0 时该请求 hot | source-checked |
| mlx-serve | DSpark 树轮（`dsparkTreeStep` 选中的分支） | 不改 | native |
| cuteafd | DSpark（RTX dSpark proposal + target verify） | 不加 KAD，原生跑；cuteafd 自带逐请求长度策略 | native |

按批定宽的引擎（SGLang、TensorRT-LLM、TensorFold、mlx-serve）整批按 K_b = max k_i 跑：草稿步数和核对长度真的按 K_b 来，低命中请求跟着高命中请求多起草几个；只截短接受数、算力照旧的做法禁止。

## 2. 策略文件（各基座通用）

一份 `kad_policy.json` 在所有基座通用，格式与 vLLM 版相同。策略是 JSON 对象，三种模式：
- `{"mode": "off"}`：k = 启动 K，不截短，等同原生。
- `{"mode": "fixed", "fixed": n}`：所有请求 k = clamp(n, 1, 启动 K)。
- `{"mode": "ladder", ...}`：每个请求一个状态 `{cur, streak, hot_until}`，新请求 `cur = base`。每次核对（本步实际送验 v > 0 个草稿，接受 a 个，a 不含 bonus token）：
  1. 全接受（a >= v）：`streak += 1`；`streak >= up_after` 时 `cur = min(cur + up_step, kmax)`，`streak = 0`。
  2. 没全接受：`streak = 0`；`down = "accepted"` 时 `cur = max(kmin, min(cur, a + down_slack))`；`down = "step"` 时 `cur = max(kmin, cur - down_step)`。
  3. 最后 `cur` 钳到 [kmin, kmax]，下一步 k = clamp(cur, 1, min(kmax, 启动 K))。
  4. 召回命中（只限带 Yee 召回的基座）：`hot_until = step + hot_ttl`，这段时间里 k = max(cur, hot_k)。

字段校验与默认值：整数字段拒绝 bool 和非整数；默认 fixed 5、base 5、kmin 5、kmax 12、up_after 1、up_step 2、down "accepted"、down_slack 1、down_step 2、hot_k 12、hot_ttl 8；`mode` 缺省为 "fixed"。默认策略（进程从没读到有效策略文件时使用）：
```json
{"mode": "ladder", "base": 5, "kmin": 5, "kmax": 12, "up_after": 1, "up_step": 2, "down": "accepted", "down_slack": 1, "down_step": 2, "hot_k": 12, "hot_ttl": 8}
```
按这份策略，请求从 5 开始，连续全接受时依次是 5、7、9、11、12，之后保持 12；某次只接受 a 个时降到 max(5, min(cur, a+1))。
热加载：覆盖策略文件即可，按 (mtime_ns, size) 判断变化，stat 最小间隔 1.0 秒。文件缺失、JSON 无效、mode 未知时保留上一份有效策略（无效时 errors 加 1）。加载成功打印 `[yee-kad] <基座名> policy loaded sha=<前12位> <规范化json>`。结构化输出 / grammar 请求：k = 启动 K，不截短。

## 3. vLLM

3.1.0 起 KAD 覆盖 vLLM 的全部草稿头，`K_max` 统一为 12，服务以 `num_speculative_tokens = 12` 启动一次。开关与环境变量：`VLLM_D_RECALL`（默认 0，总开关，KAD 需要它为 1）、`VLLM_D_RECALL_KAD`（默认 0，KAD 开关）、`VLLM_D_RECALL_KAD_POLICY`（默认 `<VLLM_D_RECALL_DIR>/kad_policy.json`）、`VLLM_D_RECALL_KAD_STATS_FILE`（默认 `<VLLM_D_RECALL_DIR>/kad_stats.json`，调度器侧统计，原子写）、`VLLM_D_RECALL_KAD_LOG_S`（默认 30，日志与统计间隔秒）、`VLLM_D_RECALL_KAD_RELOAD_S`（默认 1.0，策略文件 stat 最小间隔秒）。
启动参数（草稿数 12）原文，其它草稿头照旧只写 method 和 num_speculative_tokens：
```
--speculative-config '{"method":"dspark","num_speculative_tokens":12,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
--speculative-config '{"method":"eagle","num_speculative_tokens":12,"model":"<草稿模型路径>"}'
--speculative-config '{"method":"eagle3","num_speculative_tokens":12,"model":"<草稿模型路径>"}'
--speculative-config '{"method":"mtp","num_speculative_tokens":12}'
--speculative-config '{"method":"draft_model","num_speculative_tokens":12,"model":"<草稿模型路径>"}'
--speculative-config '{"method":"dflash","num_speculative_tokens":12,"model":"<草稿模型路径>"}'
```
确认生效的日志行原文：
```
[vllm-d-recall] kad(worker) head=<类名> method=<method> k=12 path=<varlen|steps|native>
[vllm-d-recall] kad(sched) policy loaded sha=<前12位> <规范化json>
```
启动 K 不是 12 时多打一行：`[vllm-d-recall] kad(worker) WARNING launch num_speculative_tokens=<K>, KAD needs 12 (per-request width is capped at min(12, k))`。统计文件默认路径：`<VLLM_D_RECALL_DIR>/kad_stats.json`。
各草稿头的路径：dspark 走 varlen（每请求只生成 w_i 个草稿）；eagle / eagle3 / mtp / draft_model 走 steps（本步只跑 max(w_i) 步）；dflash 与多模块 MTP 走 native（原生出满 12，由调度器按请求截短）；V1 runner 下的 ngram / suffix 等走 native。
限制：MTP 系的 `n_predict` 不整除 12 时 vLLM 会报错，要取不超过 12 的最大合法值（如 `n_predict=5` 时给 10），启动后 KAD 按实际 K 截短；ngram、suffix、medusa 等头在 vLLM 0.31 的 V2 runner 下不支持，要 `VLLM_USE_V2_MODEL_RUNNER=0`，这时只有调度器侧截短生效。回退：把策略文件改成 `{"mode": "fixed", "fixed": 5}`（约 2 秒生效）；关掉 KAD 并重启；把 `num_speculative_tokens` 改回 5 并重启。

## 4. SGLang

开关：`YEE_KAD=1`。需要事先 `yee install-hook`（写 yee.pth），YEE_KAD=1 时 Yee 的 autoload 在每个 SGLang 进程里调用 `sglang_kad.install()`。
启动参数（草稿数 12）原文：
```
YEE_KAD=1 python -m sglang.launch_server --model-path <target> \
  --speculative-algorithm EAGLE3 --speculative-draft-model-path <draft> \
  --speculative-num-steps 12 --speculative-num-draft-tokens 13 --speculative-eagle-topk 1 \
  --speculative-adaptive --speculative-adaptive-config /path/yee_kad_sglang_adaptive.json
```
`yee_kad_sglang_adaptive.json` 内容：`{"1": {"candidate_steps": [5, 7, 9, 11, 12]}}`。K_b 向上取到最近的已捕获步数，可以减少候选（例如 `[5, 8, 12]`）。
确认生效的日志行原文：
```
[yee-kad] sglang installed on sglang.srt.speculative.eagle_worker_v2.EAGLEWorkerV2 (pid N); needs --speculative-adaptive with candidate steps [5, 7, 9, 11, 12]
[yee-kad] sglang steps=... w_hist=... b_hist=...
```
启动 K 不是 12 时打印 `[yee-kad] sglang WARNING launch K=<K>, KAD needs 12 (per-request width is capped at min(12, K))`，宽度上限变成该值。统计文件默认路径：`<YEE_KAD_DIR>/kad_stats_sglang.json`（目录默认 `~/.cache/yee/kad`）。
限制：没有 GPU，未在 Mia 镜像里实测，状态为 source-checked（结构核对 + CPU 替身测试）；adaptive 运行时只支持 EAGLE / EAGLE3 topk=1，并禁用 dp_attention、two_batch_overlap、pdmux、multi_layer_eagle，这些情况下 KAD 不生效、服务照常；每个候选步数一套 draft graph + target verify graph，5 个候选的显存和捕获时间约为单一步数的 5 倍；召回命中时的 hot 加宽没有接到 SGLang 召回钩子上（召回替换草稿发生在定宽之后，不冲突）。

## 5. llama.cpp

补丁 `llama.cpp-kad` 加在 v0.9.0 上（在 `yee patch` 里）。开关：`--kad`，或环境变量 `LLAMA_ARG_KAD=1`，或 `YEE_KAD=1`。默认关；关闭时各分支与原版相同。
启动参数（草稿数 12）原文（这个版本必须显式写 `--spec-type draft-simple`：普通草稿模型的 GGUF 推断不出投机类型，投机和 KAD 都不会启动）：
```
llama-server -m <target.gguf> -md <draft.gguf> --spec-type draft-simple \
  --spec-draft-n-max 12 --spec-draft-n-min 0 --kad [--kad-policy ~/.cache/yee/kad/kad_policy.json]
```
确认生效的日志行原文：
```
[yee-kad] llama.cpp enabled: launch K=12, policy <策略文件路径>, stats <统计文件路径>
[yee-kad] llama.cpp steps=<n> mixed=<n> w_hist=<k:次数,...> mean_v=<x> mean_a=<x> tracked=<n> errors=<n> sha=<12 位>
```
启动 K 不是 12 时多打一行：`[yee-kad] llama.cpp WARNING launch K=<K>, KAD needs 12 (per-request width is capped at min(12, K))`。策略文件：`--kad-policy FNAME` / `LLAMA_ARG_KAD_POLICY`，否则 `YEE_KAD_POLICY`，否则 `$YEE_KAD_DIR/kad_policy.json`。统计文件默认路径：`<YEE_KAD_DIR>/kad_stats_llama.cpp.json`。
限制：只在 WSL CPU 上用 110M / 15M 小模型实测过草稿模型路径，EAGLE3、MTP 只编译未实跑，ngram 系列为 native；宽度按 slot 决定，slot 之间不要求等宽；单进程服务，不涉及多 rank 同步；统计文件按 `YEE_KAD_LOG_S` 间隔写，进程只活几秒时可能一次都没写。

## 6. TensorRT-LLM

开关：`YEE_KAD=1`，并在每个 worker 进程里装钩子（`yee install-hook` 后 Yee autoload 会在能导入 tensorrt_llm 的进程里调用 `trtllm_kad.install()`）。
启动参数（草稿数 12）原文，`trtllm-serve --extra_llm_api_options extra.yaml` 或 LLM API 的 `speculative_config`：
```yaml
speculative_config:
  decoding_type: Eagle3        # 或 DraftTarget / NGram 等支持 schedule 的方法
  max_draft_len: 12
  speculative_model_dir: <draft>
  draft_len_schedule: {1: 12}  # 打开逐轮动态长度路径；KAD 在这个上限内按请求降宽
```
确认生效的日志行原文：
```
[yee-kad] tensorrt-llm installed on TensorRT-LLM 1.2.1 (pid N); active when speculative_config.draft_len_schedule is set
```
启动 K 不是 12 时打印 `[yee-kad] tensorrt-llm WARNING launch K=<K>, KAD needs 12 (per-request width is capped at min(12, K))`。统计文件默认路径：`<YEE_KAD_DIR>/kad_stats_tensorrt-llm.json`。
限制：没有 GPU，未实测，状态为 source-checked；只在配置了 `draft_len_schedule` 的组合上生效，其余组合原样运行；新长度第一次出现会触发一次 CUDA graph 捕获（每个 batch size 档位 × 每个出现过的长度），显存比固定 12 多；一批里请求宽度不同时整批按 K_b 跑。

## 7. LMDeploy（native）

LMDeploy 0.18.0 的所有投机形状都按常量 `num_speculative_tokens` 构建（输入形状、CUDA graph、草稿循环次数），逐请求或逐批改宽度要动 6 处以上形状逻辑，KAD 在这里省不了算力，所以是 native：只做统计，不介入。
启动参数（草稿数 12）原文：
```
lmdeploy serve api_server <model> --speculative-algorithm <方法> --speculative-draft-model <draft> --speculative-num-draft-tokens 12
```
或 `SpeculativeConfig(method=..., num_speculative_tokens=12)`。确认生效的日志行原文（install 只打印这一行）：
```
[yee-kad] lmdeploy native: LMDeploy runs a fixed num_speculative_tokens for every batch (input shapes, CUDA graphs and the draft loop are built from it), so KAD cannot shorten drafting or verification; launch with 12 speculative tokens
```
统计文件默认路径：无（native 不写统计文件）。限制：草稿数固定 12；KAD 不介入，Yee 召回照常生效。

## 8. mlx-lm

开关：`YEE_KAD=1` 后 `from yee.integrations import mlx_lm_kad; mlx_lm_kad.install()`，或用包装命令（默认开，`YEE_KAD=0` 关）。
启动参数（草稿数 12）原文：
```
python -m yee.integrations.mlx_lm_kad server --model <target> --draft-model <draft> --num-draft-tokens 12
python -m yee.integrations.mlx_lm_kad generate --model <target> --draft-model <draft> --num-draft-tokens 12 --prompt "..."
```
mlx-lm 默认 `--num-draft-tokens 3`，不改成 12 时打印 `[yee-kad] mlx-lm WARNING launch K=3, KAD needs 12 (per-request width is capped at min(12, K))`。server 请求体里的 `num_draft_tokens` 同样是该请求的启动 K。确认生效的日志行原文：
```
[yee-kad] mlx-lm installed on mlx_lm 0.32.2 (patched mlx_lm.generate.speculative_generate_step); policy <策略文件路径>
```
统计文件默认路径：`~/.cache/yee/kad/kad_stats_mlx-lm.json`。
限制：Windows 上 mlx wheel 加载 `mlx.core` 失败，mlx 真跑在 WSL 的 Linux CPU 版（`mlx[cpu]`），Apple Silicon 上没测，代码路径相同；批处理生成（BatchGenerator）不支持草稿模型，KAD 只作用于单请求的投机路径；不同 tokenizer 的辅助生成下 v / a 按目标模型 token 计，单位略有差别，未单独实测；transformers 和 mlx-lm 都没有内置 grammar 约束的投机路径，KAD 不做特殊处理。

## 9. transformers

开关：`from yee.integrations import hf_kad; hf_kad.enable(assistant_model)`（装钩子，并把助手的 `generation_config.num_assistant_tokens=12`、`num_assistant_tokens_schedule="yee_kad"`），然后照常 `model.generate(..., assistant_model=assistant_model)`；也可以 `hf_kad.install()` 后在 generate() 里传 `num_assistant_tokens_schedule="yee_kad"`。
启动参数（草稿数 12）原文：
```
model.generate(..., assistant_model=assistant_model, num_assistant_tokens=12, num_assistant_tokens_schedule="yee_kad")
model.generate(..., prompt_lookup_num_tokens=12, num_assistant_tokens_schedule="yee_kad")
```
`YEE_KAD=1` 时本进程所有辅助生成都用 KAD；`YEE_KAD=0` 全部关闭（schedule 写了 yee_kad 也按启动值跑）。助手默认 `num_assistant_tokens=20`，此时打印 `launch K=20` 警告，宽度上限仍是 12。`assistant_confidence_threshold`（默认 0.4）的提前停止照旧生效，KAD 只是每轮上限。确认生效的日志行原文：
```
[yee-kad] transformers installed on transformers 5.22.2; opt in with num_assistant_tokens_schedule='yee_kad' or YEE_KAD=1
```
统计文件默认路径：`~/.cache/yee/kad/kad_stats_transformers.json`。
限制：辅助生成只支持 batch size 1，一次 generate() 就是一个请求；MTP 与 DFlash 类是 native；结构化输出没有内置 grammar 约束的投机路径，KAD 不做特殊处理。

## 10. TensorFold

补丁 `tensorfold/0002-kad.patch`（在 `tensorfold/0001-recall-after-priced.patch` 之后打）。开关：`TF_KAD=1` 或 `YEE_KAD=1`（1 / true / yes / on），默认关；关时 `rule.kad` 为空，各分支与 0001 相同。
启动参数（草稿数 12）原文：TensorFold 没有草稿数的命令行参数，KAD 的启动 K = `cfg.most_drafts` = min(`Model.drafts`, 窗口宽度 - 1, `max_draft`)。DeepSeek V4.1 的 `Model.drafts` 来自模型 `config.json` 的 `"dspark_block_size"`（默认 5），把它设成 12（上限 `max_block = 16`）；其它族改 `Model.drafts`：
```
"dspark_block_size": 12
```
确认生效的日志行原文：
```
[yee-kad] tensorfold enabled: launch K=12, policy <策略文件路径>, stats <统计文件路径>
[yee-kad] tensorfold steps=... mixed=... w_hist=...
```
启动 K 不是 12 时打印 `[yee-kad] tensorfold WARNING launch K=<K>, KAD needs 12 (per-request width is capped at min(12, K))`。策略文件：`TF_KAD_POLICY` 或 `YEE_KAD_POLICY`，默认 `$YEE_KAD_DIR/kad_policy.json`。统计文件默认路径：`TF_KAD_STATS_FILE` 或 `YEE_KAD_STATS_FILE`，默认 `<目录>/kad_stats_tensorfold.json`。
限制：未在 GPU 上实测（无 GPU），宽度对速度的影响要上线后看统计；KAD 在 lane 引擎进程里做决定，假定一个 lane 引擎驱动所有 rank，若部署成多进程各跑一个 lane 引擎，策略热加载可能在不同进程的不同步生效；copy 窗口和 priced 提议器的长度不受 KAD 限制；统计里的 `tracked` 是当前在引擎里的 stream 数。

## 11. mlx-serve

补丁 `mlx-serve-kad` 以 source-drop 发布，先拷 `mlx-serve-2.4`，再拷 `mlx-serve-kad`。开关：`MLX_SERVE_KAD=1` 或 `YEE_KAD=1`（1 / true / yes / on），默认关；直接读进程环境，没有登记进 `env.zig`。
启动参数（草稿数 12）原文：mlx-serve 没有草稿数的命令行参数，DSpark 块长来自模型 `config.json` 的 `"dspark_block_size"`；核对上限 `VERIFY_MAX_DRAFTS = 15`，12 合法：
```
"dspark_block_size": 12
```
确认生效的日志行原文（第一次 stage 轮时打印）：
```
[yee-kad] mlx-serve enabled: launch K=12, policy <策略文件路径>, stats <统计文件路径>
[yee-kad] mlx-serve steps=... mixed=... w_hist=... b_hist= mean_v=... mean_a=... tracked=0 errors=... sha=...
```
块长不是 12 时打印 `[yee-kad] mlx-serve WARNING launch K=<K>, KAD needs 12 (per-request width is capped at min(12, K))`。策略文件：`MLX_SERVE_KAD_POLICY` 或 `YEE_KAD_POLICY`。统计文件默认路径：`MLX_SERVE_KAD_STATS_FILE` 或 `YEE_KAD_STATS_FILE`，默认 `<目录>/kad_stats_mlx-serve.json`。
限制：没在 M5 上编译和实测（M5 不可达），`dspark.zig` / `dspark_group.zig` 的类型检查要等 M5 上 `zig build check`；DSpark 块前向一次出 `ds_block` 个草稿，把块长设成 12 后低命中请求的起草算力比块长 5 时多，KAD 只能省它的核对行数，是否更快要在 M5 上 A/B；树轮不受 KAD 限制；结构化输出没有单独判断，按普通请求处理；`tracked` 恒为 0。

## 12. cuteafd（native，原因）

cuteafd 不加 KAD，原生跑，它自带逐请求长度策略。三条原因：
1. **启动 K 到不了 12。** `--dspark-draft-limit` 的取值范围是 1..=7，默认 5；草稿器只有 5 宽和 7 宽两套训练权重，`ensure!(matches!(width, 5 | 7), "draft width must be five or seven")`；`MAX_DSPARK_PROPOSALS = 7`。所以每请求上限只能是 min(12, K) ≤ 7。
2. **默认 K = 5 时没有可调空间。** 默认策略 kmin = 5、base = 5，K = 5 时每个请求恒为 5，与原生完全相同。
3. **K = 7 时原生已经逐请求选长度。** 原生按每请求的置信度和带宽成本模型返回保留的草稿行数，整批宽度由 `choose_width`（5 或 7）选；KAD 的 ladder 在 5～7 之间只会覆盖这套更细的定价，没有额外收益。

启动参数：cuteafd 的草稿宽度上限是 7，`--dspark-draft-limit` 的合法范围是 1..=7，填 12 会被参数校验拒绝。确认生效的日志行：无 KAD 日志，cuteafd 侧只有它自己的原生日志。统计文件默认路径：无（native 不写统计文件）。
限制：草稿宽度上限 7；recall v2 的召回草稿照常生效。

## 13. 统计字段与日志行

统计文件是原子写的 JSON，字段：`steps`（引擎步数）、`w_hist`（{宽度: 次数}）、`mixed_steps`（批内宽度不全相同的步数）、`v_steps`、`sum_v`、`sum_a`、`acc_by_v`（{送验数: [核对次数, 接受总数]}）、`tracked`（当前在跑并有状态的请求数）、`errors`（KAD 内部异常计数）、`policy_sha256`（策略文件原始字节的 sha256，可直接和 `sha256sum <策略文件>` 比，确认策略生效）、`kmax_run`（启动 K）、`b_hist`（{本批实际宽度: 步数}，只在按批定宽的引擎上有）。
日志：默认每 30 秒一行（`YEE_KAD_LOG_S`），前缀 `[yee-kad] <基座名>`：
```
[yee-kad] <基座名> steps=<n> mixed=<n> w_hist=<k:次数,...> mean_v=<x> mean_a=<x> tracked=<n> errors=<n> sha=<12 位>
```
策略加载成功打印 `[yee-kad] <基座名> policy loaded sha=<前12位> <规范化json>`。vLLM 沿用自己的前缀 `[vllm-d-recall] kad(sched)` / `kad(worker)`。

## 14. 多 rank

TP / PP / DP 多进程引擎里，所有 rank 每一步必须得到相同宽度，否则形状不一致会挂死集合通信。宽度只由各 rank 都相同的信息决定：同一批请求、同一接受数。
- SGLang：TP rank0 读策略文件，经 `get_tp_group().cpu_group` 用 `broadcast_object_list` 在 `step % YEE_KAD_RELOAD_EVERY == 0`（默认 64）时同步；adaptive 本身已禁用 dp_attention，只需 TP 内同步。
- TensorRT-LLM：TP rank0 读策略文件，经引擎已有的 TP 广播（`tp_broadcast`，root=0）同步，各 rank 调度同样的请求、采样同样的 token，长度一致。
- llama.cpp、TensorFold、mlx-serve、cuteafd 是单进程，不涉及多 rank 同步。

## 15. 验证状态说明

- **真机实测（verified）**：vLLM（3.1.0 生产实测）；llama.cpp 的 `draft-simple`（WSL CPU 小模型，三次输出逐字一致）；mlx-lm（WSL Linux CPU，mlx 0.32.4 真跑）；transformers（Windows venv，torch CPU 真跑）。这四处验证了输出正确性和宽度随接受数变化。
- **真实源码编译 + 单测通过（built）**：TensorFold（Zig 0.17 编译与单测通过，含整套 lane 回归，未做 GPU 推理）；llama.cpp 的 `eagle3`、`mtp`（编译通过，没有对应模型，未实跑）。
- **源码结构核对 + CPU 模拟测试（source-checked）**：SGLang（没有 GPU，未在 Mia 镜像实测）；TensorRT-LLM（没有 GPU，未实测）；mlx-serve（M5 不可达，整树编译和 M5 测试没做，`dspark.zig` / `dspark_group.zig` 只做了语法检查，其余文件在 WSL 的 Linux Zig 0.17 真编译并跑完单测）。
- **native**：LMDeploy、cuteafd，以及上表里标 native 的路径（省不了算力的头原生跑满，只做统计）。
- **unsupported**：SGLang 上游正式版 0.5.10.post1（没有可挂钩的变宽机制）。
