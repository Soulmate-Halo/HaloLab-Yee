# llama.cpp 的 MTP 草稿头：放 CPU 还是 GPU，截断词表（3.5.2）

适用于 llama.cpp 的 `--spec-type draft-mtp`，模型是 qwen35 / qwen35moe 架构（Qwen3.5、Qwen3.6 带 MTP 层的 GGUF，例如魔搭 `unsloth/Qwen3.6-27B-MTP-GGUF`、`unsloth/Qwen3.5-27B-MTP-GGUF`）。

## 1. 用法

### 1.1 草稿头截断词表（补丁 `llama.cpp-mtp-vocab`）

```
yee patch apply llama.cpp-mtp-vocab --src <llama.cpp 仓库根> --check
yee patch apply llama.cpp-mtp-vocab --src <llama.cpp 仓库根>
cmake --build build --target llama-server
YEE_MTP_VOCAB_N=131072 llama-server -m <带 MTP 的 gguf> -ngl 99 -fa on --spec-type draft-mtp --spec-draft-n-max 3
```

- 基线 llama.cpp 提交 `a3b9c23`（2026-08-21）。补丁只改 `src/models/qwen35.cpp`（MTP 图的输出头），两处锚点在 v0.9.0（cd959bae）里也各出现一次。
- `YEE_MTP_VOCAB_N=N`（0 < N < 词表大小）：MTP 草稿头只用输出权重的前 N 行算 logits，再把 logits 补回整个词表宽度，补的位置是 -1000。主机侧缓冲、拷贝、采样器都不用改；主模型不受影响。
- 不设或设 0：与上游逐位一致。草稿都经主模型核对，输出不变。
- 生效日志：`[yee-mtp-vocab] MTP draft head limited to first 131072 of 248320 tokens`。
- N 怎么选：用 `yee/tools/mtp_bench/vocab_coverage.py` 量生成 token 的 id 分布。Qwen3.6 的中文 token 多在 98K～131K，N=131072 覆盖 98%；N=98304 只覆盖 82%，接受率掉得比省下的多。

### 1.2 草稿头放 CPU（原版 llama.cpp 就支持）

```
llama-server -m <gguf> -ngl 99 -fa on --spec-type draft-mtp --spec-draft-n-max 3 \
  -md <同一个 gguf> -devd none -ngld 0 -td 16
```

草稿上下文用单独加载的 CPU 模型实例，内存里多一份模型（Q4_K_M 约 17 GB）。qwen35 的 `ctx_other` 为空，草稿上下文有自己的 KV，验证后照常补写。

## 2. mx4 实测（2026-11-15）

环境：AMD Ryzen AI Max+ 395（16 核，AVX-512），Radeon 8060S（gfx1151），124 GiB LPDDR5X 由 CPU 与 GPU 共用；Ubuntu 24.04，llama.cpp a3b9c23 Vulkan 构建；模型 `Qwen3.6-27B-Q4_K_M.gguf`（17,106,773,120 字节，64 层 + 1 层 MTP，词表 248,320）。测速用 `mtp_bench`：15 条提示，贪心，max_tokens 256，单并发，每个配置单独起服务。

### 2.1 单独跑的速度

| 测试 | 结果 | 折算 |
|---|---:|---|
| GPU 解码 1 个 token | 12.50 tok/s | 80 ms，约 213 GB/s |
| GPU 一次算 2 / 4 / 8 个 token | 81 / 93 / 200 ms | 8 个的批是 4 个的 2.2 倍 |
| CPU 解码（16 线程，不用 GPU） | 6.30 tok/s | 约 108 GB/s |

### 2.2 草稿头放哪里

| 配置 | tok/s | 对比无投机 | 接受率 |
|---|---:|---:|---:|
| 无投机 | 12.45 | 1.00 | - |
| 草稿在 GPU，k=1 / 2 / 3 | 20.12 / 24.22 / 24.29 | 1.62 / 1.95 / 1.95 | 86.1% / 77.4% / 68.5% |
| 草稿在 CPU，k=1 / 3 | 17.90 / 20.60 | 1.44 / 1.65 | 86.4% / 68.4% |
| 草稿在 GPU，k=3，N=131072 | 25.50 | 2.05 | 66.5% |
| 草稿在 GPU，k=2，N=131072 | 24.82 | 1.99 | 75.9% |
| 草稿在 CPU，k=3，N=131072 | 20.70 | 1.66 | 66.4% |
| 草稿在 GPU，k=3，N=98304 | 22.15 | 1.78 | 53.3% |
| 打补丁不设变量，GPU，k=3 | 24.33 | 1.95 | 68.5% |

- 所有 k=3 配置（GPU / CPU 草稿、N=131072 / 98304、补丁关闭）的 15 条输出与原版 GPU k=3 逐字一致。与无投机相比，所有投机配置都有同样的 8 条输出不同，来自验证时一次算 k+1 个 token 与单 token 解码的数值差异。
- k=3 时每轮平均出 3.05 个 token：GPU 草稿每轮 126 ms，CPU 草稿每轮 148 ms。CPU 每步草稿约 16 ms，GPU 约 11 ms。
- 截断词表对 GPU 草稿快 5.0%（k=3），对 CPU 草稿只快 0.5%：CPU 每步的大头不在读输出头权重上。

### 2.3 CPU 和 GPU 同时干活

| 情形 | 单独 | 同时 | 变化 |
|---|---:|---:|---:|
| GPU 算 4 个 token，CPU 16 线程同时解码 | 43.11 t/s | 37.97 t/s | -11.9% |
| GPU 解码，CPU 只开 4 线程 | 12.50 tok/s | 11.17 tok/s | -10.6% |
| CPU 解码，GPU 同时解码 | 6.30 tok/s | 1.79 tok/s | -72% |

## 3. 结论

1. CPU 与 GPU 共用内存的机器（Strix Halo、DGX Spark、Apple M 系列）：草稿头留在 GPU。放 CPU 串行跑慢 15%；和 GPU 验证同时跑，CPU 只剩 28% 的速度，GPU 还慢 10%。
2. “GPU 验证时 CPU 起草下一轮”还受依赖限制：下一轮第一个草稿要用本轮验证出的 bonus token 和 hidden。只能先假设本轮全部接受、猜 bonus 再起草（PEARL 式）。mx4 的接受率下，k=3 全部接受约 51%，再乘 bonus 猜中约 70%，命中约 35%。
3. strata 在 Strix Halo 上也把 CPU/GPU 异步路径关了（setup 写 `DS4_ASYNC_T=4`，同步比异步快 9%～31%）。strata 用 CPU 算放不进显存的 MoE 专家，草稿层（MTP）放在 GPU。
4. CPU 有独立内存通道的机器（独显 + 主机内存，最好是多通道 DDR5 的服务器 CPU）先跑 `contention.sh`：GPU 忙时 CPU 能保持大部分速度，才值得把草稿头挪到 CPU 并与验证重叠。

## 4. 后续

- 按出现频率选草稿词表（取最常用的 32K～64K 个 token，加一张 id 映射表），比“前 N 行”省得多。
- MTP 层权重再降精度（strata 的 `--mtp-q4`）。
- Vulkan 8 个 token 的验证批是 4 个的 2.2 倍，限制了 k 和并发；换 ROCm/HIP 后端或调矩阵向量核再测。
- 复述、改写类请求叠加召回（`llama.cpp-recall`），召回的草稿不用算。

## 5. 重叠方案（没有实现，供 CPU 有独立内存通道的机器用）

位置：`tools/server/server-context.cpp` 里 `llama_decode(ctx_tgt, batch_view)` 之后（异步返回）、`common_speculative_process` 之前（会同步）。

1. 本轮起草后先不要裁掉草稿上下文里草稿链的 KV（上游在起草后裁回 `ckpt.pos_max + 1`）。
2. GPU 验证期间，在 CPU 上从最后一个草稿 d_k 和它的链式 hidden 继续算 m = k + 1 步，得到 e_1..e_m（e_1 是 bonus 的猜测），记下每步的 token、位置和 hidden。
3. 进 `process()` 前只裁 [n_past, n_past + k]，`process()` 会用主模型 hidden 重写这段；保留 n_past + k + 1 之后的预草稿 KV。llama.cpp 的 KV 不会按位置覆盖，同一位置不能留两份。
4. 下一轮：上一轮 k 个全部接受且 bonus == e_1 时，直接用 e_2..e_{k+1} 作为草稿，跳过草稿解码，链式 hidden 设为第 k+1 步的输出；否则裁掉预草稿 KV，正常起草。
5. 预草稿要设时间上限（按前几轮验证耗时估），超时就停，避免 GPU 等 CPU。

## 6. 回退

- 不重编：去掉 `YEE_MTP_VOCAB_N` 重启，即与上游相同。
- 还原源码：仓库根执行 `git apply -R "$(yee patch path llama.cpp-mtp-vocab)"`，再重编 llama-server。
- 草稿头放回 GPU：去掉 `-md`、`-devd none`、`-ngld 0`、`-td`。
