# 非贪心（采样）请求为什么无损

这一页说明 temperature>0、top-k、top-p 这类采样请求用上召回草稿后，输出分布为什么不变，各基座分别怎么做到，以及怎么关掉。

## 原理

投机解码里，草稿只是「提议」，每个 token 都要由目标模型验证。采样请求走的是拒绝采样：设目标模型在这个位置的分布为 p，草稿的分布为 q，草稿 token x 以 min(1, p(x)/q(x)) 的概率被接受；被拒绝时从残差分布 max(0, p − q) 归一化后重新采样。

召回草稿是从历史记录里「抄」来的确定的 token，没有自己的分布。只要验证时把它的 q 当成 one-hot（q(x)=1，其余为 0）：

- x 被接受的概率是 p(x)；
- 被拒绝时残差是「p 去掉 x 后归一化」；
- 两部分合起来，这个位置最终产出的 token 仍然严格服从 p。

p 是经过 penalty、temperature、top-k、top-p 处理后的目标分布，所以这些采样参数的效果都保留。抄得准，接受率就高、速度就快；抄得不准只是多一次被拒，不影响正确性。

## 各基座怎么做到

| 基座 | 做法 | 采样请求 |
|---|---|---|
| SGLang DSPARK | 验证前把召回位置的 `corrected_logits` 改成 one-hot（`onehot_rows`），接受 p(x)、残差为 p 去掉 x | 支持，`RECALL_ACCEL_NONGREEDY` 控制 |
| SGLang DFLASH / EAGLE / EAGLE3 / MTP / STANDALONE / UNO / NGRAM | 链式草稿（EAGLE 类要求 topk=1）直接替换 draft token；rejection-sampling 模式（`draft_probs` 非空）和 grammar 批次不参与 | 支持，`RECALL_ACCEL_NONGREEDY` 控制 |
| vLLM V2（DSpark / DFlash / EAGLE / MTP） | 投机器不保存 `draft_logits` 时（`draft_sample_method` 为默认的 greedy），vLLM 的拒绝采样器本来就把草稿当 one-hot（log q = 0）；保存了 `draft_logits` 的概率型草稿器，采样批次不参与 | 支持，`RECALL_ACCEL_NONGREEDY` 控制 |
| TensorFold（1.0.12 起内置 Recall） | 用带种子的确定性采样逐位验证，上游测试 "greedy and sampled" 覆盖 | 支持，TensorFold 自身行为 |
| TensorFold + Yee 补丁 0001 | priced 提议器给空时改问 Recall，验证路径同上；Windows 上的新增测试覆盖贪心、T=0.8、T=1.0 三档输出逐 token 一致且轮数更少 | 支持，`TF_RECALL_AFTER_PRICED=1` 开启 |
| mlx-serve（M5 xypd 版） | 贪心和采样请求都走召回，统计分开；带 penalty 的请求不走投机 | 支持 |
| cuteafd | 补丁只在贪心请求上生效 | 不支持，采样请求照常运行、不加速 |

## 开关

- `RECALL_ACCEL_NONGREEDY`（等价的新名字 `YEE_NONGREEDY`）：默认 `1`，采样批次也接受召回草稿；设为 `0` / `false` / `no` / `off` 后只有全贪心的批次参与。只影响 SGLang 和 vLLM 适配器。
- TensorFold、mlx-serve、cuteafd 不读这个变量，它们的行为见上表。

## 不参与的情况（任何基座都一样安全）

- structured output / grammar 批次；
- CUDA graph 捕获、dummy run、profile run；
- SGLang EAGLE 的 topk>1（真正的树）；
- 概率型草稿器（保存草稿分布）的采样批次。

这些情况下服务按原样运行，只是这一步拿不到召回加速。

## 怎么自己验证

- 包内测试：`yee selftest` 里的 `test_nongreedy.py` 用模拟的拒绝采样核对分布。
- 线上：同一个采样请求固定 seed 跑两次（第二次能命中召回），比较输出；再对比 `yee status` 里的接受统计。
