# Phase 4C overfit32 results

仅使用冻结的 32 条 train 样本；这些指标只表示训练集 learnability。

固定 P2、seed `20260916`，只训练四个 `text_hidden_fcs.*` tensor（2,754,304
parameters）。实际配置为 batch/accumulation=1、BF16、AdamW `lr=4e-5`、weight decay
`0.05`、16-step linear warmup 后 cosine，共 10 epoch/320 optimizer steps。这个 optimizer
配置沿用 `phase4_overfit_protocol.md` 的预注册值，因此没有采用用户指令中仅在协议无
精确值时使用的 `1e-4 / weight_decay=0 / constant` fallback。

| checkpoint | group Macro IoU | group Macro Dice | sample Macro IoU | Micro IoU | empty rate | disjoint rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.164664 | 0.189729 | 0.164664 | 0.126421 | 0.437500 | 0.218750 |
| step32 | 0.359869 | 0.447115 | 0.359869 | 0.387907 | 0.031250 | 0.031250 |
| step128 | 0.509776 | 0.605529 | 0.509776 | 0.579544 | 0.000000 | 0.031250 |
| step320 | 0.569468 | 0.674806 | 0.569468 | 0.657489 | 0.000000 | 0.000000 |

Best training checkpoint: `step320`.

## Training evidence

32 条样本均恰好出现 10 次；每步 loss/gradient 均 finite，4/4 trainable tensor 有非零
gradient，冻结参数 gradient 数为 0。无 OOM、无训练重试。

| quantity | step 1 | step 320 | minimum | first epoch mean | last epoch mean |
|---|---:|---:|---:|---:|---:|
| language loss | 0.252669 | 0.306876 | 0.243167 | 0.276101 | 0.277235 |
| mask CE | 2.234216 | 0.083364 | 0.017682 | 0.950996 | 0.294163 |
| Dice loss | 0.468146 | 0.137567 | 0.008934 | 0.359105 | 0.191986 |
| total loss | 2.955030 | 0.527808 | 0.286114 | 1.586202 | 0.763383 |

纯 320-step 循环耗时 112.999 s；模型构建 39.912 s。PyTorch peak allocated/reserved
为 7,336.494/9,714.0 MiB。

## Production inference comparison

同一次 HF 模型加载内依次恢复 baseline 并覆盖 step32/128/320 的全部四个 tensor。
训练后重跑的 baseline 与训练前 32 个预测 mask 的 SHA-256 全部相同。step320 的
Macro IoU 相对 baseline `+0.404804`，达到预注册“明显成功”门槛；empty rate
`43.75% → 0%`，overlap rate `34.375% → 100%`。16/16 个 chart/referring 组合的
mean IoU 都提升，增益范围为 `+0.003507`（bar/category）到 `+0.936499`
（bar/appearance）。这证明 strategy A 在固定训练集上有学习/记忆能力，不代表 val、
test 或真实数据泛化。

完整 checkpoint、逐步日志和预测 mask 位于 `/tmp/chartground_edit_phase4c_overfit32`
与 `/tmp/chartground_edit_phase4c_eval`；仓库只保留紧凑 metrics、summary 和固定每组首个
样本的 gallery。
