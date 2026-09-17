# Phase 5A full-train validation results

仅使用冻结的 64 条 val 样本进行 checkpoint 选择；未访问 test。

| checkpoint | group Macro IoU | group Macro Dice | sample Macro IoU | Micro IoU | empty rate | disjoint rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.182199 | 0.216221 | 0.182199 | 0.192302 | 0.437500 | 0.156250 |
| step192 | 0.265243 | 0.342522 | 0.265243 | 0.252220 | 0.078125 | 0.093750 |
| step576 | 0.385975 | 0.485005 | 0.385975 | 0.396003 | 0.000000 | 0.078125 |
| step960 | 0.426042 | 0.532503 | 0.426042 | 0.438493 | 0.000000 | 0.078125 |
| step1344 | 0.417203 | 0.518894 | 0.417203 | 0.421713 | 0.000000 | 0.078125 |
| step1920 | 0.406564 | 0.509340 | 0.406564 | 0.418913 | 0.000000 | 0.078125 |

Selected checkpoint: `step960`.

## Training

固定 synthetic_v1 train 192 条、P2、seed `20260916`，只训练四个
`text_hidden_fcs.*` tensor（2,754,304 parameters）。配置为 BF16、batch/accumulation
`1/1`、AdamW `lr=4e-5`、weight decay `0.05`、96-step linear warmup 后 cosine，
10 epoch/1920 optimizer steps；每条 train 样本恰好出现 10 次。

| epoch | language | mask CE | Dice | total |
|---:|---:|---:|---:|---:|
| 1 | 0.274937 | 0.546797 | 0.318577 | 1.140312 |
| 2 | 0.275674 | 0.411659 | 0.269106 | 0.956438 |
| 3 | 0.274719 | 0.466016 | 0.258645 | 0.999379 |
| 4 | 0.276003 | 0.417235 | 0.239114 | 0.932352 |
| 5 | 0.275242 | 0.432463 | 0.233874 | 0.941579 |
| 6 | 0.275129 | 0.377528 | 0.220345 | 0.873002 |
| 7 | 0.276737 | 0.381412 | 0.211912 | 0.870060 |
| 8 | 0.273885 | 0.336425 | 0.206668 | 0.816977 |
| 9 | 0.275122 | 0.346658 | 0.207061 | 0.828841 |
| 10 | 0.275376 | 0.324230 | 0.201959 | 0.801565 |

训练循环 679.365 秒，模型构建 22.009 秒；peak allocated/reserved 为
7,336.783/9,178.0 MiB。所有 loss/gradient finite，4/4 trainable tensors 有非零
gradient，冻结参数 gradient 为 0；无 OOM、无重试。

## Validation selection

baseline 在 `1e-9` 容差内复现 Phase 3B P2 的 Macro IoU/Dice、empty rate 和
nonempty count，随后才顺序评测五份 checkpoint。step960 相对 baseline Macro IoU
提升 `+0.243843`，达到“明显有效”标准。选择只使用 16-group Macro IoU；无需启用
tie-break。16 个 chart/referring 组合中 14 个提升、2 个下降：`bar/appearance`
`-0.000317`，`bar/category` `-0.010559`。selected checkpoint 最强组为
`bar/category`（IoU `0.936221`），最弱为 `bar/appearance`（IoU `0`）。这些结果只用于
val checkpoint 选择；本阶段未访问 test。
