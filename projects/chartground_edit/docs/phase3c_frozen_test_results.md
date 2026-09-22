# Phase 3C frozen synthetic_v1 test baseline

运行日期：2026-09-16  
数据范围：仅 `synthetic_v1` test，64 条  
固定 Prompt：P2 `target_only_zh`；未比较或重新选择 Prompt

## 实验身份与完整性

- checkpoint revision：`15837dcaecc304714a1f0f069e74f47e47521c7f`
- manifest SHA-256：`ebad55fd98356204e572ffe6607a16a34c9dde8a916a9977a7f08bc4aed2ba82`
- test sample-ID list SHA-256：`3061387f0012b13c2fd998d81819df0e7648e8e47ae9ad86ff06490ecfe2e5c4`
- Prompt registry SHA-256：`dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0`
- protocol SHA-256（前/后）：`3e2aa95825b369746db5bd667912bbfd74d0288c5c9a5ddb9c75ce3bb3593478` / `3e2aa95825b369746db5bd667912bbfd74d0288c5c9a5ddb9c75ce3bb3593478`
- 模型加载 1 次；backend 调用 64/64；验证通过：True

## 整体指标

| metric | result |
|---|---:|
| execution success | 64/64 (100.0000%) |
| mask contract valid | 64/64 (100.0000%) |
| `[SEG]` | 64/64 (100.0000%) |
| nonempty / empty | 37 / 27 |
| overlap / nonempty-disjoint | 30 / 7 |
| 16-group Macro IoU / Dice | 0.198033 / 0.242366 |
| sample Macro IoU / Dice | 0.198033 / 0.242366 |
| Micro IoU / Dice | 0.225301 / 0.367748 |
| median IoU / Dice | 0.000000 / 0.000000 |
| mean / median latency (ms) | 609.330 / 580.492 |

## 16 个 chart/referring 组合

| group | n | mean IoU | mean Dice | empty | disjoint |
|---|---:|---:|---:|---:|---:|
| bar/appearance | 4 | 0.000000 | 0.000000 | 0.7500 | 0.2500 |
| bar/category | 4 | 0.719720 | 0.734540 | 0.0000 | 0.2500 |
| bar/legend | 4 | 0.710187 | 0.729437 | 0.2500 | 0.0000 |
| bar/trend | 4 | 0.236659 | 0.243146 | 0.2500 | 0.5000 |
| confidence_band/appearance | 4 | 0.000017 | 0.000034 | 0.7500 | 0.0000 |
| confidence_band/category | 4 | 0.152883 | 0.189736 | 0.7500 | 0.0000 |
| confidence_band/legend | 4 | 0.290622 | 0.367430 | 0.2500 | 0.2500 |
| confidence_band/trend | 4 | 0.172712 | 0.255116 | 0.5000 | 0.0000 |
| line/appearance | 4 | 0.000000 | 0.000000 | 1.0000 | 0.0000 |
| line/category | 4 | 0.016746 | 0.031389 | 0.5000 | 0.2500 |
| line/legend | 4 | 0.331132 | 0.449720 | 0.0000 | 0.0000 |
| line/trend | 4 | 0.100505 | 0.175269 | 0.0000 | 0.2500 |
| scatter/appearance | 4 | 0.000000 | 0.000000 | 1.0000 | 0.0000 |
| scatter/category | 4 | 0.150029 | 0.228433 | 0.5000 | 0.0000 |
| scatter/legend | 4 | 0.127549 | 0.221413 | 0.0000 | 0.0000 |
| scatter/trend | 4 | 0.159762 | 0.252185 | 0.2500 | 0.0000 |

## Chart type 分组

| value | n | mean IoU | mean Dice | empty | disjoint |
|---|---:|---:|---:|---:|---:|
| bar | 16 | 0.416641 | 0.426781 | 0.3125 | 0.2500 |
| confidence_band | 16 | 0.154059 | 0.203079 | 0.5625 | 0.0625 |
| line | 16 | 0.112096 | 0.164095 | 0.3750 | 0.1250 |
| scatter | 16 | 0.109335 | 0.175508 | 0.4375 | 0.0000 |

## Referring type 分组

| value | n | mean IoU | mean Dice | empty | disjoint |
|---|---:|---:|---:|---:|---:|
| appearance | 16 | 0.000004 | 0.000008 | 0.8750 | 0.0625 |
| category | 16 | 0.259844 | 0.296025 | 0.4375 | 0.1250 |
| legend | 16 | 0.364872 | 0.442000 | 0.1250 | 0.0625 |
| trend | 16 | 0.167409 | 0.231429 | 0.2500 | 0.1875 |

## Edit action 分组

| value | n | mean IoU | mean Dice | empty | disjoint |
|---|---:|---:|---:|---:|---:|
| extract | 16 | 0.136019 | 0.155777 | 0.5625 | 0.1250 |
| highlight | 16 | 0.364663 | 0.425869 | 0.1875 | 0.1250 |
| recolor | 16 | 0.129929 | 0.175275 | 0.4375 | 0.1250 |
| remove | 16 | 0.161520 | 0.212541 | 0.5000 | 0.0625 |

## Difficulty 分组

| value | n | mean IoU | mean Dice | empty | disjoint |
|---|---:|---:|---:|---:|---:|
| easy | 21 | 0.329028 | 0.380962 | 0.2381 | 0.1429 |
| hard | 22 | 0.103122 | 0.120980 | 0.6364 | 0.0909 |
| medium | 21 | 0.166467 | 0.230935 | 0.3810 | 0.0952 |

## Distractor count 分组

| value | n | mean IoU | mean Dice | empty | disjoint |
|---|---:|---:|---:|---:|---:|
| 2 | 21 | 0.329028 | 0.380962 | 0.2381 | 0.1429 |
| 3 | 21 | 0.166467 | 0.230935 | 0.3810 | 0.0952 |
| 4 | 22 | 0.103122 | 0.120980 | 0.6364 | 0.0909 |

## Bootstrap 95% CI

- 16-group Macro IoU：0.198033，95% CI [0.101680, 0.316589]。
- 16-group Macro Dice：0.242366，95% CI [0.139124, 0.360458]。
- 该区间只描述 test 估计不确定性，未用于修改 Prompt。

## P2 validation / test 对比

| metric | val | test | test - val |
|---|---:|---:|---:|
| Macro IoU | 0.182199 | 0.198033 | +0.015834 |
| Macro Dice | 0.216221 | 0.242366 | +0.026145 |
| empty rate | 0.437500 | 0.421875 | -0.015625 |
| disjoint rate | 0.156250 | 0.109375 | -0.046875 |

Validation worst group：`bar/trend` (0.000000)；test worst group：`bar/appearance` (0.000000)。

### 逐组合 validation / test 差异

| group | val IoU | test IoU | Δ IoU | val Dice | test Dice | Δ Dice |
|---|---:|---:|---:|---:|---:|---:|
| bar/appearance | 0.000317 | 0.000000 | -0.000317 | 0.000634 | 0.000000 | -0.000634 |
| bar/category | 0.946780 | 0.719720 | -0.227059 | 0.972493 | 0.734540 | -0.237953 |
| bar/legend | 0.718873 | 0.710187 | -0.008686 | 0.734067 | 0.729437 | -0.004630 |
| bar/trend | 0.000000 | 0.236659 | +0.236659 | 0.000000 | 0.243146 | +0.243146 |
| confidence_band/appearance | 0.000000 | 0.000017 | +0.000017 | 0.000000 | 0.000034 | +0.000034 |
| confidence_band/category | 0.087384 | 0.152883 | +0.065499 | 0.129503 | 0.189736 | +0.060234 |
| confidence_band/legend | 0.272006 | 0.290622 | +0.018616 | 0.347715 | 0.367430 | +0.019714 |
| confidence_band/trend | 0.162853 | 0.172712 | +0.009859 | 0.197229 | 0.255116 | +0.057888 |
| line/appearance | 0.000000 | 0.000000 | +0.000000 | 0.000000 | 0.000000 | +0.000000 |
| line/category | 0.000000 | 0.016746 | +0.016746 | 0.000000 | 0.031389 | +0.031389 |
| line/legend | 0.252687 | 0.331132 | +0.078445 | 0.326583 | 0.449720 | +0.123138 |
| line/trend | 0.183081 | 0.100505 | -0.082575 | 0.286182 | 0.175269 | -0.110913 |
| scatter/appearance | 0.000000 | 0.000000 | +0.000000 | 0.000000 | 0.000000 | +0.000000 |
| scatter/category | 0.130610 | 0.150029 | +0.019419 | 0.201131 | 0.228433 | +0.027302 |
| scatter/legend | 0.094909 | 0.127549 | +0.032641 | 0.155746 | 0.221413 | +0.065667 |
| scatter/trend | 0.065678 | 0.159762 | +0.094084 | 0.108251 | 0.252185 | +0.143934 |

这些变化只用于描述分布与泛化落差，不触发 Prompt 修改或 validation 重跑。

## 编辑闭环

- 非空预测编辑尝试：37；成功：37。
- 空预测/非法预测跳过编辑：27。
- 每个编辑输入均记录为 `predicted_mask` 并以保存 mask 的 SHA-256 独立核对；编辑成功不代表分割正确。

## 失败模式与产物

最弱组为 `bar/appearance`，mean IoU=0.000000。worst 10 样本如下；gallery 固定按每组字典序最小 sample 展示，未按效果挑选。

主要系统性失败仍是 appearance 指代：16 条中 14 条为空，四种 chart 的 appearance
mean IoU 几乎为 0；`line/appearance` 和 `scatter/appearance` 均为 4/4 空预测。
7 条 nonempty-disjoint 中，`bar/trend` 占 2 条，bar 全部 16 条合计占 4 条。难度从
easy 到 medium/hard 时 mean IoU 为 0.329028/0.166467/0.103122，同时 empty rate 从
23.81% 上升到 38.10%/63.64%；由于 difficulty 与 distractor count 一一对应，不能把
下降单独归因于任一因素。较强组仍集中在大目标：`bar/category`、`bar/legend` 和
`line/legend`。

整体 test Macro IoU/Dice 比 val 高 0.015834/0.026145，empty 与 disjoint rate 分别低
1.5625/4.6875 个百分点；这不是所有组一致改善。`bar/category` IoU 下降 0.227059，
`line/trend` 下降 0.082575，而 `bar/trend`、`scatter/trend` 和 `line/legend` 分别上升
0.236659、0.094084 和 0.078445。这里只描述 split 间分布变化，不据此调整 P2。

| sample | group | IoU | Dice | execution | mask valid | empty | disjoint |
|---|---|---:|---:|---:|---:|---:|---:|
| cgev1_bar_appearance_0f377fba33 | bar/appearance | 0.000000 | 0.000000 | True | True | True | False |
| cgev1_bar_appearance_26770ffc87 | bar/appearance | 0.000000 | 0.000000 | True | True | True | False |
| cgev1_bar_appearance_3efba99d21 | bar/appearance | 0.000000 | 0.000000 | True | True | True | False |
| cgev1_bar_appearance_62897d2d74 | bar/appearance | 0.000000 | 0.000000 | True | True | False | True |
| cgev1_bar_category_04e33d518e | bar/category | 0.000000 | 0.000000 | True | True | False | True |
| cgev1_bar_legend_1c77354376 | bar/legend | 0.000000 | 0.000000 | True | True | True | False |
| cgev1_bar_trend_0f419263bb | bar/trend | 0.000000 | 0.000000 | True | True | True | False |
| cgev1_bar_trend_501efeec22 | bar/trend | 0.000000 | 0.000000 | True | True | False | True |
| cgev1_bar_trend_ef677a2abf | bar/trend | 0.000000 | 0.000000 | True | True | False | True |
| cgev1_confidence_band_appearance_b8cfa20f03 | confidence_band/appearance | 0.000000 | 0.000000 | True | True | True | False |

- 完整临时输出：`<WORK_DIR>`
- 标量记录：`results/phase3c_frozen_test_metrics.jsonl`
- 聚合结果：`results/phase3c_frozen_test_summary.json`
- gallery：`assets/phase3c_frozen_test_gallery.png`

gallery 已按原始分辨率实际打开检查：16 组均出现，四列依次为 original、GT overlay、
P2 prediction overlay 和 edited result；空预测显示 `EMPTY / EDIT SKIPPED`，IoU=0 与
错误图元案例未隐藏。

测试、compileall、Git 检查和退出后 GPU 状态在最终验收与 `EXPERIMENTS.md` 中记录。
