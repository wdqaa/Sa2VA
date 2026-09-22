# Phase 7A synthetic_v2 projection-only results

固定 Strategy A：仅训练 2,754,304 个 `text_hidden_fcs` 参数；评测仅使用 synthetic_v2 val，未访问 test。

| checkpoint | group Macro IoU | group Macro Dice | sample Macro IoU | sample Macro Dice | Micro IoU | Micro Dice | empty | overlap | disjoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.097575 | 0.145105 | 0.097575 | 0.145105 | 0.164430 | 0.282422 | 0.365625 | 0.503125 | 0.131250 |
| v1_step960 | 0.178815 | 0.264848 | 0.178815 | 0.264848 | 0.237077 | 0.383286 | 0.003125 | 0.856250 | 0.140625 |
| step960 | 0.174547 | 0.246992 | 0.174547 | 0.246992 | 0.260382 | 0.413180 | 0.100000 | 0.787500 | 0.112500 |
| step1920 | 0.138563 | 0.198037 | 0.138563 | 0.198037 | 0.177029 | 0.300806 | 0.100000 | 0.762500 | 0.137500 |
| step2880 | 0.165129 | 0.232370 | 0.165129 | 0.232370 | 0.220428 | 0.361231 | 0.081250 | 0.784375 | 0.134375 |
| step3840 | 0.203185 | 0.282499 | 0.203185 | 0.282499 | 0.291516 | 0.451432 | 0.056250 | 0.843750 | 0.100000 |
| step4800 | 0.210365 | 0.292119 | 0.210365 | 0.292119 | 0.296622 | 0.457530 | 0.040625 | 0.862500 | 0.096875 |

Selected checkpoint: `step4800`（16-group Macro IoU → Macro Dice → Micro IoU → earlier checkpoint）。

## Training

| epoch | language | mask CE | Dice | total |
|---:|---:|---:|---:|---:|
| 1 | 0.272143 | 0.503174 | 0.377092 | 1.152409 |
| 2 | 0.272405 | 0.431986 | 0.348294 | 1.052686 |
| 3 | 0.272200 | 0.407342 | 0.333992 | 1.013534 |
| 4 | 0.272421 | 0.383625 | 0.320156 | 0.976203 |
| 5 | 0.272183 | 0.366617 | 0.312231 | 0.951032 |

完成 `4800/4800` optimizer steps；训练耗时 `2043.7s`，模型构建 `21.9s`。峰值 allocated/reserved 显存为 `9269.4/12048.0 MiB`。OOM=`False`，retry=`0`。

## Best checkpoint: 16 chart/referring groups

| group | zero-shot IoU | v1 step960 IoU | best IoU | Δ zero-shot | Δ v1 |
|---|---:|---:|---:|---:|---:|
| bar/appearance | 0.039542 | 0.183548 | 0.305160 | +0.265617 | +0.121612 |
| bar/category | 0.044517 | 0.137587 | 0.145792 | +0.101275 | +0.008205 |
| bar/legend | 0.103525 | 0.161571 | 0.260609 | +0.157084 | +0.099038 |
| bar/trend | 0.193060 | 0.175825 | 0.187148 | -0.005912 | +0.011322 |
| confidence_band/appearance | 0.213467 | 0.312505 | 0.341239 | +0.127772 | +0.028734 |
| confidence_band/category | 0.123215 | 0.307823 | 0.373113 | +0.249897 | +0.065290 |
| confidence_band/legend | 0.104535 | 0.235524 | 0.283570 | +0.179035 | +0.048046 |
| confidence_band/trend | 0.229733 | 0.339748 | 0.268679 | +0.038946 | -0.071069 |
| line/appearance | 0.050085 | 0.113254 | 0.148977 | +0.098892 | +0.035723 |
| line/category | 0.034200 | 0.092130 | 0.168883 | +0.134682 | +0.076753 |
| line/legend | 0.076003 | 0.123495 | 0.210634 | +0.134632 | +0.087140 |
| line/trend | 0.104887 | 0.116706 | 0.114017 | +0.009130 | -0.002689 |
| scatter/appearance | 0.033451 | 0.147207 | 0.150236 | +0.116785 | +0.003030 |
| scatter/category | 0.058370 | 0.155224 | 0.134634 | +0.076264 | -0.020591 |
| scatter/legend | 0.079692 | 0.117915 | 0.154258 | +0.074566 | +0.036343 |
| scatter/trend | 0.072925 | 0.140983 | 0.118900 | +0.045975 | -0.022083 |

## Best checkpoint: diagnostic groups

| axis | group | count | mean IoU | mean Dice | empty rate |
|---|---|---:|---:|---:|---:|
| action | extract | 80 | 0.201244 | 0.278728 | 0.037500 |
| action | highlight | 80 | 0.223560 | 0.308621 | 0.037500 |
| action | recolor | 80 | 0.211182 | 0.294852 | 0.075000 |
| action | remove | 80 | 0.205476 | 0.286275 | 0.012500 |
| difficulty | easy | 112 | 0.288275 | 0.377927 | 0.026786 |
| difficulty | hard | 96 | 0.119594 | 0.187769 | 0.041667 |
| difficulty | medium | 112 | 0.210260 | 0.295755 | 0.053571 |
| degradation_type | blur | 16 | 0.187395 | 0.261573 | 0.000000 |
| degradation_type | blur+screenshot_scale | 21 | 0.228269 | 0.322328 | 0.095238 |
| degradation_type | clean | 26 | 0.200688 | 0.295464 | 0.000000 |
| degradation_type | jpeg | 8 | 0.149335 | 0.229751 | 0.000000 |
| degradation_type | jpeg+antialias_x2 | 38 | 0.195797 | 0.292185 | 0.000000 |
| degradation_type | jpeg+blur | 28 | 0.254543 | 0.315432 | 0.035714 |
| degradation_type | jpeg+blur+antialias_x2 | 44 | 0.200601 | 0.280865 | 0.022727 |
| degradation_type | jpeg+blur+screenshot_scale | 16 | 0.263634 | 0.357492 | 0.000000 |
| degradation_type | jpeg+blur+screenshot_scale+antialias_x2 | 35 | 0.238197 | 0.329200 | 0.028571 |
| degradation_type | jpeg+screenshot_scale | 28 | 0.161112 | 0.211202 | 0.142857 |
| degradation_type | jpeg+screenshot_scale+antialias_x2 | 41 | 0.195549 | 0.272510 | 0.048780 |
| degradation_type | screenshot_scale | 19 | 0.243937 | 0.335914 | 0.105263 |
| theme | dark | 160 | 0.229751 | 0.312337 | 0.043750 |
| theme | light | 160 | 0.190980 | 0.271902 | 0.037500 |

## Interpretation

- 相对 zero-shot Macro IoU：`+0.112790`。
- 相对 v1 step960 迁移：`+0.031550`。
- Empty rate：`4.0625%`；相对 zero-shot 提升组数：`15/16`。
- 相对 v1 step960 提升组数：`12/16`；相对 zero-shot 未提升组：`bar/trend`。
- 最强组：`confidence_band/category` (`0.373113`)；最弱组：`line/trend` (`0.114017`)。
- 预注册阈值检查：`{'delta_vs_v1_step960': False, 'delta_vs_zero_shot': True, 'empty_rate': True, 'improved_groups': True}`；全部通过：`False`。
- Strategy A 明显优于原始 zero-shot，且空预测受到控制，但相对强劲的 v1 step960 迁移仅提高 0.031550，未达到预注册的 +0.05 门槛。
- 因此可以把小型 LoRA 作为下一项受控 Strategy B 消融；本阶段未启动 LoRA，也未据 validation 结果重训 Strategy A。
- 本阶段仅评测 synthetic_v2 val；未访问 synthetic_v2 test。
