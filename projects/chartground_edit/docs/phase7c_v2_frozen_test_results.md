# Phase 7C synthetic_v2 frozen-test results

The four fixed states were evaluated once on the 320-sample synthetic_v2 test split. Strategy B remains the final strategy because it was selected on validation before this test.

| state | Macro IoU | Macro Dice | Sample IoU | Sample Dice | Micro IoU | Micro Dice | empty | overlap | disjoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero_shot | 0.081553 | 0.125826 | 0.081553 | 0.125826 | 0.122480 | 0.218231 | 0.343750 | 0.493750 | 0.162500 |
| v1_step960 | 0.192929 | 0.281413 | 0.192929 | 0.281413 | 0.239054 | 0.385865 | 0.003125 | 0.884375 | 0.112500 |
| v2_strategy_a | 0.231966 | 0.318796 | 0.231966 | 0.318796 | 0.308443 | 0.471465 | 0.028125 | 0.853125 | 0.118750 |
| v2_strategy_b | 0.294018 | 0.386910 | 0.294018 | 0.386910 | 0.398550 | 0.569948 | 0.015625 | 0.928125 | 0.056250 |

All four states completed 320/320 execution-success, mask-contract-valid and `[SEG]` records.

| state | median IoU | median Dice | mean latency ms | median latency ms |
|---|---:|---:|---:|---:|
| zero_shot | 0.000000 | 0.000000 | 594.831 | 575.080 |
| v1_step960 | 0.125658 | 0.223258 | 593.790 | 575.531 |
| v2_strategy_a | 0.148157 | 0.258029 | 594.157 | 576.268 |
| v2_strategy_b | 0.228814 | 0.372337 | 635.532 | 613.596 |

## Fixed comparisons

- `a_minus_v1` Macro IoU/Dice: `+0.039037/+0.037383`.
- `a_minus_zero_shot` Macro IoU/Dice: `+0.150413/+0.192970`.
- `b_minus_a` Macro IoU/Dice: `+0.062052/+0.068114`.
- `b_minus_zero_shot` Macro IoU/Dice: `+0.212465/+0.261084`.
- `v1_minus_zero_shot` Macro IoU/Dice: `+0.111376/+0.155587`.

## 16 chart/referring groups

| group | zero-shot | v1 | v2-A | v2-B | B−A |
|---|---:|---:|---:|---:|---:|
| bar/appearance | 0.026497 | 0.148661 | 0.247235 | 0.282766 | +0.035532 |
| bar/category | 0.026704 | 0.118265 | 0.240347 | 0.301030 | +0.060683 |
| bar/legend | 0.077010 | 0.124638 | 0.314429 | 0.319457 | +0.005028 |
| bar/trend | 0.175978 | 0.195386 | 0.224210 | 0.270799 | +0.046589 |
| confidence_band/appearance | 0.154175 | 0.296122 | 0.336774 | 0.448927 | +0.112153 |
| confidence_band/category | 0.143901 | 0.450372 | 0.359869 | 0.480035 | +0.120165 |
| confidence_band/legend | 0.060705 | 0.245421 | 0.339324 | 0.450906 | +0.111582 |
| confidence_band/trend | 0.131807 | 0.328745 | 0.313546 | 0.512460 | +0.198914 |
| line/appearance | 0.018852 | 0.082266 | 0.154566 | 0.155862 | +0.001296 |
| line/category | 0.053341 | 0.125754 | 0.179533 | 0.172501 | -0.007032 |
| line/legend | 0.070248 | 0.171682 | 0.237137 | 0.264043 | +0.026906 |
| line/trend | 0.074521 | 0.110441 | 0.102781 | 0.142316 | +0.039534 |
| scatter/appearance | 0.013870 | 0.166479 | 0.129031 | 0.179470 | +0.050439 |
| scatter/category | 0.051397 | 0.195980 | 0.204909 | 0.244938 | +0.040029 |
| scatter/legend | 0.146071 | 0.202050 | 0.204904 | 0.286683 | +0.081779 |
| scatter/trend | 0.079776 | 0.124607 | 0.122860 | 0.192097 | +0.069238 |

B versus zero-shot improves `16/16` groups and declines on `0`. B versus A improves `15/16` and declines on `1`.

## Difficulty and trend

| difficulty | zero-shot | v1 | v2-A | v2-B | B−A |
|---|---:|---:|---:|---:|---:|
| easy | 0.098254 | 0.200749 | 0.327770 | 0.428996 | +0.101226 |
| hard | 0.063940 | 0.160276 | 0.138226 | 0.163734 | +0.025508 |
| medium | 0.079950 | 0.213098 | 0.216510 | 0.270712 | +0.054202 |

Trend B−A IoU deltas:

- `confidence_band/trend`: `+0.198914`.
- `line/trend`: `+0.039534`.
- `scatter/trend`: `+0.069238`.

## Bootstrap

| paired difference | metric | estimate | 95% CI |
|---|---|---:|---:|
| b_minus_zero_shot | Macro IoU | +0.212465 | [+0.165020, +0.261090] |
| b_minus_zero_shot | Macro Dice | +0.261084 | [+0.205629, +0.315260] |
| b_minus_a | Macro IoU | +0.062052 | [+0.038416, +0.088712] |
| b_minus_a | Macro Dice | +0.068114 | [+0.040236, +0.097902] |
| a_minus_v1 | Macro IoU | +0.039037 | [+0.007900, +0.071926] |
| a_minus_v1 | Macro Dice | +0.037383 | [-0.000248, +0.075580] |

Fixed seed `20260916`, 10,000 paired group bootstrap draws.

## Strategy B failures

Error counts: `{'boundary error': 42, 'over-segmentation': 94, 'partial target': 58, 'under-segmentation': 108, 'wrong series': 18}`. IoU < 0.5: `256/320`. The largest overall category is `under-segmentation`; the largest hard-split category is `under-segmentation`. Partial-target is therefore not the largest remaining category.

Hard counts: `{'boundary error': 11, 'over-segmentation': 24, 'partial target': 11, 'under-segmentation': 43, 'wrong series': 7}`.

Worst six (deterministic sort by IoU, Dice, then sample ID):

- `cgev2_bar_appearance_9a7e3eff7828`: IoU/Dice `0.000000/0.000000`, wrong series.
- `cgev2_bar_appearance_aa73797fde12`: IoU/Dice `0.000000/0.000000`, wrong series.
- `cgev2_bar_category_2d20d46bdf7d`: IoU/Dice `0.000000/0.000000`, under-segmentation.
- `cgev2_bar_category_f801c5a14d50`: IoU/Dice `0.000000/0.000000`, wrong series.
- `cgev2_bar_trend_01e9d17f055d`: IoU/Dice `0.000000/0.000000`, under-segmentation.
- `cgev2_confidence_band_appearance_f67d1cd324ab`: IoU/Dice `0.000000/0.000000`, wrong series.

Strategy B editing: attempted/succeeded/skipped-empty = `315/315/5`.

## Integrity

- Formal backend calls: `1280`.
- Model loads: `1`.
- Protocol SHA-256 start/end: `c7e82d6c7377eb1d4ae38a8051f73acfac13278c1ea93e7ebf4a0c69c90dcc22` / `c7e82d6c7377eb1d4ae38a8051f73acfac13278c1ea93e7ebf4a0c69c90dcc22`.
- Restoration: `{'additional_inference_calls': 0, 'lora_disabled': True, 'original_projection_exact': True, 'sentinel_sample_id': 'cgev2_line_category_c150037d8a1e', 'sentinel_saved_mask_hash_exact': True}`.
- No training, retry, threshold tuning, state reselection, synthetic_v2 train/val inference, or Strategy C occurred.
