# Phase 7B synthetic_v2 small-LLM-LoRA protocol and results

## Frozen protocol

- Strategy B starts from the original Sa2VA full PTH; no Phase 4/5/7 projection checkpoint is a training initializer.
- Full PTH SHA-256 is `5aa030f3203487281abcb57d7dbed72bba618b9f08859e1c020085454b2822e6`; base revision is `899155015275a9b7338c7f4677e19c784e0e5a21`; Sa2VA revision is `15837dcaecc304714a1f0f069e74f47e47521c7f`.
- Dataset is synthetic_v2 train 960 / val 320, manifest `1815d127d9104db1e1d91d2dddd8080c099a4f84d922896f655910adca2154be`. Test is forbidden.
- Prompt is P2 `target_only_zh` (registry/template SHA-256 `dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0` / `37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806`); seed `20260916`; BF16; batch/accumulation `1/1`; 5 epochs / 4,800 optimizer steps.
- Alignment is labels-aware strict one-to-one; legacy `fix_number=5` is disabled.
- Trainable projection is the four `text_hidden_fcs.*` tensors. LoRA targets exactly decoder layers 20–27 attention `q_proj/k_proj/v_proj/o_proj`, rank 16, alpha 32, dropout 0.05, bias none.
- Projection uses AdamW lr `4e-5`, weight decay `0.05`; LoRA uses lr `1e-4`, weight decay `0.01`; both share 240-step warmup and cosine decay.
- Validation evaluates only the five Strategy B checkpoints. Strategy A metrics are read from frozen Phase 7A artifacts; zero-shot and Strategy A inference are not rerun.
- Frozen Phase 7A summary/metrics SHA-256 are `08ef85b5a9e4d30d0896800a1e320c503632fc2df3ab5e5413621964c87b99c1` / `a847499c20af8b00fc1e9ee136ccd5c3a9cfebe36e954f25f4f4dab0fd28e638`.
- B replaces A only when Macro IoU gain is at least 0.02, empty rate is no more than A + 2 percentage points, and at most two groups decline by more than 0.03.

## Smoke and training

Smoke used `cgev2_line_category_1609c9161fd4` with supervised `[SEG]` / GT mask = `1/1`. Losses were finite; projection/LoRA gradient norms were `11.268785/1.155943`; frozen gradient count was `0`; checkpoint reload was exact.

Trainable parameters: projection `2,754,304`, LoRA `1,245,184`, total `3,999,488` / model `2,317,402,418` (`0.1726%`). The exact 68 tensor names are retained in the summary JSON.

| epoch | language loss | mask CE | Dice loss | total loss | projection grad | LoRA grad |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.020627 | 0.492507 | 0.374837 | 0.887970 | 8.575108 | 1.366672 |
| 2 | 0.000284 | 0.420678 | 0.339819 | 0.760781 | 6.902080 | 1.773286 |
| 3 | 0.000253 | 0.391523 | 0.314065 | 0.705841 | 7.380035 | 2.319130 |
| 4 | 0.000194 | 0.351059 | 0.291715 | 0.642968 | 7.087402 | 2.286100 |
| 5 | 0.000146 | 0.335890 | 0.281630 | 0.617666 | 7.082073 | 2.493900 |

Completed `4800/4800` steps; every train sample appeared `5` times. Formal training took `63.74` minutes after a `26.44` second build; peak allocated/reserved memory was `14407.8/20698.0` MiB. The formal run had no OOM and no retry.

## Validation

| checkpoint | Macro IoU | Macro Dice | Micro IoU | Micro Dice | empty | overlap | disjoint |
|---|---:|---:|---:|---:|---:|---:|---:|
| step960 | 0.213671 | 0.300999 | 0.310396 | 0.473744 | 0.046875 | 0.893750 | 0.059375 |
| step1920 | 0.229796 | 0.321776 | 0.303514 | 0.465686 | 0.018750 | 0.890625 | 0.090625 |
| step2880 | 0.238043 | 0.321669 | 0.363885 | 0.533600 | 0.031250 | 0.865625 | 0.103125 |
| step3840 | 0.272492 | 0.362596 | 0.400954 | 0.572402 | 0.012500 | 0.884375 | 0.103125 |
| step4800 | 0.273399 | 0.364193 | 0.401351 | 0.572806 | 0.012500 | 0.896875 | 0.090625 |

Best B by the preregistered order is `step4800`.

## A/B decision

Strategy A step4800 Macro IoU/Dice was `0.210365/0.292119`; best B was `0.273399/0.364193`.

- Macro IoU delta B−A: `+0.063034`。
- Macro Dice delta B−A: `+0.072073`。
- Empty-rate delta B−A: `-0.028125`。
- Selection checks: `{'empty_rate_within_0_02': True, 'groups_declining_over_0_03_at_most_2': True, 'macro_iou_gain_at_least_0_02': True}`。
- Select Strategy B: `True`。
- Paired group bootstrap: `{'dice': {'ci95': [0.03615376536890373, 0.11492287151865223], 'difference': 0.07207345700493827}, 'iou': {'ci95': [0.0319900732060343, 0.10114108577104586], 'difference': 0.06303357250272669}, 'iterations': 10000, 'seed': 20260916}`。
- Failure categories A/B: `{'strategy_a': {'boundary error': 36, 'over-segmentation': 77, 'partial target': 53, 'under-segmentation': 123, 'wrong series': 31}, 'strategy_b': {'boundary error': 44, 'over-segmentation': 81, 'partial target': 54, 'under-segmentation': 112, 'wrong series': 29}}`。

## Targeted slices

| slice | A IoU | B IoU | delta |
|---|---:|---:|---:|
| line/trend | 0.114017 | 0.126764 | +0.012747 |
| scatter/trend | 0.118900 | 0.165472 | +0.046571 |
| confidence_band/trend | 0.268679 | 0.527687 | +0.259008 |
| hard difficulty | 0.119594 | 0.155202 | +0.035608 |

The deterministic failure classifier changes wrong-series from 31 to 29 and partial-target from 53 to 54; LoRA therefore helps trend/hard overall but does not eliminate partial-target errors.

This stage ran one fixed Strategy B training and its five validation checkpoints. It did not run Strategy C, zero-shot inference, Strategy A inference, or synthetic_v2 test.
