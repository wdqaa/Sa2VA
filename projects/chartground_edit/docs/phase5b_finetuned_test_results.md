# Phase 5B fine-tuned test results

唯一一次正式推理覆盖 `synthetic_v1` test 64 条；使用冻结的 P2 和 Phase 5A `step960` projection。Zero-shot 数值直接读取 Phase 3C 保存结果，未重新运行 baseline。

| metric | fine-tuned | zero-shot | delta |
|---|---:|---:|---:|
| 16-group Macro IoU | 0.428948 | 0.198033 | +0.230916 |
| 16-group Macro Dice | 0.534238 | 0.242366 | +0.291872 |
| Micro IoU | 0.397803 | 0.225301 | +0.172502 |
| Micro Dice | 0.569183 | 0.367748 | +0.201435 |
| Empty rate | 0.000000 | 0.421875 | -0.421875 |
| Overlap rate | 0.906250 | 0.468750 | +0.437500 |
| Nonempty-disjoint rate | 0.093750 | 0.109375 | -0.015625 |

Sample Macro IoU/Dice: `0.428948` / `0.534238`; median IoU/Dice: `0.378353` / `0.548924`.

Bootstrap 16-group Macro IoU 95% CI: `[0.320991, 0.555574]`; Dice 95% CI: `[0.425557, 0.647002]`.

## 16 chart/referring groups

| group | zero-shot IoU | fine-tuned IoU | delta | fine-tuned Dice |
|---|---:|---:|---:|---:|
| bar/appearance | 0.000000 | 0.000000 | +0.000000 | 0.000000 |
| bar/category | 0.719720 | 0.961904 | +0.242184 | 0.980572 |
| bar/legend | 0.710187 | 0.967094 | +0.256908 | 0.983240 |
| bar/trend | 0.236659 | 0.482488 | +0.245829 | 0.491086 |
| confidence_band/appearance | 0.000017 | 0.283871 | +0.283854 | 0.434302 |
| confidence_band/category | 0.152883 | 0.342590 | +0.189707 | 0.465156 |
| confidence_band/legend | 0.290622 | 0.544823 | +0.254200 | 0.692796 |
| confidence_band/trend | 0.172712 | 0.317812 | +0.145100 | 0.450533 |
| line/appearance | 0.000000 | 0.396911 | +0.396911 | 0.526439 |
| line/category | 0.016746 | 0.370229 | +0.353483 | 0.500160 |
| line/legend | 0.331132 | 0.509690 | +0.178559 | 0.641756 |
| line/trend | 0.100505 | 0.339807 | +0.239302 | 0.478769 |
| scatter/appearance | 0.000000 | 0.200363 | +0.200363 | 0.328253 |
| scatter/category | 0.150029 | 0.451097 | +0.301068 | 0.595985 |
| scatter/legend | 0.127549 | 0.370451 | +0.242901 | 0.500809 |
| scatter/trend | 0.159762 | 0.324042 | +0.164280 | 0.477946 |

提升/下降/持平组数：15 / 0 / 1。
最大提升：`line/appearance` (+0.396911)；最大下降：`bar/appearance` (+0.000000)。
最强组：`bar/legend` (0.967094)；最弱组：`bar/appearance` (0.000000)。

## Execution and editing

- Execution success: 64/64; mask contract: 64/64; `[SEG]`: 64/64.
- Nonempty/empty: 64 / 0; overlap/nonempty-disjoint: 58 / 6.
- Predicted-mask edits attempted/succeeded: 64 / 64; empty predictions skipped: 0.
- Mean/median latency: 604.748 / 572.532 ms.
- 编辑只使用预测 mask；编辑成功不代表分割正确。

完整标量与分组结果见对应 JSON/JSONL；gallery 每组按 sample ID 确定性选择一条，未按效果筛选。
