# Phase 7C synthetic_v2 frozen-test protocol

Status: preregistered and frozen before any synthetic_v2 test model call

Protocol version: `phase7c-v2-frozen-test-v1`

Preregistration date: 2026-09-19

Bootstrap seed / iterations: `20260916` / `10000`

## Experiment boundary

This is the only frozen evaluation of the fixed synthetic_v2 test split. No
training, optimizer step, Prompt or data change, threshold tuning, checkpoint
selection, failed-sample retry, or Strategy C is allowed. Test results cannot
change the fact that Strategy B step4800 was selected on validation before this
protocol was written.

The runner may read only the 320 `split=test` records. Each of the four fixed
states must process every record exactly once in manifest order, giving exactly
`4 * 320 = 1280` formal `predict_forward` calls and 1280 metric rows. A sample
exception becomes a zero-mask failure record and is not retried. If a global
failure occurs after the first formal call, the run stops and must not restart
without an explicit new human decision.

## Frozen identities

- Model: `ByteDance/Sa2VA-InternVL3-2B`
- Sa2VA revision: `15837dcaecc304714a1f0f069e74f47e47521c7f`
- Base: `OpenGVLab/InternVL3-2B`
- Base revision: `899155015275a9b7338c7f4677e19c784e0e5a21`
- Full PTH SHA-256: `5aa030f3203487281abcb57d7dbed72bba618b9f08859e1c020085454b2822e6`
- synthetic_v2 manifest: `projects/chartground_edit/data/synthetic_v2/annotations.jsonl`
- Manifest SHA-256: `1815d127d9104db1e1d91d2dddd8080c099a4f84d922896f655910adca2154be`
- Test samples: `320`; 16 chart/referring groups with 20 samples each
- Test sample-ID list SHA-256: `34ca02e0e79e1b208600ad844f42b4e24fb6c4ee96ed3bd910f06fc127a9ad2b`

The sample-ID hash input is the 320 test `sample_id` values in manifest order,
each followed by LF, including the final ID.

Frozen checkpoints:

| state | runtime checkpoint | SHA-256 |
|---|---|---|
| v1 step960 | `/home/dqwang/Model/ChartGround-Edit/chartground_projection_step960.pth` | `64c0d109d2985893ba1f2ba4c4fe7acc4265dc758e5d56ecb6ac8e2aa791f41e` |
| v2 Strategy A step4800 | `/home/dqwang/Model/ChartGround-Edit/chartground_v2_projection_step4800.pth` | `6c8f30d20e52b1e0473b8a9bdd69e682c944cbdcf6bed8c0c45f30a399eab3c4` |
| v2 Strategy B step4800 | `/home/dqwang/Model/ChartGround-Edit/chartground_v2_lora_step4800.pth` | `c47ce4e38a9b1679c766d6360d66b6a3b69286d36b9d7cde50c70991ae475b97` |

Strategy B contains exactly four `text_hidden_fcs.*` tensors and 64 LoRA
tensors. Its LoRA is fixed to decoder layers 20–27 attention
`q_proj/k_proj/v_proj/o_proj`, rank 16, alpha 32, dropout 0.05, bias none and no
`modules_to_save`.

## Prompt and inference contract

- Prompt: P2 `target_only_zh`
- Prompt registry SHA-256: `dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0`
- P2 template SHA-256: `37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806`
- BF16, one GPU, one HF model load, deterministic generation
- No GT field, action, edit parameter or target metadata enters model input
- The first returned prediction mask is used; GT never selects or repairs it
- Existing backend post-processing is frozen: remove only leading singleton
  dimensions, accept bool/0–1/0–255 masks, reject invalid values or dimensions,
  and use nearest-neighbor only when restoring the original image size
- Legal empty masks are saved and score IoU/Dice zero; invalid output or a
  sample exception is represented by an original-size empty failure mask

Formal state order:

1. `zero_shot`: restore the original four projection tensors; LoRA disabled.
2. `v1_step960`: load only the frozen v1 projection; LoRA disabled.
3. `v2_strategy_a`: load only the frozen v2-A projection; LoRA disabled.
4. `v2_strategy_b`: load all four B projection tensors and all 64 LoRA tensors;
   enable only the fixed B adapter.

Before each state, all four projection tensors are overwritten and the LoRA
enabled/disabled state is asserted. After B, the runner restores the byte-exact
original projection snapshot and disables LoRA. To preserve both the exact
1280-call budget and the no-repeat rule, it performs no additional sentinel
inference: the fixed sentinel is the first manifest test ID,
`cgev2_line_category_c150037d8a1e`; its zero-shot mask hash is recorded during
its single formal call and its saved artifact hash is rechecked after restore,
while restoration itself is verified by exact projection tensor hashes plus the
disabled-adapter assertion.

## Metrics and comparisons

For every state, report execution success, mask-contract validity, `[SEG]`,
nonempty/empty, overlap/nonempty-disjoint, 16-group Macro IoU/Dice, sample Macro
IoU/Dice, Micro IoU/Dice, median IoU/Dice and mean/median latency. Report groups
by chart/referring, chart type, referring type, action, difficulty, theme and
degradation type.

The metric implementation remains the existing boolean intersection/union and
Dice implementation used by Phase 7A/7B. Group Macro averages the 20-sample
mean of each of the 16 chart/referring groups with equal group weight. No
threshold is introduced.

Required deltas are B−zero-shot, A−zero-shot, B−A, v1−zero-shot and
A−v1. Paired bootstrap resamples the 16 matched chart/referring groups with
replacement, 16 groups per draw, for 10,000 draws using NumPy
`default_rng(20260916)`. Report 2.5/97.5 percentiles for IoU and Dice for
B−zero-shot, B−A and A−v1. These intervals are descriptive only.

## Editing and failure analysis

Only Strategy B enters the editing backend. A nonempty predicted mask drives
the record's original action and edit parameters. Empty predictions are marked
`edit_skipped_empty`; GT is never substituted.

The frozen deterministic error taxonomy is
`chartground_edit.visualization.phase5b_saved.classify_failure` (source SHA-256
`a3b6e54ede3fb7997987462e171cbe6b587963cb41c5c6d294a4cfb0ee775681`):

- no true positives with nonempty prediction: wrong series;
- recall below 0.6 with precision at least 0.55: partial target;
- FP greater than `max(1.35 * FN, 8)`: over-segmentation;
- FN greater than `max(1.35 * FP, 8)`: under-segmentation;
- otherwise: boundary error.

Report all five counts for B, the number with IoU below 0.5, the deterministic
worst six ordered by `(IoU, Dice, sample_id)`, hard-difficulty failures and the
test deltas for line/scatter/confidence-band trend. None may alter model choice.
