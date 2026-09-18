# ChartGround-Edit projection model card

## Model summary

This checkpoint is a projection-only adaptation of
`ByteDance/Sa2VA-InternVL3-2B`. It contains four `text_hidden_fcs.*` tensors
and does not contain the LLM, vision encoder, InternVL `mlp1`, or SAM2 weights.

| Field | Value |
|---|---|
| Base model | Sa2VA-InternVL3-2B |
| Base InternVL identity | `OpenGVLab/InternVL3-2B@899155015275a9b7338c7f4677e19c784e0e5a21` |
| Sa2VA identity | `15837dcaecc304714a1f0f069e74f47e47521c7f` |
| Training strategy | Projection-only (`text_hidden_fcs`) |
| Trainable parameters | 2,754,304 (about 0.12% of the 2.316B-parameter training model) |
| Prompt | P2 `target_only_zh` |
| Training data | `synthetic_v1` train, 192 samples |
| Selection | step960, selected once on the 64-sample val split |
| Checkpoint size | About 11 MB |
| SHA-256 | `64c0d109d2985893ba1f2ba4c4fe7acc4265dc758e5d56ecb6ac8e2aa791f41e` |

Training selected only assistant `[SEG]` tokens whose labels were not `-100`
and enforced one supervised token per GT mask. All non-projection modules were
frozen.

## Evaluation

The checkpoint was evaluated once on the frozen 64-sample `synthetic_v1` test
split. All figures below are synthetic-benchmark results, not evidence of
stable generalization to real scientific publications.

| Metric | Zero-shot Sa2VA | Projection-tuned |
|---|---:|---:|
| 16-group Macro IoU | 0.198033 | 0.428948 |
| 16-group Macro Dice | 0.242366 | 0.534238 |
| Micro IoU | 0.225301 | 0.397803 |
| Micro Dice | 0.367748 | 0.569183 |
| Empty rate | 42.1875% | 0% |
| Overlap rate | 46.875% | 90.625% |

The frozen protocol and detailed group results are available in
[phase5b_finetuned_test_protocol.md](docs/phase5b_finetuned_test_protocol.md)
and [phase5b_finetuned_test_results.md](docs/phase5b_finetuned_test_results.md).

## Subsequent synthetic_v2 LoRA ablation

A separate experimental Strategy B artifact adds rank-16 LoRA to the last eight
LLM attention layers while retaining the four projection tensors (3,999,488
trainable parameters total). It was selected on synthetic_v2 validation before
one frozen 320-sample test. Its test Macro IoU/Dice is `0.294018/0.386910`,
versus `0.231966/0.318796` for projection-only Strategy A and
`0.081553/0.125826` for zero-shot. This does not replace or alter the v1
projection-only checkpoint documented above. The separate Strategy B artifact
has SHA-256 `c47ce4e38a9b1679c766d6360d66b6a3b69286d36b9d7cde50c70991ae475b97`.
See
[phase7c_v2_frozen_test_results.md](docs/phase7c_v2_frozen_test_results.md).

## Loading and inference

The checkpoint must be loaded together with the matching Sa2VA HF model. The
project CLI validates its SHA-256, keys, tensor shapes, training step, and base
identities before loading:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
projects/sa2va/.venv/bin/python \
  projects/chartground_edit/scripts/run_chartground_edit.py \
  --checkpoint <SA2VA_CHECKPOINT> \
  --projection-checkpoint <PROJECTION_CHECKPOINT> \
  --image <IMAGE> \
  --referring-expression "图中上升最快的折线" \
  --action highlight --strength 0.65 \
  --output-dir <OUTPUT_DIR> \
  --device cuda:0 --dtype bfloat16
```

Do not merge the four tensors into an unrelated Sa2VA revision. Unknown keys,
missing tensors, shape mismatches, and identity mismatches are rejected.

## Intended use

- Demonstrating natural-language referring segmentation on simple scientific
  charts resembling the synthetic training distribution
- Producing a predicted mask for deterministic highlight, recolor, extract,
  and remove operations
- Research and educational comparison of projection-only adaptation

The checkpoint is not intended for safety-critical document processing,
automatic scientific measurement, or unattended editing of source figures.

## Limitations

- Training and quantitative evaluation use Pillow-generated charts.
- The frozen-test `bar/appearance` group remains at 0 IoU.
- Stable performance on real paper figures has not been established.
- `remove` is deterministic replacement, not generative inpainting.
- Only the projection was adapted; the LLM, visual encoder, and SAM2 were not.
- Complex subplots, 3D charts, heatmaps, composite charts, and character-level
  OCR segmentation are outside the current scope.

## License and upstream dependencies

Repository code is provided under the repository's Apache-2.0 license. The
projection checkpoint is a derived model artifact and requires the matching
Sa2VA, InternVL3, and SAM2 components. Users are responsible for reviewing and
complying with the licenses and terms of those upstream models and any data
used with them. No upstream model weights are included in this repository.
