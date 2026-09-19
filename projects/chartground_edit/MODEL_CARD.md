# ChartGround-Edit Strategy B model card

## Model summary

ChartGround-Edit Strategy B is a parameter-efficient adapter for
`ByteDance/Sa2VA-InternVL3-2B`. It is not a standalone model: the checkpoint
contains four `text_hidden_fcs.*` tensors plus 64 LLM LoRA tensors and must be
loaded with the exact Sa2VA base revision below. It contains no vision encoder,
InternVL `mlp1`, SAM2, embedding, LM-head, or full LLM weights.

| Field | Value |
|---|---|
| Sa2VA base | `ByteDance/Sa2VA-InternVL3-2B@15837dcaecc304714a1f0f069e74f47e47521c7f` |
| InternVL base identity | `OpenGVLab/InternVL3-2B@899155015275a9b7338c7f4677e19c784e0e5a21` |
| Training full PTH SHA-256 | `5aa030f3203487281abcb57d7dbed72bba618b9f08859e1c020085454b2822e6` |
| synthetic_v2 manifest SHA-256 | `1815d127d9104db1e1d91d2dddd8080c099a4f84d922896f655910adca2154be` |
| Prompt | P2 `target_only_zh` |
| Projection tensors | 4 `text_hidden_fcs.*` tensors |
| LoRA targets | decoder layers 20–27, attention `q_proj/k_proj/v_proj/o_proj` |
| LoRA configuration | rank 16, alpha 32, dropout 0.05, bias none |
| LoRA tensors | 64; no `modules_to_save` |
| Trainable parameters | 3,999,488 / 2,317,402,418 (about 0.1726%) |
| Checkpoint step | 4800 |
| Checkpoint size | 16,032,698 bytes (about 15.3 MiB) |
| Checkpoint SHA-256 | `c47ce4e38a9b1679c766d6360d66b6a3b69286d36b9d7cde50c70991ae475b97` |

Training selected only assistant `[SEG]` tokens whose labels were not `-100`
and enforced one supervised token per GT mask. The vision encoder, InternVL
`mlp1`, SAM2, embeddings, LM head, LLM MLPs, and decoder layers 0–19 remained
frozen.

## Training, selection, and evaluation

Strategy B was initialized from the original Sa2VA full PTH, not from a prior
projection checkpoint. It trained on the 960-sample synthetic_v2 train split
for five epochs. Strategy A and B were compared on the fixed 320-sample val
split; B step4800 won the preregistered rule and was frozen before test.

The 320-sample synthetic_v2 test split was then evaluated exactly once in the
fixed Zero-shot → v1 projection → v2 projection → v2 projection + LoRA order.
The test result did not trigger checkpoint reselection, tuning, or retraining.
All results below are synthetic-benchmark measurements, not evidence of stable
generalization to real scientific publications.

| Model | Trainable | Macro IoU | Macro Dice | Micro IoU | Empty |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 0 | 0.081553 | 0.125826 | 0.122480 | 34.3750% |
| v1 projection | 2,754,304 | 0.192929 | 0.281413 | 0.239054 | 0.3125% |
| v2 projection | 2,754,304 | 0.231966 | 0.318796 | 0.308443 | 2.8125% |
| v2 projection + LoRA | 3,999,488 | **0.294018** | **0.386910** | **0.398550** | **1.5625%** |

B improves Macro IoU by `+0.212465` over zero-shot and `+0.062052` over
Strategy A. The paired group bootstrap 95% CI for B−A Macro IoU is
`[0.038416, 0.088712]`. Full identities, groups, bootstrap results, and the
single-run integrity record are available in the
[frozen protocol](docs/phase7c_v2_frozen_test_protocol.md) and
[Phase 7C results](docs/phase7c_v2_frozen_test_results.md).

The older synthetic_v1 projection-only artifact remains a distinct checkpoint;
Strategy B does not overwrite or continue training from it.

## Loading and inference

The project CLI checks the released adapter SHA-256, all 68 keys, tensor shapes,
training step, Prompt, LoRA configuration, and base/full-PTH/data identities.
Unknown, missing, or mismatched weights are rejected.

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
projects/sa2va/.venv/bin/python \
  projects/chartground_edit/scripts/run_chartground_edit.py \
  --checkpoint <SA2VA_CHECKPOINT> \
  --adapter-checkpoint <CHARTGROUND_ADAPTER> \
  --image <INPUT_IMAGE> \
  --referring-expression "图中上升最快的折线" \
  --action highlight --strength 0.65 \
  --output-dir <OUTPUT_DIR> \
  --device cuda:0 --dtype bfloat16
```

The CLI accepts `highlight`, `recolor`, `extract`, and `remove` and writes the
predicted mask, overlay, edited image when applicable, and a JSON record. It
does not accept a ground-truth mask.

## Intended use

- Research and educational study of referring segmentation on synthetic
  scientific charts
- Producing predicted masks for deterministic highlight, recolor, extract, and
  remove operations
- Controlled comparison of projection-only and small-LLM-LoRA adaptation

The adapter is not intended for safety-critical document processing, automatic
scientific measurement, or unattended editing of source figures.

## Limitations

- All quantitative training and evaluation used Pillow-generated charts.
- The final synthetic_v2 Macro IoU is `0.294018`, not production-level accuracy;
  256/320 test samples have IoU below 0.5.
- The largest failure category is `under-segmentation`, and `line/trend` remains
  the weakest chart/referring group.
- Quantitative OOD evaluation on real paper figures has not been completed.
- `remove` performs deterministic filling, not generative content restoration.
- The vision encoder and SAM2 are not adapted. Complex subplots, 3D charts,
  heatmaps, composite charts, and character-level OCR segmentation remain out
  of scope.

## License and upstream dependencies

Repository code is provided under the repository's Apache-2.0 license. This is
a derived adapter and requires the matching Sa2VA, InternVL3, and SAM2
components. No upstream model weights are included. Users must review and
comply with the licenses and terms of those upstream models; see the
[Sa2VA upstream README](../sa2va/README.md) for project references and citation
information.
