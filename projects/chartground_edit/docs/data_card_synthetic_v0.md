# Synthetic v0 data card

## Scope

`synthetic_v0` is a 32-sample local smoke-test dataset for ChartGround-Edit Phase 1A. It exists to validate the JSONL protocol, binary-mask semantics, synchronized geometry and visual inspection loop. It is not a benchmark and does not support claims about model quality.

## Source and license

All images, masks, numeric series and annotations are generated locally by `chartground_edit.datasets.synthetic` using NumPy and Pillow. No external dataset, model or downloaded asset is used. Repository licensing applies to the generator and its generated fixtures.

## Composition

- 4 chart types: `line`, `bar`, `scatter`, `confidence_band`
- 4 referring types: `category`, `appearance`, `legend`, `trend`
- 2 samples for every chart/referring combination
- 4 edit actions distributed evenly
- deterministic default seed: `20260915`

The manifest-relative `images/`, `masks/` and `visualizations/` directories, `annotations.jsonl`, `summary.json`, and `gallery.png` are generated artifacts. The root `.gitignore` excludes data directories, so regenerate them locally with the documented command.

## Split rule

Samples whose generation index modulo 8 is 0 use `val`, modulo 8 is 1 use `test`, and all others use `train`, producing 24 train, 4 validation and 4 test samples. This tiny dataset is only a protocol fixture; it does not claim leakage-safe benchmark splits.

## Mask policy

See `annotation_spec_v0.md`. Masks are same-size, single-channel lossless PNGs with values only in `{0, 255}`. Legend glyphs are never included. Curve masks include line and markers but exclude bands. Confidence bands are independent filled-region targets.

## Known limitations

- Pillow-rendered charts are deliberately simple and do not represent real publication diversity.
- Text, legends, occlusion, anti-aliasing and hard negatives are limited.
- Instructions are template-generated and are not a linguistic-quality benchmark.
- No model inference or training result is associated with this data version.

