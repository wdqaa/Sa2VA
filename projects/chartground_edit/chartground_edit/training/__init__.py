"""Train-only, model-free contracts for ChartGround-Edit Phase 4."""

from .data_adapter import (
    ASSISTANT_TARGET,
    EXPECTED_MANIFEST_SHA256,
    Phase4DataSample,
    Phase4TrainDataset,
    validate_training_output_dir,
)
from .subsets import (
    audit_phase4_subsets,
    build_overfit32_manifest,
    build_smoke1_manifest,
    prepare_phase4_subsets,
)
from .alignment import (
    EXPECTED_MASKS_PER_SAMPLE,
    IGNORE_INDEX,
    OBJECT_COUNT_POLICY,
    SEG_TOKEN_SELECTION,
    select_supervised_seg_tokens,
    validate_strict_one_to_one,
)

__all__ = [
    "ASSISTANT_TARGET",
    "EXPECTED_MANIFEST_SHA256",
    "EXPECTED_MASKS_PER_SAMPLE",
    "IGNORE_INDEX",
    "OBJECT_COUNT_POLICY",
    "Phase4DataSample",
    "Phase4TrainDataset",
    "SEG_TOKEN_SELECTION",
    "audit_phase4_subsets",
    "build_overfit32_manifest",
    "build_smoke1_manifest",
    "prepare_phase4_subsets",
    "select_supervised_seg_tokens",
    "validate_strict_one_to_one",
    "validate_training_output_dir",
]
