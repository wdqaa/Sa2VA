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

__all__ = [
    "ASSISTANT_TARGET",
    "EXPECTED_MANIFEST_SHA256",
    "Phase4DataSample",
    "Phase4TrainDataset",
    "audit_phase4_subsets",
    "build_overfit32_manifest",
    "build_smoke1_manifest",
    "prepare_phase4_subsets",
    "validate_training_output_dir",
]
