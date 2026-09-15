"""Dataset protocol, reader, synthetic generator, and synchronized transforms."""

from .reader import ChartGroundDataset, ChartGroundSample
from .schema import SchemaValidationError, validate_annotation, validate_jsonl

__all__ = [
    "ChartGroundDataset",
    "ChartGroundSample",
    "SchemaValidationError",
    "validate_annotation",
    "validate_jsonl",
]

