"""Dataset protocol, reader, synthetic generator, and synchronized transforms."""

from .reader import ChartGroundDataset, ChartGroundSample
from .reader_v1 import ChartGroundV1Dataset, ChartGroundV1Sample
from .schema import SchemaValidationError, validate_annotation, validate_jsonl
from .schema_v1 import validate_annotation_v1, validate_jsonl_v1

__all__ = [
    "ChartGroundDataset",
    "ChartGroundSample",
    "ChartGroundV1Dataset",
    "ChartGroundV1Sample",
    "SchemaValidationError",
    "validate_annotation",
    "validate_jsonl",
    "validate_annotation_v1",
    "validate_jsonl_v1",
]
