"""Dataset protocol, reader, synthetic generator, and synchronized transforms."""

from .reader import ChartGroundDataset, ChartGroundSample
from .reader_v1 import ChartGroundV1Dataset, ChartGroundV1Sample
from .reader_v2 import ChartGroundV2Dataset, ChartGroundV2Sample
from .schema import SchemaValidationError, validate_annotation, validate_jsonl
from .schema_v1 import validate_annotation_v1, validate_jsonl_v1
from .schema_v2 import validate_annotation_v2, validate_jsonl_v2

__all__ = [
    "ChartGroundDataset",
    "ChartGroundSample",
    "ChartGroundV1Dataset",
    "ChartGroundV1Sample",
    "ChartGroundV2Dataset",
    "ChartGroundV2Sample",
    "SchemaValidationError",
    "validate_annotation",
    "validate_jsonl",
    "validate_annotation_v1",
    "validate_jsonl_v1",
    "validate_annotation_v2",
    "validate_jsonl_v2",
]
