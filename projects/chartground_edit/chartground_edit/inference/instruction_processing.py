"""Deterministic decomposition of synthetic-v0 editing instructions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class InstructionProcessingError(ValueError):
    """Raised when an instruction does not match a registered synthetic-v0 template."""


@dataclass(frozen=True)
class InstructionParts:
    """Information that is explicitly recoverable from one instruction string."""

    original_instruction: str
    referring_expression: str
    edit_action: str
    edit_parameters: dict[str, str]
    extraction_rule: str


ACTION_SUFFIXES = {
    "highlight": "高亮",
    "recolor": "改成红色",
    "extract": "提取到透明背景",
    "remove": "用背景色移除",
}

ACTION_PARAMETERS = {
    "highlight": {},
    "recolor": {"color": "red"},
    "extract": {"background": "transparent"},
    "remove": {"fill": "background_color"},
}

CHART_NOUNS = {
    "line": "曲线",
    "bar": "柱子",
    "scatter": "散点序列",
    "confidence_band": "置信区间",
}

REFERENCE_FRAMES = {
    "category": ("类别 ", " 对应的{noun}"),
    "appearance": ("颜色为 ", " 的{noun}"),
    "legend": ("图例中 ", " 对应的{noun}"),
    "trend": ("趋势为 ", " 的{noun}"),
}


def extract_synthetic_v0_instruction(annotation: dict[str, Any]) -> InstructionParts:
    """Extract the contiguous target phrase using only registered template fields.

    ``target_attributes`` is intentionally never read. ``chart_type``,
    ``referring_type`` and ``edit_action`` only select template literals that
    must themselves be present in the original instruction.
    """
    required = ("instruction", "chart_type", "referring_type", "edit_action")
    missing = [field for field in required if field not in annotation]
    if missing:
        raise InstructionProcessingError(
            f"annotation is missing required field(s): {', '.join(missing)}"
        )
    instruction = annotation["instruction"]
    chart_type = annotation["chart_type"]
    referring_type = annotation["referring_type"]
    edit_action = annotation["edit_action"]
    if type(instruction) is not str or not instruction:
        raise InstructionProcessingError("instruction must be a non-empty string")
    if chart_type not in CHART_NOUNS:
        raise InstructionProcessingError(f"unsupported chart_type: {chart_type!r}")
    if referring_type not in REFERENCE_FRAMES:
        raise InstructionProcessingError(
            f"unsupported referring_type: {referring_type!r}"
        )
    if edit_action not in ACTION_SUFFIXES:
        raise InstructionProcessingError(f"unsupported edit_action: {edit_action!r}")

    prefix = "请将"
    action_suffix = ACTION_SUFFIXES[edit_action]
    if not instruction.startswith(prefix) or not instruction.endswith(action_suffix):
        raise InstructionProcessingError(
            "instruction does not match the synthetic-v0 action wrapper for "
            f"edit_action={edit_action!r}"
        )
    referring_expression = instruction[
        len(prefix) : len(instruction) - len(action_suffix)
    ]
    if not referring_expression:
        raise InstructionProcessingError("referring expression is empty")
    _validate_reference_frame(
        referring_expression,
        chart_type=chart_type,
        referring_type=referring_type,
    )
    if referring_expression not in instruction:
        raise InstructionProcessingError(
            "extracted referring expression is not a contiguous instruction substring"
        )
    return InstructionParts(
        original_instruction=instruction,
        referring_expression=referring_expression,
        edit_action=edit_action,
        edit_parameters=dict(ACTION_PARAMETERS[edit_action]),
        extraction_rule=(
            "synthetic_v0: strip exact '请将' prefix and registered action suffix; "
            f"validate {referring_type}/{chart_type} reference frame"
        ),
    )


def _validate_reference_frame(
    expression: str, *, chart_type: str, referring_type: str
) -> None:
    noun = CHART_NOUNS[chart_type]
    prefix, suffix_template = REFERENCE_FRAMES[referring_type]
    suffix = suffix_template.format(noun=noun)
    if not expression.startswith(prefix) or not expression.endswith(suffix):
        raise InstructionProcessingError(
            "referring expression does not match the registered synthetic-v0 "
            f"{referring_type}/{chart_type} template"
        )
    middle = expression[len(prefix) : len(expression) - len(suffix)]
    if not middle.strip():
        raise InstructionProcessingError(
            "referring expression contains no explicit target value"
        )
