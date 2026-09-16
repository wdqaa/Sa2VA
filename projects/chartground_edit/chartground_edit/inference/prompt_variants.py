"""Pre-registered Prompt variants for the Phase 2C val-only diagnostic."""

from __future__ import annotations


FULL_INSTRUCTION = "full_instruction"
TARGET_ONLY_EN = "target_only_en"
TARGET_ONLY_ZH = "target_only_zh"
PROMPT_VARIANT_ORDER = (FULL_INSTRUCTION, TARGET_ONLY_EN, TARGET_ONLY_ZH)

PROMPT_TEMPLATES = {
    FULL_INSTRUCTION: (
        "<image>Please segment the chart element targeted by this instruction: "
        "{instruction}\nPlease respond with a segmentation mask."
    ),
    TARGET_ONLY_EN: (
        "<image>Please segment the chart element described by this referring expression: "
        "{referring_expression}\nPlease respond with a segmentation mask."
    ),
    TARGET_ONLY_ZH: (
        "<image>请分割图中由以下指代表达式指定的图表元素：{referring_expression}\n"
        "请使用 [SEG] 标记返回分割掩码。"
    ),
}


def build_prompt_variant(
    variant: str, *, instruction: str, referring_expression: str
) -> str:
    """Build exactly one registered Prompt without translation or enrichment."""
    if variant not in PROMPT_TEMPLATES:
        raise ValueError(f"unknown Prompt variant: {variant!r}")
    if type(instruction) is not str or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    if type(referring_expression) is not str or not referring_expression.strip():
        raise ValueError("referring_expression must be a non-empty string")
    return PROMPT_TEMPLATES[variant].format(
        instruction=instruction,
        referring_expression=referring_expression,
    )


def parse_prompt_variants(value: str) -> tuple[str, ...]:
    """Require all three pre-registered variants and return canonical order."""
    if type(value) is not str:
        raise ValueError("prompt variants must be a comma-separated string")
    requested = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(requested) != len(set(requested)):
        raise ValueError("prompt variants must not contain duplicates")
    unknown = sorted(set(requested) - set(PROMPT_VARIANT_ORDER))
    if unknown:
        raise ValueError(f"unknown Prompt variant(s): {', '.join(unknown)}")
    if set(requested) != set(PROMPT_VARIANT_ORDER):
        raise ValueError(
            "Phase 2C requires exactly: " + ",".join(PROMPT_VARIANT_ORDER)
        )
    return PROMPT_VARIANT_ORDER
