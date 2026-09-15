"""Model-independent editing operations driven by a binary ground-truth mask."""

from __future__ import annotations

import colorsys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


EDIT_ACTIONS = frozenset({"highlight", "recolor", "extract", "remove"})
DEFAULT_COLOR = (230, 57, 70)
DEFAULT_HIGHLIGHT_STRENGTH = 0.65
DEFAULT_NEIGHBOR_RADIUS = 5


class EditingError(ValueError):
    """Base exception for invalid edit inputs or parameters."""


class ImageValidationError(EditingError):
    """Raised when the source image is not an RGB Pillow image."""


class MaskValidationError(EditingError):
    """Raised when the mask shape, type, or values are invalid."""


class EmptyMaskError(MaskValidationError):
    """Raised because v0 deliberately rejects masks with no foreground."""


class ParameterValidationError(EditingError):
    """Raised when action parameters are unknown or malformed."""


def edit(
    image: Image.Image,
    mask: Image.Image | np.ndarray,
    action: str,
    parameters: Mapping[str, Any] | None = None,
) -> Image.Image:
    """Apply one deterministic edit to an RGB Pillow image.

    Args:
        image: Source image. Phase 1B deliberately accepts only Pillow RGB.
        mask: Pillow ``L``/``1`` image or a two-dimensional NumPy array. Bool,
            ``{0, 1}``, and ``{0, 255}`` masks are accepted; other values fail.
        action: One of ``highlight``, ``recolor``, ``extract``, or ``remove``.
        parameters: Action-specific values documented in ``editing_spec_v0.md``.

    Returns:
        RGB output for highlight/recolor/remove, or RGBA for extract.

    Empty masks raise :class:`EmptyMaskError`; all-foreground masks are valid.
    """
    _validate_image(image)
    if type(action) is not str or action not in EDIT_ACTIONS:
        raise ParameterValidationError(
            f"action must be one of {sorted(EDIT_ACTIONS)}, got {action!r}"
        )
    parameter_map = _validate_parameter_mapping(parameters)
    binary_mask = _normalize_mask(mask, expected_size=image.size)
    source = np.asarray(image, dtype=np.uint8)

    if action == "highlight":
        _reject_unknown(parameter_map, {"strength"}, action)
        strength = _unit_float(
            parameter_map.get("strength", DEFAULT_HIGHLIGHT_STRENGTH), "strength"
        )
        return _highlight(source, binary_mask, strength)
    if action == "recolor":
        _reject_unknown(parameter_map, {"color"}, action)
        color = parse_color(parameter_map.get("color", DEFAULT_COLOR))
        return _recolor(source, binary_mask, color)
    if action == "extract":
        _reject_unknown(parameter_map, set(), action)
        return _extract(source, binary_mask)

    _reject_unknown(parameter_map, {"fill_mode", "color", "neighbor_radius"}, action)
    fill_mode = parameter_map.get("fill_mode", "color")
    if fill_mode not in {"color", "neighbor"}:
        raise ParameterValidationError(
            "remove fill_mode must be either 'color' or 'neighbor'"
        )
    fallback_color = parse_color(parameter_map.get("color", (255, 255, 255)))
    radius = parameter_map.get("neighbor_radius", DEFAULT_NEIGHBOR_RADIUS)
    if type(radius) is not int or not 1 <= radius <= 50:
        raise ParameterValidationError("neighbor_radius must be an integer in [1, 50]")
    return _remove(source, binary_mask, fill_mode, fallback_color, radius)


def parse_color(value: Any) -> tuple[int, int, int]:
    """Parse ``#RGB``, ``#RRGGBB``, or a three-integer RGB sequence."""
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 4 and text.startswith("#"):
            text = "#" + "".join(character * 2 for character in text[1:])
        if len(text) != 7 or not text.startswith("#"):
            raise ParameterValidationError(
                "color must use #RGB or #RRGGBB hexadecimal syntax"
            )
        try:
            return tuple(int(text[index : index + 2], 16) for index in (1, 3, 5))
        except ValueError as exc:
            raise ParameterValidationError(f"invalid hexadecimal color: {value!r}") from exc
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        components = tuple(value)
        if len(components) != 3:
            raise ParameterValidationError("RGB color must contain exactly 3 values")
        if any(type(component) is not int or not 0 <= component <= 255 for component in components):
            raise ParameterValidationError("RGB color values must be integers in [0, 255]")
        return components
    raise ParameterValidationError(
        "color must be a hexadecimal string or a three-integer RGB sequence"
    )


def _validate_image(image: Image.Image) -> None:
    if not isinstance(image, Image.Image):
        raise ImageValidationError("image must be a Pillow Image")
    if image.mode != "RGB":
        raise ImageValidationError(f"image mode must be 'RGB', got {image.mode!r}")
    if image.width <= 0 or image.height <= 0:
        raise ImageValidationError("image dimensions must be positive")


def _validate_parameter_mapping(
    parameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise ParameterValidationError("parameters must be a mapping or None")
    if any(type(key) is not str for key in parameters):
        raise ParameterValidationError("parameter names must be strings")
    return dict(parameters)


def _reject_unknown(
    parameters: Mapping[str, Any], allowed: set[str], action: str
) -> None:
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ParameterValidationError(
            f"unknown parameters for {action}: {', '.join(unknown)}"
        )


def _normalize_mask(
    mask: Image.Image | np.ndarray, *, expected_size: tuple[int, int]
) -> np.ndarray:
    if isinstance(mask, Image.Image):
        if mask.mode not in {"1", "L"}:
            raise MaskValidationError(
                f"Pillow mask mode must be '1' or 'L', got {mask.mode!r}"
            )
        array = np.asarray(mask)
    elif isinstance(mask, np.ndarray):
        array = mask
    else:
        raise MaskValidationError("mask must be a Pillow Image or NumPy array")

    if array.ndim != 2:
        raise MaskValidationError(
            f"mask must be two-dimensional, got shape {array.shape}"
        )
    actual_size = (int(array.shape[1]), int(array.shape[0]))
    if actual_size != expected_size:
        raise MaskValidationError(
            f"image/mask size mismatch: image={expected_size}, mask={actual_size}"
        )
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise MaskValidationError(f"unsupported mask dtype: {array.dtype}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise MaskValidationError("mask must not contain NaN or infinite values")
    values = set(np.unique(array).tolist())
    if not (values.issubset({0, 1}) or values.issubset({0, 255})):
        raise MaskValidationError(
            f"mask must contain only 0/1 or 0/255, got {sorted(values)}"
        )
    binary = np.asarray(array != 0, dtype=bool)
    if not binary.any():
        raise EmptyMaskError("empty masks are not editable in v0")
    return binary


def _unit_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ParameterValidationError(f"{name} must be a number in [0, 1]")
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ParameterValidationError(f"{name} must be a number in [0, 1]")
    return numeric


def _highlight(source: np.ndarray, mask: np.ndarray, strength: float) -> Image.Image:
    source_float = source.astype(np.float32)
    luminance = np.tensordot(
        source_float, np.asarray((0.299, 0.587, 0.114), dtype=np.float32), axes=([-1], [0])
    )
    muted = np.repeat((luminance * 0.55)[..., None], 3, axis=2)
    result = np.rint((1.0 - strength) * source_float + strength * muted)
    result[mask] = source[mask]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _recolor(
    source: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]
) -> Image.Image:
    result = source.copy()
    pixels = source[mask].astype(np.float32) / 255.0
    source_lightness = (pixels.max(axis=1) + pixels.min(axis=1)) / 2.0
    hue, _, saturation = colorsys.rgb_to_hls(*(component / 255.0 for component in color))
    chroma = (1.0 - np.abs(2.0 * source_lightness - 1.0)) * saturation
    hue_sector = hue * 6.0
    secondary = chroma * (1.0 - abs(hue_sector % 2.0 - 1.0))
    zero = np.zeros_like(chroma)
    sector = int(hue_sector) % 6
    components = (
        (chroma, secondary, zero),
        (secondary, chroma, zero),
        (zero, chroma, secondary),
        (zero, secondary, chroma),
        (secondary, zero, chroma),
        (chroma, zero, secondary),
    )[sector]
    offset = source_lightness - chroma / 2.0
    recolored = np.stack(components, axis=1) + offset[:, None]
    result[mask] = np.rint(np.clip(recolored, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(result)


def _extract(source: np.ndarray, mask: np.ndarray) -> Image.Image:
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    rgba = np.concatenate((source, alpha[..., None]), axis=2)
    return Image.fromarray(rgba, mode="RGBA")


def _remove(
    source: np.ndarray,
    mask: np.ndarray,
    fill_mode: str,
    fallback_color: tuple[int, int, int],
    neighbor_radius: int,
) -> Image.Image:
    if fill_mode == "neighbor":
        mask_image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8))
        filter_size = 2 * neighbor_radius + 1
        dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(filter_size))) != 0
        ring = np.logical_and(dilated, ~mask)
        if ring.any():
            fill_color = tuple(
                int(value) for value in np.rint(np.median(source[ring], axis=0))
            )
        else:
            fill_color = fallback_color
    else:
        fill_color = fallback_color
    result = source.copy()
    result[mask] = fill_color
    return Image.fromarray(result)

