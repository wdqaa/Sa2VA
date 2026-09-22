"""Synchronized geometric transforms for an RGB image and binary mask."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class TransformParams:
    """Fully materialized geometry parameters that can be replayed."""

    resize_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    horizontal_flip: bool


def sample_transform_params(
    original_size: tuple[int, int],
    *,
    resize_size: tuple[int, int],
    crop_size: tuple[int, int],
    horizontal_flip_probability: float,
    seed: int,
) -> TransformParams:
    """Sample reproducible resize/crop/flip parameters from one seed.

    All sizes use Pillow's ``(width, height)`` convention.
    """
    if original_size[0] <= 0 or original_size[1] <= 0:
        raise ValueError("original_size must be positive")
    if resize_size[0] <= 0 or resize_size[1] <= 0:
        raise ValueError("resize_size must be positive")
    crop_width, crop_height = crop_size
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("crop_size must be positive")
    if crop_width > resize_size[0] or crop_height > resize_size[1]:
        raise ValueError("crop_size must fit inside resize_size")
    if not 0.0 <= horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal_flip_probability must be in [0, 1]")

    rng = np.random.default_rng(seed)
    max_left = resize_size[0] - crop_width
    max_top = resize_size[1] - crop_height
    left = int(rng.integers(0, max_left + 1))
    top = int(rng.integers(0, max_top + 1))
    do_flip = bool(rng.random() < horizontal_flip_probability)
    return TransformParams(
        resize_size=resize_size,
        crop_box=(left, top, left + crop_width, top + crop_height),
        horizontal_flip=do_flip,
    )


def synchronized_resize(
    image: Image.Image,
    mask: Image.Image,
    size: tuple[int, int],
    *,
    image_resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> tuple[Image.Image, Image.Image]:
    """Resize image with smooth interpolation and mask with nearest neighbor."""
    if image.size != mask.size:
        raise ValueError(f"image/mask size mismatch: {image.size} != {mask.size}")
    resized_image = image.resize(size, resample=image_resample)
    resized_mask = mask.convert("L").resize(size, resample=Image.Resampling.NEAREST)
    return resized_image, _force_binary(resized_mask)


def synchronized_crop(
    image: Image.Image,
    mask: Image.Image,
    crop_box: tuple[int, int, int, int],
) -> tuple[Image.Image, Image.Image]:
    """Apply the same Pillow crop box to image and mask."""
    if image.size != mask.size:
        raise ValueError(f"image/mask size mismatch: {image.size} != {mask.size}")
    left, top, right, bottom = crop_box
    if left < 0 or top < 0 or right > image.width or bottom > image.height:
        raise ValueError(f"crop_box {crop_box} is outside image size {image.size}")
    if right <= left or bottom <= top:
        raise ValueError("crop_box must have positive area")
    return image.crop(crop_box), _force_binary(mask.convert("L").crop(crop_box))


def synchronized_horizontal_flip(
    image: Image.Image, mask: Image.Image
) -> tuple[Image.Image, Image.Image]:
    """Flip image and mask together around the vertical axis."""
    if image.size != mask.size:
        raise ValueError(f"image/mask size mismatch: {image.size} != {mask.size}")
    return ImageOps.mirror(image), _force_binary(ImageOps.mirror(mask.convert("L")))


def apply_transform_params(
    image: Image.Image,
    mask: Image.Image,
    params: TransformParams,
    *,
    image_resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> tuple[Image.Image, Image.Image]:
    """Replay one sampled transform identically on image and mask."""
    image, mask = synchronized_resize(
        image, mask, params.resize_size, image_resample=image_resample
    )
    image, mask = synchronized_crop(image, mask, params.crop_box)
    if params.horizontal_flip:
        image, mask = synchronized_horizontal_flip(image, mask)
    return image, _force_binary(mask)


def _force_binary(mask: Image.Image) -> Image.Image:
    array = np.asarray(mask, dtype=np.uint8)
    binary = np.where(array >= 128, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")

