"""Coordinate and image-scaling helpers for the manual mask editor."""

from __future__ import annotations

import numpy as np
from PIL import Image


def fit_manual_editor_image(image, max_width=600, max_height=720):
    """Scale a large editor image to fit the page, preserving aspect ratio."""
    scale = min(1.0, max_width / image.width, max_height / image.height)
    if scale >= 1.0:
        return image, 1.0
    size = (max(1, round(image.width * scale)),
            max(1, round(image.height * scale)))
    resampling = getattr(Image, "Resampling", Image)
    return image.resize(size, resampling.LANCZOS), scale


def manual_editor_click_to_source(value, scale, source_shape):
    """Validate a displayed-image click and map it to source-mask pixels."""
    if not isinstance(value, dict) or "x" not in value or "y" not in value:
        return None
    try:
        x = int(float(value["x"]) / scale)
        y = int(float(value["y"]) / scale)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    height, width = source_shape
    return {"x": min(max(x, 0), width - 1),
            "y": min(max(y, 0), height - 1)}


def manual_editor_mask_to_source(mask, source_shape):
    """Map a displayed canvas stroke back to the full-resolution mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == tuple(source_shape):
        return mask
    height, width = source_shape
    resampling = getattr(Image, "Resampling", Image)
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (width, height), resampling.NEAREST)
    return np.asarray(resized) > 0
