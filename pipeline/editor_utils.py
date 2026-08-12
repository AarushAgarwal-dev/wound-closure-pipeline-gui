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


def extended_split_line_mask(drawn_mask, source_shape):
    """Extend a drawn line through the full image while preserving its angle.

    The canvas line only needs to touch the cell. Extending it ensures that a
    short drag still cuts completely through a wide cell instead of leaving a
    one-pixel connection around either endpoint.
    """
    from skimage.draw import line

    drawn_mask = np.asarray(drawn_mask, dtype=bool)
    ys, xs = np.nonzero(drawn_mask)
    result = np.zeros(source_shape, dtype=bool)
    if len(xs) < 2:
        result[:drawn_mask.shape[0], :drawn_mask.shape[1]] = drawn_mask
        return result

    points = np.column_stack((xs, ys)).astype(float)
    center = points.mean(axis=0)
    centered = points - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    direction = axes[0]
    distance = float(np.hypot(*source_shape)) * 2.0
    start = center - direction * distance
    end = center + direction * distance
    rr, cc = line(round(start[1]), round(start[0]),
                  round(end[1]), round(end[0]))
    valid = ((rr >= 0) & (rr < source_shape[0]) &
             (cc >= 0) & (cc < source_shape[1]))
    result[rr[valid], cc[valid]] = True
    return result


def split_labels_with_drawn_line(labels, drawn_mask, thickness=3):
    """Partition each touched label into exactly two labels.

    Returns ``(updated_labels, touched_ids, split_ids)``. A touched label is
    left unchanged unless the drawn line has cell pixels on both sides. Every
    original cell pixel is retained: pixels in the thick line band are merged
    into the larger side instead of becoming background or a third fragment.
    """
    original = np.asarray(labels)
    drawn_mask = np.asarray(drawn_mask, dtype=bool)
    touched_ids = [int(v) for v in np.unique(original[drawn_mask]) if v > 0]
    if not touched_ids:
        return original.copy(), [], []

    line_y, line_x = np.nonzero(drawn_mask)
    if len(line_x) < 2:
        return original.copy(), touched_ids, []
    line_points = np.column_stack((line_x, line_y)).astype(float)
    center = line_points.mean(axis=0)
    _, _, axes = np.linalg.svd(line_points - center, full_matrices=False)
    direction = axes[0]
    # A normal vector gives a stable signed side for every source-mask pixel.
    normal = np.array((-direction[1], direction[0]))
    half_band = max(0.0, float(thickness) / 2.0)

    updated = original.copy()
    next_id = int(updated.max()) + 1
    split_ids = []
    for cell_id in touched_ids:
        cell = original == cell_id
        cell_y, cell_x = np.nonzero(cell)
        signed_distance = ((cell_x - center[0]) * normal[0] +
                           (cell_y - center[1]) * normal[1])
        first = signed_distance < -half_band
        second = signed_distance > half_band

        # If a very thick band covers one entire side, retry at the line
        # centre. We still require pixels on both sides to accept the split.
        if not first.any() or not second.any():
            first = signed_distance < 0
            second = signed_distance > 0
        if not first.any() or not second.any():
            continue

        line_band = ~(first | second)
        if np.count_nonzero(first) >= np.count_nonzero(second):
            first |= line_band
        else:
            second |= line_band

        # Preserve the old ID on the larger side and use exactly one new ID on
        # the other. This retains the complete original cell area.
        if np.count_nonzero(second) > np.count_nonzero(first):
            first, second = second, first
        updated[cell] = 0
        updated[cell_y[first], cell_x[first]] = cell_id
        updated[cell_y[second], cell_x[second]] = next_id
        next_id += 1
        split_ids.append(cell_id)

    return updated, touched_ids, split_ids


def apply_brush_stroke(labels, drawn_mask, replace_existing=False):
    """Apply a freehand brush stroke as one new label.

    By default the brush fills background only. ``replace_existing`` allows a
    correction stroke to replace pixels belonging to existing cells.
    Returns ``(updated_labels, new_id, changed_pixel_count)``.
    """
    updated = np.asarray(labels).copy()
    drawn_mask = np.asarray(drawn_mask, dtype=bool)
    target = drawn_mask if replace_existing else drawn_mask & (updated == 0)
    changed = int(np.count_nonzero(target))
    if not changed:
        return updated, None, 0
    new_id = int(updated.max()) + 1 if updated.max() > 0 else 1
    updated[target] = new_id
    return updated, new_id, changed
