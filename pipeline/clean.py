"""
clean.py  --  PHASE 1 (mask cleaning)
=====================================

Refine the raw segmentation into ``cleaned_mask.tiff``:
  * drop objects smaller than ``min_cell_area_px`` (noise / fragments),
  * fill internal holes,
  * optionally remove cells touching the image border (incomplete),
  * optionally smooth boundaries (median filter on labels),
  * relabel 1..N per frame so ids are compact.

(The "Chang" cleaning step in the workflow diagram.)
"""

from __future__ import annotations

import os

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from skimage.measure import label as cc_label


def _drop_by_area(labels, min_area, max_area=None):
    """Zero out labels whose pixel area is outside [min_area, max_area]."""
    out = labels.copy()
    ids, counts = np.unique(out, return_counts=True)
    for l, c in zip(ids, counts):
        if l == 0:
            continue
        if c < min_area or (max_area is not None and c > max_area):
            out[out == l] = 0
    return out


def clean_frame(mask, params, max_area_px=None):
    out = mask.copy()
    # fill holes per label
    if params.fill_holes:
        filled = np.zeros_like(out)
        for lab in np.unique(out):
            if lab == 0:
                continue
            filled[ndi.binary_fill_holes(out == lab)] = lab
        out = filled
    # remove border-touching cells
    if params.remove_border_cells:
        out = clear_border(out)
    # smooth boundaries (may create stray fragments -> filtered below)
    if params.smooth_boundaries:
        out = ndi.median_filter(out, size=3)
    # compact relabel (keep cells separate -> connectivity 1)
    out = cc_label(out, connectivity=1)
    # final size filtering AFTER relabel: drops 1-px fragments and any
    # implausibly large catch-all region (e.g. a watershed background basin)
    if params.min_cell_area_px > 0:
        out = _drop_by_area(out, params.min_cell_area_px, max_area_px)
        out = cc_label(out, connectivity=1)
    return out.astype(np.uint16)


def _frame_max_area(mask, min_area):
    """Per-frame upper area cut: 8x the median plausible-cell area, so a
    watershed background catch-all is dropped but real cells are kept."""
    ids, counts = np.unique(mask, return_counts=True)
    areas = counts[(ids != 0) & (counts >= min_area)]
    if areas.size < 3:
        return None
    return float(max(np.median(areas) * 8.0, min_area * 30.0))


def run(masks, params, progress=None):
    cleaned = np.zeros_like(masks)
    for t in range(masks.shape[0]):
        max_area = _frame_max_area(masks[t], params.min_cell_area_px)
        cleaned[t] = clean_frame(masks[t], params, max_area_px=max_area)
        if progress:
            progress((t + 1) / masks.shape[0], f"Cleaning frame {t + 1}/{masks.shape[0]}")
    os.makedirs(params.out_dir, exist_ok=True)
    path = os.path.join(params.out_dir, "cleaned_mask.tiff")
    tifffile.imwrite(path, cleaned)
    return cleaned, path
