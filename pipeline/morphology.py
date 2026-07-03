"""
morphology.py  --  PHASE 2 / Step 4 (morphological analysis)
============================================================

Per-cell shape + neighbourhood, adapted from the team's
``Cell_Shape_2D_SingleFrame`` notebook (Author: Linlin Li).

For every cell in every frame:
  * boundary = convex hull of the cell pixels (outlier-filtered),
  * Area (shoelace), Perimeter, Circularity = 4*pi*A/P^2,
  * Shape index = P / sqrt(A)   (vertex-model fluidity; SI* ~ 3.81),
  * Aspect ratio = lambda1/lambda2 and Elongation = 1 - lambda2/lambda1 (PCA),
  * Neighbour count via shared borders (``find_neighbors_from_labels``,
    4-connectivity) -- with an optional gap-tolerant "dilate" mode for the
    background gaps that Cellpose leaves between cells.

Output ``morphology.csv`` with columns:
    frame, time_min, cell_id, area_um2, perimeter_um, circularity,
    shape_index, aspect_ratio, elongation, n_neighbors, cx, cy
"""

from __future__ import annotations

import csv
import os

import numpy as np
from scipy.spatial import ConvexHull

try:                                   # PCA: prefer sklearn (their code), fall back to numpy
    from sklearn.decomposition import PCA
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

from wound_analysis.intercalation import adjacency_edges, neighbor_dict


# --------------------------------------------------------------------------- #
# neighbours (their find_neighbors_from_labels, vectorised, + gap tolerance)
# --------------------------------------------------------------------------- #
def find_neighbors(label_img, method="touch", dist_px=3):
    """dict label -> neighbour count.  'touch' = 4-connectivity shared border;
    'dilate' = expand labels ``dist_px`` first (tolerates background gaps)."""
    img = label_img
    if method == "dilate" and dist_px > 0:
        from skimage.segmentation import expand_labels
        img = expand_labels(label_img, distance=dist_px)
    nb = neighbor_dict(adjacency_edges(img))
    return {int(l): len(nb.get(int(l), ())) for l in np.unique(label_img) if l}


# --------------------------------------------------------------------------- #
# shape metrics (their compute_shape_metrics, trimmed to the diagram's outputs)
# --------------------------------------------------------------------------- #
def _filter_outliers(pts, iqr_mult=1.5, z_thresh=2.5, pct_high=95):
    if len(pts) < 4:
        return pts
    cen = pts.mean(0)
    d = np.linalg.norm(pts - cen, axis=1)
    q1, q3 = np.percentile(d, [25, 75])
    keep = (d <= q3 + iqr_mult * (q3 - q1))
    keep &= np.abs((d - d.mean()) / (d.std() + 1e-10)) <= z_thresh
    keep &= d <= np.percentile(d, pct_high)
    return pts[keep] if keep.sum() >= 3 else pts


def _pca_axes(verts):
    """Return (aspect_ratio, elongation) from PCA on boundary vertices."""
    v = verts - verts.mean(0)
    if _HAVE_SK and len(v) >= 2:
        pca = PCA(n_components=2).fit(v)
        lam = np.sqrt(np.maximum(pca.explained_variance_, 0))
    else:
        cov = np.cov(v.T)
        lam = np.sqrt(np.maximum(np.linalg.eigvalsh(cov)[::-1], 0))
    if lam.size == 2 and lam[1] > 1e-10:
        return float(lam[0] / lam[1]), float(1.0 - lam[1] / lam[0])
    return np.nan, np.nan


def compute_shape_metrics(cell_mask_2d, x_scale, y_scale, min_px=50, use_hull=True):
    coords = np.column_stack(np.where(cell_mask_2d > 0))    # (row, col)
    if len(coords) < min_px:
        return None
    pts = coords[:, ::-1].astype(float)                     # (x, y) px
    pts[:, 0] *= x_scale
    pts[:, 1] *= y_scale
    pts = _filter_outliers(pts)
    if len(pts) < 3:
        return None

    if use_hull:
        try:
            hull = ConvexHull(pts)
            b = pts[hull.vertices]
            ang = np.arctan2(b[:, 1] - b[:, 1].mean(), b[:, 0] - b[:, 0].mean())
            b = b[np.argsort(ang)]
        except Exception:
            b = pts
    else:
        b = pts
    x, y = b[:, 0], b[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perim = float(np.sum(np.sqrt(np.diff(np.r_[x, x[0]]) ** 2 + np.diff(np.r_[y, y[0]]) ** 2)))
    if area < 1e-9 or perim < 1e-9:
        return None
    circ = float(np.clip(4 * np.pi * area / perim ** 2, 0, 1))
    shape_index = perim / np.sqrt(area)
    ar, elong = _pca_axes(b)
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()
    return dict(area_um2=area, perimeter_um=perim, circularity=circ,
                shape_index=float(shape_index), aspect_ratio=ar, elongation=elong,
                cx=float(cx), cy=float(cy))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(labels, params, frame_idx=None, progress=None):
    px = params.px_size_um
    dt_min = params.dt_s / 60.0
    T = labels.shape[0]
    if frame_idx is None:
        frame_idx = list(range(T))
    rows = []
    for t in range(T):
        nbc = find_neighbors(labels[t], params.neighbor_method, params.neighbor_dist_px)
        for lab in np.unique(labels[t]):
            if lab == 0:
                continue
            m = compute_shape_metrics(labels[t] == lab, px, px,
                                      params.min_cell_pixels, params.shape_use_convexhull)
            if m is None:
                continue
            m["cell_id"] = int(lab)
            m["n_neighbors"] = int(nbc.get(int(lab), 0))
            m["frame"] = int(frame_idx[t])
            m["time_min"] = round(t * dt_min, 4)
            rows.append(m)
        if progress:
            progress((t + 1) / T, f"Morphology frame {t + 1}/{T}")

    cols = ["frame", "time_min", "cell_id", "area_um2", "perimeter_um", "circularity",
            "shape_index", "aspect_ratio", "elongation", "n_neighbors", "cx", "cy"]
    os.makedirs(params.out_dir, exist_ok=True)
    path = os.path.join(params.out_dir, "morphology.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    return rows, path
