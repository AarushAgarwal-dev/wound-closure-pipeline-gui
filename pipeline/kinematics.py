"""
kinematics.py  --  PHASE 2 / Step 5 (edge detection & kinematics)
=================================================================

For a tracked cell, detect its boundary in each frame ("find 1 from 0"),
sample ``n_edge_points`` evenly along the edge, follow those points over time
and compute their velocity vectors (u, v).  Writes ``edge_velocity.csv`` with:

    frame, time_min, cell_id, point_index, x_um, y_um, u_um_min, v_um_min, speed_um_min

Points are made comparable across frames by resampling each closed contour to a
fixed number of arc-length-spaced points and anchoring index 0 at the boundary
vertex pointing along +x from the cell centroid.
"""

from __future__ import annotations

import csv
import os

import numpy as np
from skimage.measure import find_contours


# --------------------------------------------------------------------------- #
# contour helpers
# --------------------------------------------------------------------------- #
def _resample_closed(xy, n):
    """Resample a closed polyline to ``n`` arc-length-spaced points."""
    pts = np.vstack([xy, xy[:1]])
    seg = np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    if total == 0:
        return np.repeat(xy[:1], n, axis=0)
    targets = np.linspace(0, total, n, endpoint=False)
    x = np.interp(targets, s, pts[:, 0])
    y = np.interp(targets, s, pts[:, 1])
    return np.column_stack([x, y])


def cell_boundary(mask_bool, n_points):
    """Ordered (n, 2) edge points (x, y), anchored at +x from the centroid."""
    cs = find_contours(mask_bool.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)              # longest contour
    xy = np.column_stack([c[:, 1], c[:, 0]])   # (x, y)
    # enforce counter-clockwise winding (positive signed area)
    area = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
    if area < 0:
        xy = xy[::-1]
    res = _resample_closed(xy, n_points)
    # anchor: rotate so index 0 is the point at the smallest angle from centroid
    cen = res.mean(0)
    ang = np.arctan2(res[:, 1] - cen[1], res[:, 0] - cen[0])
    k = int(np.argmin(np.abs(((ang + np.pi) % (2 * np.pi)) - np.pi)))
    return np.roll(res, -k, axis=0)


# --------------------------------------------------------------------------- #
# cell selection
# --------------------------------------------------------------------------- #
def most_persistent_cell(tracked):
    ids, counts = np.unique(tracked[tracked > 0], return_counts=True)
    # weight by number of frames present, not pixel count
    present = {}
    for t in range(tracked.shape[0]):
        for l in np.unique(tracked[t]):
            if l:
                present[int(l)] = present.get(int(l), 0) + 1
    return int(max(present, key=present.get)) if present else 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run(tracked, params, frame_idx=None, progress=None):
    px = params.px_size_um
    dt_min = params.dt_s / 60.0
    T = tracked.shape[0]
    if frame_idx is None:
        frame_idx = list(range(T))
    n = params.n_edge_points

    cid = params.edge_cell_id
    if cid is None or cid < 0:
        cid = most_persistent_cell(tracked)

    # boundary points per frame (None where the cell is absent)
    contours = {}
    for t in range(T):
        m = tracked[t] == cid
        if m.sum() < n:           # need enough boundary pixels
            contours[t] = None
            continue
        b = cell_boundary(m, n)
        contours[t] = b * px if b is not None else None
        if progress:
            progress((t + 1) / T, f"Edge sampling frame {t + 1}/{T}")

    # velocity between consecutive present frames (per point)
    rows = []
    frames_present = [t for t in range(T) if contours[t] is not None]
    for j, t in enumerate(frames_present):
        b = contours[t]
        # forward difference to next present frame
        if j + 1 < len(frames_present):
            tn = frames_present[j + 1]
            dtf = (tn - t) * dt_min
            vel = (contours[tn] - b) / max(dtf, 1e-6)
        else:
            vel = np.full_like(b, np.nan)
        for i in range(n):
            u, v = vel[i]
            rows.append(dict(frame=int(frame_idx[t]), time_min=round(t * dt_min, 4),
                             cell_id=int(cid), point_index=i,
                             x_um=float(b[i, 0]), y_um=float(b[i, 1]),
                             u_um_min=float(u), v_um_min=float(v),
                             speed_um_min=float(np.hypot(u, v))))

    cols = ["frame", "time_min", "cell_id", "point_index",
            "x_um", "y_um", "u_um_min", "v_um_min", "speed_um_min"]
    os.makedirs(params.out_dir, exist_ok=True)
    path = os.path.join(params.out_dir, "edge_velocity.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows, path, cid, contours, frames_present


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def plot(tracked, stack, contours, frames_present, cid, params, frame_idx=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wound_analysis import detection

    px = params.px_size_um
    dt_min = params.dt_s / 60.0
    n = params.n_edge_points
    if not frames_present:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    # (a) quiver of edge velocity on a representative frame
    t = frames_present[len(frames_present) // 3]
    ax = axes[0]
    ax.imshow(detection.normalize(stack[t]), cmap="gray")
    b = contours[t] / px
    j = frames_present.index(t)
    if j + 1 < len(frames_present):
        tn = frames_present[j + 1]
        vel = (contours[tn] - contours[t]) / max((tn - t) * dt_min, 1e-6)
        ax.quiver(b[:, 0], b[:, 1], vel[:, 0], -vel[:, 1], np.hypot(vel[:, 0], vel[:, 1]),
                  cmap="jet", scale=40, width=0.004)
    ax.plot(np.r_[b[:, 0], b[0, 0]], np.r_[b[:, 1], b[0, 1]], "-", color="lime", lw=1)
    cy, cx = tracked[t].shape[0] // 2, tracked[t].shape[1] // 2
    bx, by = b.mean(0)
    ax.set_xlim(bx - 70, bx + 70); ax.set_ylim(by + 70, by - 70)
    ax.set_title(f"Edge-velocity vectors  (cell {cid}, t={t * dt_min:.1f} min)")
    ax.axis("off")

    # (b) speed kymograph: edge point index vs time
    ax = axes[1]
    speed = np.full((n, len(frames_present)), np.nan)
    for j, t in enumerate(frames_present):
        if j + 1 < len(frames_present):
            tn = frames_present[j + 1]
            vel = (contours[tn] - contours[t]) / max((tn - t) * dt_min, 1e-6)
            speed[:, j] = np.hypot(vel[:, 0], vel[:, 1])
    im = ax.imshow(speed, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, len(frames_present) * dt_min, 0, n])
    ax.set_xlabel("time (min)"); ax.set_ylabel("edge point index")
    ax.set_title("Edge-point speed kymograph")
    fig.colorbar(im, ax=ax, label="speed (µm/min)")
    fig.tight_layout()
    path = os.path.join(params.out_dir, "edge_velocity_plot.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
