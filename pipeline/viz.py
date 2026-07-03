"""
viz.py  --  visualisations (GIFs, trajectory, neighbour & shape maps)
====================================================================

Adapted from the team's notebooks (Author: Linlin Li): consistent-colour
``render_tracked`` + tracking GIF, trajectory overlay, cells-per-frame curve,
neighbour-count maps, and shape-metric colour maps.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio.v2 as iio


# --------------------------------------------------------------------------- #
# tracked-mask rendering + GIF  (their render_tracked / colours)
# --------------------------------------------------------------------------- #
def track_colors(track_ids, seed=42):
    rng = np.random.default_rng(seed)
    return {int(t): rng.random(3) for t in track_ids}


def render_tracked(frame_mask, colors):
    canvas = np.zeros((*frame_mask.shape, 3), dtype=np.float32)
    for tid in np.unique(frame_mask):
        if tid == 0:
            continue
        canvas[frame_mask == tid] = colors.get(int(tid), [0.5, 0.5, 0.5])
    return canvas


def save_tracking_gif(tracked, out_dir, duration=0.25, name="tracking_result.gif"):
    """Animated GIF: one GIF-frame per movie-frame, consistent colour per id."""
    ids = sorted({int(i) for t in range(tracked.shape[0]) for i in np.unique(tracked[t]) if i})
    colors = track_colors(ids)
    frames = [(render_tracked(tracked[t], colors) * 255).astype(np.uint8)
              for t in range(tracked.shape[0])]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    iio.mimsave(path, frames, duration=duration, loop=0)
    return path, colors


def save_overlay_gif(stack, tracked, out_dir, duration=0.25,
                     name="tracking_overlay.gif"):
    """GIF of raw frames with coloured cell outlines (id-consistent)."""
    from skimage.segmentation import find_boundaries
    from wound_analysis import detection
    ids = sorted({int(i) for t in range(tracked.shape[0]) for i in np.unique(tracked[t]) if i})
    colors = track_colors(ids)
    frames = []
    for t in range(stack.shape[0]):
        base = detection.normalize(stack[t])
        rgb = np.dstack([base] * 3).astype(np.float32)
        bnd = find_boundaries(tracked[t], mode="outer")
        col = render_tracked(tracked[t], colors)
        rgb[bnd] = col[bnd]
        frames.append((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    iio.mimsave(path, frames, duration=duration, loop=0)
    return path


def trajectory_plot(tracked, out_dir, min_presence=None, name="trajectory_plot.png"):
    T = tracked.shape[0]
    ids = sorted({int(i) for t in range(T) for i in np.unique(tracked[t]) if i})
    colors = track_colors(ids)
    if min_presence is None:
        min_presence = T // 2
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(render_tracked(tracked[0], colors), alpha=0.4)
    for i in ids:
        xs, ys = [], []
        present = 0
        for t in range(T):
            yy, xx = np.where(tracked[t] == i)
            if xx.size:
                xs.append(xx.mean()); ys.append(yy.mean()); present += 1
        if present >= min_presence:
            ax.plot(xs, ys, "-", color=colors[i], lw=0.9, alpha=0.8)
    ax.set_title(f"Cell trajectories (present in ≥{min_presence} frames)")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)"); ax.axis("image")
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path


def cells_per_frame_plot(tracked, times_min, out_dir, name="cells_per_frame.png"):
    T = tracked.shape[0]
    counts = [len(np.unique(tracked[t])) - 1 for t in range(T)]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(times_min, counts, "o-", ms=4, color="#1f77b4")
    ax.set_xlabel("time (min)"); ax.set_ylabel("tracked cells")
    ax.set_title("Number of tracked cells per frame"); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# neighbour-count map  (their label2rgb_by_neighbor_count)
# --------------------------------------------------------------------------- #
def neighbor_count_rgb(mask, counts, cmap="viridis", vmin=None, vmax=None):
    rgb = np.zeros((*mask.shape, 3), np.float32)
    vals = list(counts.values()) or [0]
    vmin = min(vals) if vmin is None else vmin
    vmax = max(vals) if vmax is None else vmax
    norm = mcolors.Normalize(vmin, max(vmax, vmin + 1), clip=True)
    cmo = plt.get_cmap(cmap)
    for lab in np.unique(mask):
        if lab == 0:
            continue
        rgb[mask == lab] = cmo(norm(counts.get(int(lab), 0)))[:3]
    return rgb, norm, cmo


# --------------------------------------------------------------------------- #
# shape-metric colour map  (their plot_cells_by_metric)
# --------------------------------------------------------------------------- #
def metric_rgb(mask, value_by_label, cmap="RdYlBu_r", lo=5, hi=95):
    vals = np.array([v for v in value_by_label.values() if np.isfinite(v)])
    rgb = np.zeros((*mask.shape, 3), np.float32)
    if vals.size == 0:
        return rgb, None, None
    vmin, vmax = np.percentile(vals, [lo, hi])
    norm = mcolors.Normalize(vmin, vmax)
    cmo = plt.get_cmap(cmap)
    for lab, v in value_by_label.items():
        if np.isfinite(v):
            rgb[mask == lab] = cmo(norm(v))[:3]
    return rgb, norm, cmo
