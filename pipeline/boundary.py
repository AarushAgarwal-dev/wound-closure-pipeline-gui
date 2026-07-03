"""
boundary.py  --  Cell Cluster Boundary / Wound Edge Detection & Analysis
=========================================================================

Adapted from the team's ``Cell_Cluster_Boundary`` notebook.

Detects the inner boundary (wound edge) of the cell cluster per timeframe,
samples seed vertices, tracks them across frames, and computes wound-closure
velocity.  Also assigns cells to concentric *layers* (BFS from the wound)
and samples fluorescence intensity along the wound boundary.

Outputs:
  - cluster_boundary_points.csv   — timeframe, point_id, x, y
  - cluster_boundary_velocity.csv — timeframe_from, timeframe_to, point_id, dx, dy, speed
  - cell_layers.csv               — timeframe, cell_label, layer
  - boundary_intensity.csv        — timeframe, point_id, intensity
  - wound_area.csv                — timeframe, wound_area_px
  - Various matplotlib plot PNGs
"""

from __future__ import annotations

import csv
import os
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.ndimage import binary_dilation
from skimage import measure


# ---------------------------------------------------------------------------
# Helper functions (from the notebook)
# ---------------------------------------------------------------------------

def get_inner_contour(frame: np.ndarray, min_area: int):
    """
    Inner boundary = wound edge (cells vs wound).

    Strategy:
      1. All zero-value pixels are either background (touches border) or wound.
      2. Label connected zero-regions, drop those touching the image border.
      3. Largest remaining region is the wound; return its contour and centroid.

    Returns (pts, centroid) or (None, None).
    """
    black = (frame == 0).astype(np.uint8)
    labeled = measure.label(black, connectivity=2)

    h, w = frame.shape
    border_labels = (
        set(labeled[0, :])
        | set(labeled[-1, :])
        | set(labeled[:, 0])
        | set(labeled[:, -1])
    )
    border_labels.discard(0)

    best_label, best_area = None, 0
    for region in measure.regionprops(labeled):
        if region.label in border_labels:
            continue
        if region.area > best_area:
            best_area = region.area
            best_label = region.label

    if best_label is None or best_area < min_area:
        return None, None

    wound_mask = (labeled == best_label).astype(np.uint8)
    contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None

    pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    ys, xs = np.where(wound_mask)
    centroid = np.array([xs.mean(), ys.mean()])
    return pts, centroid


def sample_contour(pts: np.ndarray, n: int) -> np.ndarray:
    """Evenly sample n points along a closed contour by arc length."""
    diffs = np.diff(pts, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
    close_len = np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1])
    total_len = cumlen[-1] + close_len

    targets = np.linspace(0.0, total_len, n, endpoint=False)
    sampled = np.empty((n, 2))
    for i, s in enumerate(targets):
        if s <= cumlen[-1]:
            idx = min(int(np.searchsorted(cumlen, s, side="right")) - 1,
                      len(pts) - 2)
            denom = seg_len[idx] if seg_len[idx] > 0 else 1.0
            t = (s - cumlen[idx]) / denom
            sampled[i] = pts[idx] + t * (pts[idx + 1] - pts[idx])
        else:
            t = (s - cumlen[-1]) / close_len if close_len > 0 else 0.0
            sampled[i] = pts[-1] + t * (pts[0] - pts[-1])
    return sampled


def order_by_angle(pts: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Sort points counter-clockwise by angle from centroid."""
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    return pts[np.argsort(angles)]


def align_to_previous(pts_prev: np.ndarray, pts_curr: np.ndarray) -> np.ndarray:
    """
    Cyclic rotation (+/- direction) of pts_curr that minimises total
    Euclidean distance to pts_prev. Keeps point_id stable across frames.
    """
    n = len(pts_prev)
    best_cost, best = np.inf, pts_curr
    for direction in (1, -1):
        cand = pts_curr[::direction]
        for shift in range(n):
            rolled = np.roll(cand, shift, axis=0)
            cost = np.sum(np.hypot(
                pts_prev[:, 0] - rolled[:, 0],
                pts_prev[:, 1] - rolled[:, 1],
            ))
            if cost < best_cost:
                best_cost, best = cost, rolled.copy()
    return best


def point_id_color(pid, n_points):
    """Stable colour for a boundary point id (same id -> same colour, every
    frame and in every plot). Evenly spaced hues around the colour wheel."""
    n = max(int(n_points), 1)
    return plt.cm.hsv((int(pid) % n) / n)


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def process_all_frames(stack, n_points=10, min_wound_area=200,
                       progress=None, fill_all=True):
    """
    Detect the wound boundary in each frame, sample seed vertices, and
    align them across frames.

    If ``fill_all`` is True, frames where no wound edge is detected (e.g. after
    the wound has closed, or it shrank below ``min_wound_area``) are filled by
    carrying the nearest detected boundary forward (and back-filling any leading
    gap). This guarantees EVERY frame gets boundary points, so the downstream
    intensity / velocity / heatmaps span the whole movie. ``detected`` marks
    which frames were real detections vs carried.

    Returns (boundary_df_rows, frame_stats).
    """
    import pandas as pd
    T = stack.shape[0]
    per_frame = [None] * T          # sampled boundary (n_points, 2) or None
    detected = [False] * T
    prev_inner = None
    frame_stats = []

    for t in range(T):
        frame = stack[t]
        inner_pts, wound_centroid = get_inner_contour(frame, min_wound_area)
        if inner_pts is not None and len(inner_pts) >= n_points:
            sampled_inner = sample_contour(inner_pts, n_points)
            sampled_inner = order_by_angle(sampled_inner, wound_centroid)
            if prev_inner is not None:
                sampled_inner = align_to_previous(prev_inner, sampled_inner)
            prev_inner = sampled_inner
            per_frame[t] = sampled_inner
            detected[t] = True
            frame_stats.append({"frame": t, "wound_area": int(inner_pts.shape[0])})
        if progress:
            progress((t + 1) / T, f"Boundary frame {t + 1}/{T}")

    if fill_all:
        # forward-fill gaps with the last detected boundary ...
        last = None
        for t in range(T):
            if per_frame[t] is not None:
                last = per_frame[t]
            elif last is not None:
                per_frame[t] = last
        # ... then back-fill any leading gap with the first detected boundary
        nxt = None
        for t in range(T - 1, -1, -1):
            if per_frame[t] is not None:
                nxt = per_frame[t]
            elif nxt is not None:
                per_frame[t] = nxt

    rows = []
    for t in range(T):
        sampled_inner = per_frame[t]
        if sampled_inner is None:
            continue
        for pid in range(n_points):
            rows.append({
                "timeframe": t,
                "point_id": pid,
                "x": round(float(sampled_inner[pid, 0]), 3),
                "y": round(float(sampled_inner[pid, 1]), 3),
                "detected": bool(detected[t]),
            })

    df = pd.DataFrame(rows, columns=["timeframe", "point_id", "x", "y", "detected"])
    return df, frame_stats


def compute_velocity(boundary_df, time_interval=1.0, px_um=1.0):
    """Per-point velocity between consecutive frames.

    As well as the raw displacement magnitude, this returns two physically
    calibrated, biologically meaningful columns:

      * ``speed_um_min``    -- |displacement| in µm/min.
      * ``v_radial_um_min`` -- the *closure* velocity: displacement projected
        onto the inward radial direction (toward the wound centre), in µm/min.
        POSITIVE = boundary moving inward (wound closing); NEGATIVE = retreating
        (opening).

    The radial (normal) component is what biologists mean by "wound-closure
    velocity". The bare ``|dx, dy|`` magnitude also counts *tangential* sliding
    of the re-sampled seed vertices along the boundary — a re-sampling artefact,
    not real closure — so it over-states the true closure rate.

    Parameters
    ----------
    time_interval : seconds between consecutive frames (params.dt_s).
    px_um         : micrometres per pixel (params.px_size_um).
    """
    import pandas as pd
    vel_rows = []
    frames = sorted(boundary_df["timeframe"].unique())
    dt_min = (time_interval / 60.0) if time_interval > 0 else 1.0  # s -> min
    for i in range(len(frames) - 1):
        t0, t1 = frames[i], frames[i + 1]
        f0 = boundary_df[boundary_df["timeframe"] == t0].set_index("point_id")
        f1 = boundary_df[boundary_df["timeframe"] == t1].set_index("point_id")
        # wound centre for this frame = centroid of the boundary ring
        cx, cy = f0["x"].mean(), f0["y"].mean()
        for pid in f0.index.intersection(f1.index):
            x0, y0 = f0.loc[pid, "x"], f0.loc[pid, "y"]
            dx = f1.loc[pid, "x"] - x0
            dy = f1.loc[pid, "y"] - y0
            # inward radial unit vector (point -> wound centre)
            rx, ry = cx - x0, cy - y0
            rnorm = np.hypot(rx, ry)
            ux, uy = (rx / rnorm, ry / rnorm) if rnorm > 1e-9 else (0.0, 0.0)
            v_radial_px = dx * ux + dy * uy          # +ve = inward (closing)
            mag_px = np.hypot(dx, dy)
            vel_rows.append({
                "timeframe_from": t0,
                "timeframe_to": t1,
                "point_id": pid,
                "dx": round(dx, 3),
                "dy": round(dy, 3),
                "speed": round(mag_px / time_interval, 4),               # px/s (legacy)
                "speed_um_min": round(mag_px * px_um / dt_min, 4),       # µm/min
                "v_radial_um_min": round(v_radial_px * px_um / dt_min, 4),  # µm/min, +inward
            })
    return pd.DataFrame(vel_rows)


def compute_wound_area(stack, min_wound_area=200):
    """Compute wound area in pixels for each frame."""
    import pandas as pd
    rows = []
    for t in range(stack.shape[0]):
        frame = stack[t]
        inner_pts, _ = get_inner_contour(frame, min_wound_area)
        if inner_pts is not None:
            # Use wound mask area
            black = (frame == 0).astype(np.uint8)
            labeled = measure.label(black, connectivity=2)
            h, w = frame.shape
            border_labels = (
                set(labeled[0, :]) | set(labeled[-1, :])
                | set(labeled[:, 0]) | set(labeled[:, -1])
            )
            border_labels.discard(0)
            best_label, best_area = None, 0
            for region in measure.regionprops(labeled):
                if region.label in border_labels:
                    continue
                if region.area > best_area:
                    best_area, best_label = region.area, region.label
            rows.append({"timeframe": t, "wound_area_px": best_area})
    return pd.DataFrame(rows)


def get_wound_mask_for_frame(frame, min_area):
    """Return binary wound mask (uint8) or None."""
    black = (frame == 0).astype(np.uint8)
    labeled = measure.label(black, connectivity=2)
    h, w = frame.shape
    border_labels = (
        set(labeled[0, :]) | set(labeled[-1, :])
        | set(labeled[:, 0]) | set(labeled[:, -1])
    )
    border_labels.discard(0)
    best_label, best_area = None, 0
    for region in measure.regionprops(labeled):
        if region.label in border_labels:
            continue
        if region.area > best_area:
            best_area, best_label = region.area, region.label
    if best_label is None or best_area < min_area:
        return None
    return (labeled == best_label).astype(np.uint8)


def find_cell_layers(frame, wound_mask):
    """
    BFS outward from the wound boundary through the cell adjacency graph.
    Returns {cell_label: layer_number}. Layer 1 = cells touching the wound.
    """
    assigned = {}
    # Layer 1: any cell label whose pixels touch the dilated wound
    dilated_wound = binary_dilation(wound_mask, iterations=1)
    layer1 = set(np.unique(frame[dilated_wound])) - {0}
    for lbl in layer1:
        assigned[lbl] = 1

    layer_num = 2
    while True:
        front_mask = np.isin(frame, list(assigned.keys())).astype(np.uint8)
        dilated_front = binary_dilation(front_mask, iterations=1)
        new_cells = set(np.unique(frame[dilated_front])) - {0} - set(assigned.keys())
        if not new_cells:
            break
        for lbl in new_cells:
            assigned[lbl] = layer_num
        layer_num += 1

    return assigned


def compute_cell_layers(stack, min_wound_area=200, progress=None):
    """Compute cell layer assignments for every frame."""
    import pandas as pd
    T = stack.shape[0]
    rows = []
    layer_cache = {}

    for t in range(T):
        frame = stack[t]
        wound_mask = get_wound_mask_for_frame(frame, min_wound_area)
        if wound_mask is None:
            continue
        layers = find_cell_layers(frame, wound_mask)
        layer_cache[t] = layers
        for cell_label, layer in layers.items():
            rows.append({
                "timeframe": t,
                "cell_label": int(cell_label),
                "layer": layer,
            })
        if progress:
            progress((t + 1) / T, f"Layers frame {t + 1}/{T}")

    df = pd.DataFrame(rows, columns=["timeframe", "cell_label", "layer"])
    return df, layer_cache


def sample_boundary_intensity(stack_raw, boundary_df, ring_px=5):
    """
    For each seed vertex in each frame, sample the mean intensity in a
    small ring around that vertex from the raw image stack.
    """
    import pandas as pd
    rows = []
    H, W = stack_raw.shape[1], stack_raw.shape[2]
    for _, row in boundary_df.iterrows():
        t = int(row["timeframe"])
        x, y = int(round(row["x"])), int(round(row["y"]))
        r = ring_px
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        patch = stack_raw[t, y0:y1, x0:x1].astype(np.float64)
        rows.append({
            "timeframe": t,
            "point_id": int(row["point_id"]),
            "intensity": round(float(patch.mean()), 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run all and save CSVs
# ---------------------------------------------------------------------------

def run(stack_masks, stack_raw, params, progress=None):
    """
    Run the full boundary analysis pipeline.

    Parameters
    ----------
    stack_masks : np.ndarray  — (T, H, W) label masks (cleaned or tracked)
    stack_raw   : np.ndarray  — (T, H, W) raw intensity images
    params      : object with .out_dir, .boundary_n_points, .boundary_min_wound_area,
                  .dt_s (frame interval in seconds)

    Returns dict of results paths + DataFrames.
    """
    import pandas as pd
    os.makedirs(params.out_dir, exist_ok=True)

    n_points = getattr(params, "boundary_n_points", 10)
    min_wound = getattr(params, "boundary_min_wound_area", 200)
    dt = params.dt_s
    ring_px = getattr(params, "boundary_ring_px", 5)

    results = {}

    # 1. Boundary points
    bdf, fstats = process_all_frames(stack_masks, n_points, min_wound, progress)
    bp_path = os.path.join(params.out_dir, "cluster_boundary_points.csv")
    bdf.to_csv(bp_path, index=False)
    results["boundary_df"] = bdf
    results["boundary_path"] = bp_path

    # 2. Velocity
    px_um = float(getattr(params, "px_size_um", 1.0) or 1.0)
    vel_df = compute_velocity(bdf, dt, px_um)
    vel_path = os.path.join(params.out_dir, "cluster_boundary_velocity.csv")
    vel_df.to_csv(vel_path, index=False)
    results["vel_df"] = vel_df
    results["vel_path"] = vel_path

    # 3. Wound area
    wa_df = compute_wound_area(stack_masks, min_wound)
    wa_path = os.path.join(params.out_dir, "wound_area.csv")
    wa_df.to_csv(wa_path, index=False)
    results["wa_df"] = wa_df
    results["wa_path"] = wa_path

    # 4. Cell layers
    layer_df, layer_cache = compute_cell_layers(stack_masks, min_wound)
    layer_path = os.path.join(params.out_dir, "cell_layers.csv")
    layer_df.to_csv(layer_path, index=False)
    results["layer_df"] = layer_df
    results["layer_path"] = layer_path
    results["layer_cache"] = layer_cache

    # 5. Boundary intensity (from raw images)
    int_df = sample_boundary_intensity(stack_raw, bdf, ring_px)
    int_path = os.path.join(params.out_dir, "boundary_intensity.csv")
    int_df.to_csv(int_path, index=False)
    results["int_df"] = int_df
    results["int_path"] = int_path

    # 6. Plots
    results["wound_area_plot"] = _plot_wound_area(wa_df, dt, params.out_dir)
    results["velocity_heatmap"] = _plot_velocity_heatmap(vel_df, n_points,
                                                          dt, params.out_dir)
    results["speed_intensity_plot"] = _plot_speed_vs_intensity(
        vel_df, int_df, dt, params.out_dir)

    return results


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _plot_wound_area(wa_df, dt, out_dir):
    """Wound area over time."""
    if wa_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    t = wa_df["timeframe"].values * dt
    a = wa_df["wound_area_px"].values
    ax.plot(t, a, "o-", color="steelblue", linewidth=2, markersize=5)
    ax.fill_between(t, 0, a, alpha=0.15, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wound area (px)")
    ax.set_title("Wound Area Over Time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "wound_area_plot.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_velocity_heatmap(vel_df, n_points, dt, out_dir):
    """Per-point velocity (speed) heatmap over time."""
    if vel_df.empty:
        return None
    import pandas as pd
    pivot = vel_df.pivot_table(index="point_id", columns="timeframe_from",
                                values="speed", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="plasma",
                   origin="lower",
                   extent=[pivot.columns.min() * dt,
                           pivot.columns.max() * dt,
                           0, pivot.shape[0]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Boundary point ID")
    ax.set_title("Velocity Heatmap (px/s per seed vertex)")
    fig.colorbar(im, ax=ax, label="Speed (px/s)")
    # label each row with its boundary point id (matches the numbers on the
    # arrows in the velocity vector overlay)
    ax.set_yticks([i + 0.5 for i in range(pivot.shape[0])])
    ax.set_yticklabels([str(int(p)) for p in pivot.index])
    fig.tight_layout()
    path = os.path.join(out_dir, "velocity_heatmap.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_speed_vs_intensity(vel_df, int_df, dt, out_dir):
    """Normalised migration speed vs fluorescence intensity over time."""
    if vel_df.empty or int_df.empty:
        return None
    import pandas as pd
    spd_stats = (vel_df.groupby("timeframe_from")["speed"]
                 .agg(mean="mean", std="std").reset_index()
                 .rename(columns={"timeframe_from": "timeframe"}))
    int_stats = (int_df.groupby("timeframe")["intensity"]
                 .agg(mean="mean", std="std").reset_index())

    common = sorted(set(spd_stats["timeframe"]) & set(int_stats["timeframe"]))
    if len(common) < 2:
        return None

    spd = spd_stats.set_index("timeframe").loc[common]
    igt = int_stats.set_index("timeframe").loc[common]

    def norm01(s):
        rng = s.max() - s.min()
        return (s - s.min()) / (rng if rng > 0 else 1)

    t_axis = np.array(common) * dt
    spd_n = norm01(spd["mean"].values)
    spd_sn = spd["std"].values / (spd["mean"].max() - spd["mean"].min() + 1e-12)
    int_n = norm01(igt["mean"].values)
    int_sn = igt["std"].values / (igt["mean"].max() - igt["mean"].min() + 1e-12)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(t_axis, spd_n - spd_sn, spd_n + spd_sn,
                    alpha=0.25, color="green")
    ax.plot(t_axis, spd_n, "o-", color="green", lw=1.5, ms=4,
            label="Migration speed")
    ax.fill_between(t_axis, int_n - int_sn, int_n + int_sn,
                    alpha=0.25, color="mediumpurple")
    ax.plot(t_axis, int_n, "o-", color="mediumpurple", lw=1.5, ms=4,
            label="Fluorescence intensity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised value")
    ax.set_title("Migration Speed vs. Fluorescence Intensity")
    ax.legend()
    ax.set_ylim(-0.1, 1.3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "speed_vs_intensity.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Streamlit-specific renderers (return image arrays or figure bytes)
# ---------------------------------------------------------------------------

def render_boundary_frame(mask_frame, raw_frame, boundary_df, frame_idx,
                          n_points):
    """
    Render a single frame with wound contour overlay + seed vertices.
    Returns a uint8 RGB image.
    """
    from wound_analysis import detection
    base = detection.normalize(raw_frame) if raw_frame is not None else \
        (mask_frame.astype(np.float32) / max(mask_frame.max(), 1))
    rgb = np.dstack([base, base, base]).copy()
    rgb = (rgb * 255).astype(np.uint8)

    # Draw wound contour from the mask
    wound_mask = get_wound_mask_for_frame(mask_frame, 1)
    if wound_mask is not None:
        contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, (255, 0, 0), 1)

    # Draw seed vertices
    sub = boundary_df[boundary_df["timeframe"] == frame_idx]
    if not sub.empty:
        pts = sub[["x", "y"]].values
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            cv2.circle(rgb, (xi, yi), 3, (255, 80, 80), -1)

    return rgb


def render_velocity_frame(mask_frame, raw_frame, boundary_df, vel_df,
                          frame_idx, arrow_scale=8):
    """
    Render a frame with velocity quiver arrows coloured by speed.
    Returns a uint8 RGB image.
    """
    from wound_analysis import detection
    base = detection.normalize(raw_frame) if raw_frame is not None else \
        (mask_frame.astype(np.float32) / max(mask_frame.max(), 1))

    H, W = mask_frame.shape
    fig, ax = plt.subplots(figsize=(W / 80, H / 80), dpi=80)
    ax.imshow(base, cmap="gray", interpolation="nearest")

    sub_vel = vel_df[vel_df["timeframe_from"] == frame_idx]
    sub_pts = boundary_df[boundary_df["timeframe"] == frame_idx]
    if not sub_vel.empty and not sub_pts.empty:
        merged = sub_pts.set_index("point_id").join(
            sub_vel.set_index("point_id"), how="inner")
        if not merged.empty:
            # colour arrows/markers by SPEED (plasma); keep the point-id number
            # labels so each boundary point is still identifiable across frames.
            xs = merged["x"].values
            ys = merged["y"].values
            dxs = merged["dx"].values
            dys = merged["dy"].values
            speeds = merged["speed"].values
            speed_max = max(float(vel_df["speed"].quantile(0.95)), 0.01)
            norm = mcolors.Normalize(vmin=0, vmax=speed_max)
            cmap_obj = plt.cm.plasma
            colors = cmap_obj(norm(speeds))
            ax.quiver(xs, ys, dxs * arrow_scale, dys * arrow_scale,
                      color=colors, scale=1, scale_units="xy", angles="xy",
                      width=0.004, headwidth=4, headlength=5, zorder=6)
            ax.scatter(xs, ys, c=speeds, cmap=cmap_obj, norm=norm, s=22,
                       zorder=7, linewidths=0)
            for pid, row in merged.iterrows():
                ax.text(row["x"], row["y"], str(int(pid)), color="white",
                        fontsize=6, fontweight="bold", ha="center",
                        va="center", zorder=8)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.tight_layout(pad=0)

    # Convert figure to image
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_layer_frame(mask_frame, layer_cache, frame_idx):
    """
    Render a frame with cells coloured by BFS layer number.
    Returns a uint8 RGB image.
    """
    layers = layer_cache.get(frame_idx, {})
    if not layers:
        H, W = mask_frame.shape
        return np.zeros((H, W, 3), dtype=np.uint8)

    max_layer = max(layers.values())
    norm = mcolors.Normalize(vmin=1, vmax=max(max_layer, 2))
    cmap_obj = plt.cm.viridis

    H, W = mask_frame.shape
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for cell_label, layer in layers.items():
        canvas[mask_frame == cell_label] = cmap_obj(norm(layer))[:3]
    return (canvas * 255).astype(np.uint8)
