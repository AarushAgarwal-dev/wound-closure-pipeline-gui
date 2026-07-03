"""
intensity.py  --  per-cell fluorescence intensity over time
===========================================================

Adapted from the team's ``Cell_Intensity_Tracking`` notebook (Author: Linlin Li).
Uses the tracked masks + raw images to measure, per tracked cell per frame:
mean / median / std / total / max / min intensity, local-background-corrected
mean, and fold-change normalised to a reference frame.

Outputs ``cell_intensity_per_frame.csv`` (long), ``cell_intensity_summary.csv``,
plus population / trajectory / heatmap / spatial-map figures.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def measure_cell_intensity(image, cell_bool, bg_ring_px=10, min_px=20):
    px = image[cell_bool]
    if px.size < min_px:
        return None
    rows, cols = np.where(cell_bool)
    r0, r1 = max(rows.min() - bg_ring_px, 0), min(rows.max() + bg_ring_px + 1, image.shape[0])
    c0, c1 = max(cols.min() - bg_ring_px, 0), min(cols.max() + bg_ring_px + 1, image.shape[1])
    bg = image[r0:r1, c0:c1][~cell_bool[r0:r1, c0:c1]]
    background = float(np.median(bg)) if bg.size else 0.0
    return dict(mean_intensity=float(px.mean()), median_intensity=float(np.median(px)),
                std_intensity=float(px.std()), total_intensity=float(px.sum()),
                max_intensity=float(px.max()), min_intensity=float(px.min()),
                cell_area_px=int(px.size), background=background,
                mean_bg_corrected=float(px.mean()) - background)


def run(stack, tracked, params, frame_idx=None, progress=None):
    T = tracked.shape[0]
    if frame_idx is None:
        frame_idx = list(range(T))
    dt_min = params.dt_s / 60.0
    rows = []
    for t in range(T):
        img = stack[t].astype(np.float32)
        for lab in np.unique(tracked[t]):
            if lab == 0:
                continue
            m = measure_cell_intensity(img, tracked[t] == lab, params.bg_ring_px)
            if m is None:
                continue
            m["track_id"] = int(lab); m["frame"] = int(frame_idx[t])
            m["t_index"] = t; m["time_min"] = round(t * dt_min, 4)
            rows.append(m)
        if progress:
            progress((t + 1) / T, f"Intensity frame {t + 1}/{T}")

    # fold-change normalised to reference frame
    if params.norm_ref_frame is not None:
        ref = {r["track_id"]: r["mean_intensity"] for r in rows
               if r["t_index"] == params.norm_ref_frame}
        for r in rows:
            base = ref.get(r["track_id"])
            r["mean_normalised"] = (r["mean_intensity"] / base) if base else np.nan
    else:
        for r in rows:
            r["mean_normalised"] = np.nan

    cols = ["frame", "time_min", "track_id", "mean_intensity", "median_intensity",
            "std_intensity", "total_intensity", "max_intensity", "min_intensity",
            "cell_area_px", "background", "mean_bg_corrected", "mean_normalised"]
    os.makedirs(params.out_dir, exist_ok=True)
    path = os.path.join(params.out_dir, "cell_intensity_per_frame.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # per-cell summary
    by_id = {}
    for r in rows:
        by_id.setdefault(r["track_id"], []).append(r["mean_intensity"])
    spath = os.path.join(params.out_dir, "cell_intensity_summary.csv")
    with open(spath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track_id", "mean_allframes", "std_allframes", "n_frames", "cv"])
        for tid, v in by_id.items():
            v = np.array(v); mu = v.mean(); sd = v.std()
            w.writerow([tid, round(mu, 3), round(sd, 3), v.size,
                        round(sd / mu, 3) if mu else 0])

    fig = _plots(rows, stack, tracked, frame_idx, params)
    return rows, path, spath, fig


def _plots(rows, stack, tracked, frame_idx, params):
    """Population dynamics, per-cell trajectories, heatmap, spatial map."""
    if not rows:
        return None
    T = tracked.shape[0]
    dt_min = params.dt_s / 60.0
    times = np.arange(T) * dt_min
    ids = sorted({r["track_id"] for r in rows})
    # matrix cells x frames of mean_intensity
    idx = {tid: i for i, tid in enumerate(ids)}
    M = np.full((len(ids), T), np.nan)
    for r in rows:
        M[idx[r["track_id"]], r["t_index"]] = r["mean_intensity"]
    present = np.sum(~np.isnan(M), axis=1)
    stable = present >= params.intensity_min_frames

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    # (a) population mean ± SD
    ax = axes[0, 0]
    mu = np.nanmean(M, 0); sd = np.nanstd(M, 0); med = np.nanmedian(M, 0)
    ax.fill_between(times, mu - sd, mu + sd, alpha=0.2, color="steelblue")
    ax.plot(times, mu, color="steelblue", lw=2, label="mean ± SD")
    ax.plot(times, med, "--", color="tomato", lw=1.5, label="median")
    ax.set_title("Population intensity dynamics"); ax.set_xlabel("time (min)")
    ax.set_ylabel("mean intensity (AU)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (b) per-cell trajectories (stable cells)
    ax = axes[0, 1]
    for i in np.where(stable)[0]:
        ax.plot(times, M[i], lw=0.5, alpha=0.4)
    ax.plot(times, mu, color="black", lw=2.5, label="population mean")
    ax.set_title(f"Per-cell trajectories (n={int(stable.sum())} stable)")
    ax.set_xlabel("time (min)"); ax.set_ylabel("mean intensity (AU)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (c) heatmap cells x frames
    ax = axes[1, 0]
    Ms = M[stable]
    if Ms.size:
        order = np.argsort(np.nanmean(Ms, 1))
        im = ax.imshow(Ms[order], aspect="auto", cmap="inferno", origin="lower",
                       extent=[times[0], times[-1], 0, Ms.shape[0]],
                       vmin=np.nanpercentile(Ms, 2), vmax=np.nanpercentile(Ms, 98))
        fig.colorbar(im, ax=ax, label="mean intensity (AU)")
    ax.set_title("Intensity heatmap (cell × time)")
    ax.set_xlabel("time (min)"); ax.set_ylabel("cell (sorted)")
    # (d) spatial map at reference frame
    ax = axes[1, 1]
    f0 = params.norm_ref_frame if params.norm_ref_frame is not None else 0
    f0 = min(f0, T - 1)
    vals = {r["track_id"]: r["mean_intensity"] for r in rows if r["t_index"] == f0}
    if vals:
        vv = np.array(list(vals.values()))
        norm = mcolors.Normalize(*np.percentile(vv, [2, 98]))
        cmo = plt.get_cmap("inferno")
        canvas = np.zeros((*tracked[f0].shape, 3), np.float32)
        for tid, v in vals.items():
            canvas[tracked[f0] == tid] = cmo(norm(v))[:3]
        ax.imshow(canvas)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmo), ax=ax, label="mean intensity (AU)")
    ax.set_title(f"Spatial intensity map (t={f0 * dt_min:.1f} min)"); ax.axis("off")
    fig.tight_layout()
    path = os.path.join(params.out_dir, "intensity_plots.png")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path
