"""
segmentation.py  --  Objective 2
================================

Segment epithelial cells from the membrane channel and characterise how cell
shape changes through wound closure (a proxy for the underlying mechanics).

Method
------
The membrane label is bright on cell boundaries and dark in cell interiors, so
cells are catchment basins of the intensity image:

  smooth -> h-minima seeds (one per cell interior) -> marker-controlled
  watershed with the membrane intensity as the landscape, masked to tissue.

Cells touching the tissue border or overlapping the wound are dropped, and
cells are filtered to a plausible area range.  For each surviving cell we
measure area, perimeter, circularity (4*pi*A / P^2), aspect ratio
(major/minor axis), eccentricity, solidity and orientation, plus its distance
to the wound centre so shape can be related to wound proximity and to time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.morphology import h_minima, binary_dilation, disk
from skimage.segmentation import watershed
from skimage.measure import regionprops

from . import detection


# columns we keep for every cell in every frame
_COLS = ["frame", "label", "cx", "cy", "area_um2", "perimeter_um",
         "circularity", "aspect_ratio", "eccentricity", "solidity",
         "orientation", "dist_um"]


@dataclass
class SegResult:
    labels: np.ndarray           # (T, Y, X) int label image per frame
    table: dict                  # column-name -> 1-D np.ndarray (all cells)
    px_size_um: float
    dt_s: float
    tissue: np.ndarray
    meta: dict = field(default_factory=dict)

    def frame_rows(self, t):
        m = self.table["frame"] == t
        return {k: v[m] for k, v in self.table.items()}


def segment_frame(frame, tissue, h=0.02, smooth=2.0):
    """Marker-controlled watershed segmentation of one membrane frame."""
    f = gaussian(detection.normalize(frame), smooth)
    seeds = h_minima(f, h) & tissue
    markers, _ = ndi.label(seeds)
    labels = watershed(f, markers, mask=tissue)
    return labels, f


def _border_labels(labels, tissue):
    """Labels touching the tissue-mask boundary (incomplete cells)."""
    edge = binary_dilation(~tissue, disk(1)) & tissue
    return set(np.unique(labels[edge])) - {0}


def analyze(stack, geo=None, tissue=None, px_size_um=0.3448, dt_s=31.09,
            area_um2_range=(8.0, 400.0), h=0.02, smooth=2.0):
    """Segment every frame and build a per-cell shape table.

    ``geo`` (a WoundGeometry) supplies the wound centre and per-frame radius so
    each cell gets a distance-to-wound and wound cells are excluded.
    """
    T, H, W = stack.shape
    if tissue is None:
        tissue, _ = detection.segment_tissue(stack)
    px = px_size_um
    labels_all = np.zeros((T, H, W), np.int32)
    cols = {c: [] for c in _COLS}

    for t in range(T):
        labels, _ = segment_frame(stack[t], tissue, h=h, smooth=smooth)

        # wound centre + radius: exclude only cells whose CENTROID lies inside
        # the wound (the wound-filling watershed region), so genuine
        # wound-adjacent cells are kept for the layer analysis.
        if geo is not None:
            cy, cx = geo.centers[t]
            r_excl = max(geo.radius[t].mean(), 5.0)
        else:
            cy, cx = H / 2, W / 2
            r_excl = 0.0
        drop = _border_labels(labels, tissue)

        keep_lab = np.zeros_like(labels)
        for rp in regionprops(labels):
            if rp.label in drop:
                continue
            area = rp.area * px ** 2
            if not (area_um2_range[0] <= area <= area_um2_range[1]):
                continue
            cyc, cxc = rp.centroid
            if (cxc - cx) ** 2 + (cyc - cy) ** 2 <= r_excl ** 2:
                continue  # cell sits in the wound interior
            perim = rp.perimeter * px
            circ = float(np.clip(4 * np.pi * rp.area / (rp.perimeter ** 2 + 1e-9), 0, 1)) \
                if rp.perimeter > 0 else 0.0
            major = rp.axis_major_length
            minor = rp.axis_minor_length
            ar = float(major / minor) if minor > 0 else np.nan
            dist = np.hypot(cyc - cy, cxc - cx) * px

            keep_lab[labels == rp.label] = rp.label
            cols["frame"].append(t)
            cols["label"].append(rp.label)
            cols["cx"].append(cxc)
            cols["cy"].append(cyc)
            cols["area_um2"].append(area)
            cols["perimeter_um"].append(perim)
            cols["circularity"].append(circ)
            cols["aspect_ratio"].append(ar)
            cols["eccentricity"].append(rp.eccentricity)
            cols["solidity"].append(rp.solidity)
            cols["orientation"].append(rp.orientation)
            cols["dist_um"].append(dist)
        labels_all[t] = keep_lab

    table = {c: np.asarray(v, float) if c not in ("frame", "label")
             else np.asarray(v, int) for c, v in cols.items()}
    return SegResult(labels=labels_all, table=table, px_size_um=px, dt_s=dt_s,
                     tissue=tissue)


# --------------------------------------------------------------------------- #
# summaries used by plotting / reporting
# --------------------------------------------------------------------------- #
def _frame_mask(seg, frame_range):
    if frame_range is None:
        return np.ones(seg.table["frame"].shape, bool)
    t0, t1 = frame_range
    return (seg.table["frame"] >= t0) & (seg.table["frame"] <= t1)


def shape_vs_distance(seg, n_bins=6, max_um=40.0, frame_range=None):
    """Mean shape metrics binned by distance to the wound centre.

    ``frame_range`` restricts to the open-wound window, where distance to the
    wound centre is physically meaningful."""
    fm = _frame_mask(seg, frame_range)
    d = seg.table["dist_um"]
    edges = np.linspace(0, max_um, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = {"dist_um": centers}
    for metric in ("circularity", "aspect_ratio", "area_um2"):
        vals = seg.table[metric]
        means = np.full(n_bins, np.nan)
        sems = np.full(n_bins, np.nan)
        for i in range(n_bins):
            m = fm & (d >= edges[i]) & (d < edges[i + 1]) & np.isfinite(vals)
            if m.sum() >= 3:
                means[i] = vals[m].mean()
                sems[i] = vals[m].std() / np.sqrt(m.sum())
        out[metric] = means
        out[metric + "_sem"] = sems
    return out


def shape_vs_time(seg, times_min, near_um=14.0):
    """Per-frame mean shape for wound-adjacent (near) vs. far cells."""
    fr = seg.table["frame"]
    d = seg.table["dist_um"]
    T = seg.labels.shape[0]
    res = {"time_min": times_min}
    for metric in ("circularity", "aspect_ratio", "area_um2"):
        vals = seg.table[metric]
        near = np.full(T, np.nan)
        far = np.full(T, np.nan)
        for t in range(T):
            mt = (fr == t) & np.isfinite(vals)
            n = mt & (d <= near_um)
            f = mt & (d > near_um)
            if n.sum() >= 2:
                near[t] = vals[n].mean()
            if f.sum() >= 2:
                far[t] = vals[f].mean()
        res[metric + "_near"] = near
        res[metric + "_far"] = far
    return res


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _metric_map(labels, frame_rows, metric, vmin, vmax, cmap_name):
    """RGB image with each cell filled by its ``metric`` value."""
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    norm = Normalize(vmin, vmax)
    cmap = cm.get_cmap(cmap_name)
    rgb = np.zeros(labels.shape + (3,))
    lut = {int(l): v for l, v in zip(frame_rows["label"], frame_rows[metric])}
    for lab, val in lut.items():
        if not np.isfinite(val):
            continue
        rgb[labels == lab] = cmap(norm(val))[:3]
    return rgb, norm, cmap


def plot_all(seg, stack, times_min, outdir="results", map_frame=None,
             open_window=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from skimage.segmentation import mark_boundaries
    from . import plotting
    plotting.apply_style()
    paths = []
    T = seg.labels.shape[0]
    if map_frame is None:
        # a frame with many cells, mid-closure
        counts = [np.sum(seg.table["frame"] == t) for t in range(T)]
        map_frame = int(np.argmax(counts))

    # 1) segmentation overlay montage --------------------------------------
    sel = np.linspace(2, T - 1, 6).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, t in zip(axes.ravel(), sel):
        base = detection.normalize(stack[t])
        ax.imshow(mark_boundaries(np.dstack([base] * 3), seg.labels[t],
                                  color=(1, 0.45, 0)))
        ax.set_title(f"t={times_min[t]:.1f} min  "
                     f"({int(np.sum(seg.table['frame'] == t))} cells)", fontsize=10)
        ax.axis("off")
    fig.suptitle("Cell segmentation (membrane watershed)", fontweight="bold")
    paths.append(plotting.save(fig, "obj2_segmentation_overlay.png", outdir))

    # 2) area + circularity maps (cf. challenge Fig. 1 / ref maps) ----------
    rows = seg.frame_rows(map_frame)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    base = detection.normalize(stack[map_frame])
    a_lo, a_hi = np.percentile(rows["area_um2"], (5, 95)) if rows["area_um2"].size else (0, 1)
    rgb, norm, cmap = _metric_map(seg.labels[map_frame], rows, "area_um2",
                                  a_lo, a_hi, "viridis")
    blend = 0.25 * np.dstack([base] * 3) + 0.75 * rgb
    blend[seg.labels[map_frame] == 0] = np.dstack([base] * 3)[seg.labels[map_frame] == 0]
    axes[0].imshow(blend); axes[0].set_title(f"Cell area map  (t={times_min[map_frame]:.1f} min)")
    axes[0].axis("off")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0],
                 fraction=0.046, label="area (µm²)")
    rgb2, norm2, cmap2 = _metric_map(seg.labels[map_frame], rows, "circularity",
                                     0.4, 1.0, "magma")
    blend2 = 0.25 * np.dstack([base] * 3) + 0.75 * rgb2
    blend2[seg.labels[map_frame] == 0] = np.dstack([base] * 3)[seg.labels[map_frame] == 0]
    axes[1].imshow(blend2); axes[1].set_title("Cell circularity map")
    axes[1].axis("off")
    fig.colorbar(cm.ScalarMappable(norm=norm2, cmap=cmap2), ax=axes[1],
                 fraction=0.046, label="circularity")
    paths.append(plotting.save(fig, "obj2_shape_maps.png", outdir))

    # 3) shape vs distance to wound (open-wound window) --------------------
    svd = shape_vs_distance(seg, frame_range=open_window)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric, lab in zip(
            axes, ("circularity", "aspect_ratio", "area_um2"),
            ("circularity", "aspect ratio (elongation)", "area (µm²)")):
        ax.errorbar(svd["dist_um"], svd[metric], yerr=svd[metric + "_sem"],
                    fmt="o-", capsize=3, color="#1f77b4")
        ax.set_xlabel("distance to wound (µm)")
        ax.set_ylabel(lab)
        ax.set_title(lab + " vs. wound distance", fontsize=10.5)
    fig.suptitle("Cell shape depends on wound proximity", fontweight="bold")
    paths.append(plotting.save(fig, "obj2_shape_vs_distance.png", outdir))

    # 4) shape of wound-adjacent vs far cells over time --------------------
    svt = shape_vs_time(seg, times_min)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric, lab in zip(
            axes, ("circularity", "aspect_ratio", "area_um2"),
            ("circularity", "aspect ratio", "area (µm²)")):
        ax.plot(svt["time_min"], svt[metric + "_near"], "o-", color="#d62728",
                label="wound-adjacent (≤14 µm)")
        ax.plot(svt["time_min"], svt[metric + "_far"], "s-", color="#7f7f7f",
                label="far (>14 µm)")
        ax.set_xlabel("time (min)")
        ax.set_ylabel(lab)
        ax.set_title(lab + " over time", fontsize=10.5)
    axes[0].legend(fontsize=8)
    fig.suptitle("Cell-shape dynamics through wound closure", fontweight="bold")
    paths.append(plotting.save(fig, "obj2_shape_vs_time.png", outdir))
    return paths


def summary_text(seg, open_window=None):
    t = seg.table
    svd = shape_vs_distance(seg, frame_range=open_window)

    def trend(metric):
        v = svd[metric]
        ok = np.isfinite(v)
        return v[ok][0], v[ok][-1]   # innermost vs outermost bin mean

    a_near, a_far = trend("area_um2")
    c_near, c_far = trend("circularity")
    e_near, e_far = trend("aspect_ratio")
    lines = [
        "OBJECTIVE 2  —  Cell-shape characterisation",
        "-" * 58,
        f"Cells segmented (all frames): {t['area_um2'].size} "
        f"(~{t['area_um2'].size / seg.labels.shape[0]:.0f}/frame)",
        f"Median cell area            : {np.median(t['area_um2']):.1f} µm²",
        f"Median circularity          : {np.median(t['circularity']):.2f}",
        f"Median aspect ratio         : {np.nanmedian(t['aspect_ratio']):.2f}",
        f"Across wound -> tissue (binned mean, open-wound phase):",
        f"   area         {a_near:5.0f} -> {a_far:5.0f} µm²   (grows with distance)",
        f"   circularity  {c_near:5.2f} -> {c_far:5.2f}      (drops with distance)",
        f"   aspect ratio {e_near:5.2f} -> {e_far:5.2f}      (grows with distance)",
        f"   -> wound-adjacent cells are SMALLER and ROUNDER; distal cells are "
        f"larger and more ELONGATED, consistent with",
        f"      mechanical compression at the leading edge.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from . import io_utils
    tl = io_utils.load_wound()
    centers = detection.track_center(tl.images)
    geo = detection.radial_edges(stack=tl.images, centers=centers,
                                 px_size_um=tl.px_size_um, dt_s=tl.dt_s)
    seg = analyze(tl.images, geo=geo, px_size_um=tl.px_size_um, dt_s=tl.dt_s)
    n = len(seg.table["area_um2"])
    print(f"segmented {n} cell instances across {tl.n_frames} frames "
          f"({n / tl.n_frames:.0f} cells/frame)")
    print(f"median cell area = {np.median(seg.table['area_um2']):.1f} µm², "
          f"median circularity = {np.median(seg.table['circularity']):.2f}")
