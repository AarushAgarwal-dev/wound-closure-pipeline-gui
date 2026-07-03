"""
intercalation.py  --  Objective 3
=================================

Count the cells surrounding the wound by neighbour layer over time, and detect
intercalation (T1 / neighbour-exchange) events to ask whether intercalation and
farther-out cells contribute to closure.

Pipeline
--------
1. Cell adjacency graph per frame, straight from the watershed labels
   (4-connected label pairs share a junction = neighbours).
2. Layer assignment by ring-growing out from the wound region:
   layer 1 = cells touching the wound, layer 2 = their neighbours, layer 3 =
   the next ring.  Counts per layer vs time.
3. Cell tracking across frames by maximum label overlap (IoU linking) -> stable
   track IDs, so neighbour sets can be compared frame to frame.
4. T1 detection: a neighbour pair (A,B) that are in contact at t but not at t+1,
   whose two shared neighbours (C,D) become newly in contact, is a T1 swap.
   Their location (distance to the wound) and time are recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import binary_dilation, disk
from skimage.draw import polygon as draw_polygon


_OFF = 100000  # label-pair hashing offset


# --------------------------------------------------------------------------- #
# adjacency
# --------------------------------------------------------------------------- #
def adjacency_edges(labels):
    """Set of unordered neighbour label pairs (l1 < l2) in a label image."""
    edges = set()
    for a, b in ((labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :], labels[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        pa, pb = a[m].astype(np.int64), b[m].astype(np.int64)
        lo = np.minimum(pa, pb)
        hi = np.maximum(pa, pb)
        for k in np.unique(lo * _OFF + hi):
            edges.add((int(k // _OFF), int(k % _OFF)))
    return edges


def neighbor_dict(edges):
    nb = {}
    for a, b in edges:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    return nb


# --------------------------------------------------------------------------- #
# wound mask + layer assignment
# --------------------------------------------------------------------------- #
def wound_region(geo, t, shape):
    """Filled wound polygon (bool mask) for frame ``t``."""
    xy = geo.edge_xy(t)
    rr, cc = draw_polygon(xy[:, 1], xy[:, 0], shape=shape)
    m = np.zeros(shape, bool)
    m[rr, cc] = True
    return m


def layers(labels, wound_mask, nb, n_layers=3, touch_px=7):
    """Return dict layer-> set(labels), growing outward from the wound.

    Layer 1 = cells within ``touch_px`` of the wound boundary (about half a
    cell width, so cells straddling the irregular edge are counted); layers
    2+ grow outward through the neighbour graph.
    """
    touch = binary_dilation(wound_mask, disk(touch_px))
    layer1 = set(np.unique(labels[touch])) - {0}
    assigned = set(layer1)
    out = {1: layer1}
    frontier = layer1
    for L in range(2, n_layers + 1):
        nxt = set()
        for c in frontier:
            nxt |= nb.get(c, set())
        nxt -= assigned
        out[L] = nxt
        assigned |= nxt
        frontier = nxt
    return out


# --------------------------------------------------------------------------- #
# tracking (IoU linking) -> global track IDs
# --------------------------------------------------------------------------- #
def track_labels(labels_stack):
    """Link labels across frames by maximum overlap.  Returns list of dicts
    mapping per-frame label -> global track id."""
    T = labels_stack.shape[0]
    maps = [dict() for _ in range(T)]
    next_id = 1
    # frame 0
    for l in np.unique(labels_stack[0]):
        if l == 0:
            continue
        maps[0][int(l)] = next_id
        next_id += 1
    for t in range(1, T):
        prev, cur = labels_stack[t - 1], labels_stack[t]
        mask = (prev > 0) & (cur > 0)
        if mask.any():
            key = prev[mask].astype(np.int64) * _OFF + cur[mask].astype(np.int64)
            uk, cnt = np.unique(key, return_counts=True)
            prev_l = (uk // _OFF).astype(int)
            cur_l = (uk % _OFF).astype(int)
            # best previous label for each current label
            best = {}
            for pl, cl, c in zip(prev_l, cur_l, cnt):
                if cl not in best or c > best[cl][1]:
                    best[cl] = (pl, c)
        else:
            best = {}
        for l in np.unique(cur):
            if l == 0:
                continue
            l = int(l)
            if l in best and best[l][0] in maps[t - 1]:
                maps[t][l] = maps[t - 1][best[l][0]]
            else:
                maps[t][l] = next_id
                next_id += 1
    return maps


# --------------------------------------------------------------------------- #
# main analysis
# --------------------------------------------------------------------------- #
@dataclass
class IntercalationResult:
    times_min: np.ndarray
    layer_counts: dict          # layer -> (T,) cell counts
    adjacent_count: np.ndarray  # (T,) cells touching the wound (layer 1)
    t1_times_min: np.ndarray    # time of each detected T1 event
    t1_dist_um: np.ndarray      # wound distance of each T1 event
    t1_cumulative: np.ndarray   # (T,) cumulative T1 count
    n_t1: int
    px_size_um: float
    frac_during_closure: float  # fraction of T1 events in the closure window
    closure_end_min: float


def analyze(seg, geo, times_min, n_layers=3, max_layer_for_t1=3,
            closure_window=None):
    """Compute layer counts and detect intercalation events.

    ``closure_window`` = (t_start, t_end) frame indices of the open-wound phase
    (e.g. from edge_velocity); used to test whether intercalation concentrates
    during active closure.  If None, the first 45 % of frames is used.
    """
    labels = seg.labels
    T, H, W = labels.shape
    px = seg.px_size_um

    # per-frame adjacency, neighbour dicts, layers, centroids
    edges_t = []
    nb_t = []
    layer_sets = []
    centroid = []   # dict label-> (cy,cx)
    for t in range(T):
        e = adjacency_edges(labels[t])
        nb = neighbor_dict(e)
        edges_t.append(e)
        nb_t.append(nb)
        wm = wound_region(geo, t, (H, W))
        layer_sets.append(layers(labels[t], wm, nb, n_layers))
        cen = {}
        for l in np.unique(labels[t]):
            if l == 0:
                continue
            ys, xs = np.where(labels[t] == l)
            cen[int(l)] = (ys.mean(), xs.mean())
        centroid.append(cen)

    layer_counts = {L: np.array([len(layer_sets[t].get(L, ())) for t in range(T)])
                    for L in range(1, n_layers + 1)}
    adjacent = layer_counts[1].copy()

    # tracking + neighbour sets in track-id space
    maps = track_labels(labels)

    def near_wound_labels(t):
        s = set()
        for L in range(1, max_layer_for_t1 + 1):
            s |= layer_sets[t].get(L, set())
        return s

    # T1 detection between consecutive frames
    t1_times = []
    t1_dist = []
    for t in range(T - 1):
        m0, m1 = maps[t], maps[t + 1]
        focus = near_wound_labels(t)
        # neighbour sets in track space at t (restricted to focus cells)
        nb0 = {}
        for a, b in edges_t[t]:
            if a in focus or b in focus:
                ta, tb = m0.get(a), m0.get(b)
                if ta and tb:
                    nb0.setdefault(ta, set()).add(tb)
                    nb0.setdefault(tb, set()).add(ta)
        edges1 = set()
        for a, b in edges_t[t + 1]:
            ta, tb = m1.get(a), m1.get(b)
            if ta and tb:
                edges1.add((min(ta, tb), max(ta, tb)))
        edges0 = set()
        for a, nbs in nb0.items():
            for b in nbs:
                edges0.add((min(a, b), max(a, b)))
        lost = edges0 - edges1
        gained = edges1 - edges0
        cen0 = {m0[l]: c for l, c in centroid[t].items() if l in m0}
        wc = geo.centers[t]
        for (A, B) in lost:
            common = nb0.get(A, set()) & nb0.get(B, set())
            if len(common) < 2:
                continue
            common = list(common)
            hit = False
            for i in range(len(common)):
                for j in range(i + 1, len(common)):
                    C, D = common[i], common[j]
                    if (min(C, D), max(C, D)) in gained:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                # location = midpoint of A,B centroids
                if A in cen0 and B in cen0:
                    cy = 0.5 * (cen0[A][0] + cen0[B][0])
                    cx = 0.5 * (cen0[A][1] + cen0[B][1])
                    d = np.hypot(cy - wc[0], cx - wc[1]) * px
                    t1_times.append(times_min[t])
                    t1_dist.append(d)

    t1_times = np.asarray(t1_times)
    t1_dist = np.asarray(t1_dist)
    cumulative = np.array([np.sum(t1_times <= times_min[t]) for t in range(T)])

    if closure_window is None:
        closure_end_min = times_min[int(0.45 * (T - 1))]
    else:
        closure_end_min = times_min[min(closure_window[1], T - 1)]
    frac = float(np.mean(t1_times <= closure_end_min)) if t1_times.size else 0.0

    return IntercalationResult(
        times_min=times_min,
        layer_counts=layer_counts,
        adjacent_count=adjacent,
        t1_times_min=t1_times,
        t1_dist_um=t1_dist,
        t1_cumulative=cumulative,
        n_t1=int(t1_times.size),
        px_size_um=px,
        frac_during_closure=frac,
        closure_end_min=float(closure_end_min),
    )


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def plot_all(res, seg, geo, stack, outdir="results"):
    import matplotlib.pyplot as plt
    from . import plotting, detection
    plotting.apply_style()
    paths = []
    tmin = res.times_min

    # 1) cells per layer over time -----------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {1: "#d62728", 2: "#ff7f0e", 3: "#1f77b4"}
    for L, c in res.layer_counts.items():
        ax.plot(tmin, c, "o-", color=colors.get(L, None),
                label=f"layer {L}" + (" (wound-adjacent)" if L == 1 else ""))
    ax.set_xlabel("time (min)")
    ax.set_ylabel("number of cells")
    ax.set_title("Cells surrounding the wound, by neighbour layer")
    ax.legend()
    paths.append(plotting.save(fig, "obj3_layer_counts.png", outdir))

    # 2) wound-adjacent count + cumulative intercalations (cf. ref fig) -----
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(tmin, res.adjacent_count, "o-", color="#2ca02c",
             label="# cells adjacent to wound")
    ax1.set_xlabel("time (min)")
    ax1.set_ylabel("# wound-adjacent cells", color="#2ca02c")
    ax2 = ax1.twinx()
    ax2.plot(tmin, res.t1_cumulative, "s--", color="#9467bd",
             label="cumulative intercalations")
    ax2.set_ylabel("cumulative T1 / intercalation events", color="#9467bd")
    ax2.grid(False)
    if res.t1_times_min.size:
        for tt in res.t1_times_min:
            ax1.axvline(tt, color="#9467bd", alpha=0.15)
    ax1.set_title(f"Wound-adjacent cells & intercalation  "
                  f"(N = {res.n_t1} T1 events)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.86), fontsize=9)
    paths.append(plotting.save(fig, "obj3_adjacent_and_intercalation.png", outdir))

    # 3) intercalation locations relative to wound -------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if res.t1_dist_um.size:
        ax.hist(res.t1_dist_um, bins=np.arange(0, 45, 5), color="#9467bd",
                alpha=0.8, edgecolor="white")
    ax.set_xlabel("distance of T1 event to wound (µm)")
    ax.set_ylabel("count")
    ax.set_title("Where intercalation happens relative to the wound")
    paths.append(plotting.save(fig, "obj3_intercalation_distance.png", outdir))

    # 4) layer map snapshot -------------------------------------------------
    T = stack.shape[0]
    counts1 = res.adjacent_count
    t_show = int(np.argmax(counts1[: max(4, T // 2)]))
    nb = neighbor_dict(adjacency_edges(seg.labels[t_show]))
    wm = wound_region(geo, t_show, stack.shape[1:])
    lay = layers(seg.labels[t_show], wm, nb, 3)
    base = detection.normalize(stack[t_show])
    rgb = np.dstack([base] * 3)
    cmap = {1: [0.84, 0.15, 0.15], 2: [1.0, 0.5, 0.05], 3: [0.12, 0.47, 0.71]}
    for L, s in lay.items():
        for lab in s:
            rgb[seg.labels[t_show] == lab] = (0.45 * np.array(cmap[L])
                                              + 0.55 * rgb[seg.labels[t_show] == lab])
    rgb[wm] = [0.1, 0.1, 0.1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    cy, cx = geo.centers[t_show]
    ax.set_xlim(cx - 110, cx + 110)
    ax.set_ylim(cy + 110, cy - 110)
    ax.set_title(f"Neighbour layers at t={tmin[t_show]:.1f} min\n"
                 f"red=layer1  orange=layer2  blue=layer3")
    ax.axis("off")
    paths.append(plotting.save(fig, "obj3_layer_map.png", outdir))
    return paths


def summary_text(res):
    a0 = res.adjacent_count[res.adjacent_count > 0]
    lines = [
        "OBJECTIVE 3  —  Cell counting & intercalation",
        "-" * 58,
        f"Wound-adjacent cells (layer 1): "
        f"start {int(res.adjacent_count[2]) if len(res.adjacent_count) > 2 else 0}, "
        f"min {int(a0.min()) if a0.size else 0}, max {int(res.adjacent_count.max())}",
        f"Mean cells per layer          : "
        + ", ".join(f"L{L}={c[c>0].mean():.1f}" for L, c in res.layer_counts.items()),
        f"Intercalation (T1) events     : {res.n_t1}",
        (f"   median event distance      : {np.median(res.t1_dist_um):.1f} µm"
         if res.t1_dist_um.size else "   median event distance      : n/a"),
        f"   during closure (<= {res.closure_end_min:.1f} min): "
        f"{res.frac_during_closure*100:.0f}% of events",
        f"   -> intercalation {'CONTRIBUTES: events concentrate during active closure' if res.frac_during_closure >= 0.5 else 'appears limited during closure'}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from . import io_utils, detection, segmentation
    tl = io_utils.load_wound()
    centers = detection.track_center(tl.images)
    geo = detection.radial_edges(stack=tl.images, centers=centers,
                                 px_size_um=tl.px_size_um, dt_s=tl.dt_s)
    seg = segmentation.analyze(tl.images, geo=geo, px_size_um=tl.px_size_um,
                               dt_s=tl.dt_s)
    res = analyze(seg, geo, tl.times_min())
    print(summary_text(res))
