"""
track.py  --  PHASE 2 / Step 3 (cell tracking)
==============================================

Stable cell IDs across time, building on the team's ``Cell_Tracking_2D``
notebook (Author: Linlin Li) and making it **more accurate**:

  * weighted match score = IoU(0.5) + size-similarity(0.3) + centroid(0.2),
    gated by min pixel-overlap / max centroid-distance / min IoU;
  * **optimal Hungarian assignment** (scipy ``linear_sum_assignment``) instead of
    greedy, so the globally best one-to-one matching is chosen each frame;
  * forward + backward propagation from a reference frame;
  * **gap closing**: track fragments separated by a few missing frames are
    re-linked (same identity), which removes most ID switches/fragmentation.

Outputs
-------
``tracked_mask.tiff``  -- cells carry their global id in every frame.
``track_summary.csv``  -- first/last frame, n_frames, coverage per track.
``traced_result_2D.npy`` -- {total_coordinate, total_traced_cell_mask, ...}.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment


def _features(mask):
    """label -> dict(centroid=(x,y), size) for all labels in a frame."""
    out = {}
    ids = np.unique(mask); ids = ids[ids != 0]
    for l in ids:
        ys, xs = np.where(mask == l)
        out[int(l)] = dict(centroid=(xs.mean(), ys.mean()), size=xs.size,
                           cx=xs.mean(), cy=ys.mean())
    return out


def _score(fa, fb, iou, p):
    size_sim = 1.0 / (1.0 + abs(fa["size"] - fb["size"]) / max(fa["size"], fb["size"]))
    dist = np.hypot(fa["centroid"][0] - fb["centroid"][0],
                    fa["centroid"][1] - fb["centroid"][1])
    cent_sim = 1.0 / (1.0 + dist / 100.0)
    return p.track_w_overlap * iou + p.track_w_size * size_sim + p.track_w_centroid * cent_sim


def match_frame(mask_curr, mask_prev, params):
    """Return dict curr_label -> prev_label using optimal/greedy matching."""
    fc = _features(mask_curr)
    fp = _features(mask_prev)
    if not fc or not fp:
        return {}
    cl = list(fc); pl = list(fp)
    S = np.zeros((len(cl), len(pl)))
    for i, c in enumerate(cl):
        a = fc[c]
        for j, p_ in enumerate(pl):
            b = fp[p_]
            if np.hypot(a["centroid"][0] - b["centroid"][0],
                        a["centroid"][1] - b["centroid"][1]) > params.track_max_distance_px:
                continue
            inter = int(np.logical_and(mask_curr == c, mask_prev == p_).sum())
            if inter < params.track_pixel_threshold:
                continue
            union = int(np.logical_or(mask_curr == c, mask_prev == p_).sum())
            iou = inter / union if union else 0
            if iou < params.track_min_iou:
                continue
            S[i, j] = _score(a, b, iou, params)

    matches = {}
    if params.track_method == "greedy":
        order = sorted(((S[i, j], i, j) for i in range(len(cl)) for j in range(len(pl)) if S[i, j] > 0),
                       reverse=True)
        uc, up = set(), set()
        for s, i, j in order:
            if i not in uc and j not in up:
                matches[cl[i]] = pl[j]; uc.add(i); up.add(j)
    else:  # hungarian (optimal)
        if S.max() > 0:
            row, col = linear_sum_assignment(-S)
            for i, j in zip(row, col):
                if S[i, j] > 0:
                    matches[cl[i]] = pl[j]
    return matches


def unified_tracking(masks, params, progress=None):
    n = len(masks)
    tracked = [np.zeros_like(masks[i], dtype=np.uint16) for i in range(n)]
    info = {"new_cells": {}, "lost_cells": {}}
    start = int(np.clip(params.track_start_frame, 0, n - 1))
    next_id = 1
    for lab in sorted(np.unique(masks[start])):
        if lab == 0:
            continue
        tracked[start][masks[start] == lab] = next_id
        next_id += 1

    def propagate(t_from, t_to):
        nonlocal next_id
        matches = match_frame(masks[t_to], masks[t_from], params)
        matched_ref = set()
        for c_label, r_label in matches.items():
            tids = np.unique(tracked[t_from][masks[t_from] == r_label])
            tids = tids[tids > 0]
            if tids.size:
                tracked[t_to][masks[t_to] == c_label] = tids[0]
                matched_ref.add(r_label)
        for c_label in np.unique(masks[t_to]):
            if c_label == 0 or c_label in matches:
                continue
            tracked[t_to][masks[t_to] == c_label] = next_id
            info["new_cells"].setdefault(t_to, []).append(next_id)
            next_id += 1

    done, total = 0, max(n - 1, 1)
    for t in range(start + 1, n):
        propagate(t - 1, t); done += 1
        if progress: progress(done / total, f"Tracking fwd {t + 1}/{n}")
    for t in range(start - 1, -1, -1):
        propagate(t + 1, t); done += 1
        if progress: progress(done / total, f"Tracking bwd {t + 1}/{n}")
    return tracked, info


def close_gaps(tracked, params):
    """Re-link track fragments separated by <= track_gap_frames missing frames.

    A track that disappears at frame te and another that appears at ts>te with
    ts-te-1 <= gap, a close centroid and similar size, are merged to one id."""
    gap = params.track_gap_frames
    if gap <= 0:
        return tracked
    n = len(tracked)
    ids = sorted({int(i) for f in tracked for i in np.unique(f) if i})
    # per-track presence, endpoint centroids/sizes
    span, head, tail = {}, {}, {}
    for i in ids:
        frames = [t for t in range(n) if np.any(tracked[t] == i)]
        if not frames:
            continue
        span[i] = (frames[0], frames[-1])
        for store, fr in ((head, frames[0]), (tail, frames[-1])):
            ys, xs = np.where(tracked[fr] == i)
            store[i] = (xs.mean(), ys.mean(), xs.size)

    remap = {}
    def root(x):
        while x in remap:
            x = remap[x]
        return x

    for a in ids:                                   # a ends, look for b starting after
        if a not in span:
            continue
        te = span[a][1]
        ax, ay, asz = tail[a]
        best, bestd = None, 1e9
        for b in ids:
            if b == a or b not in span:
                continue
            ts = span[b][0]
            if not (te < ts <= te + 1 + gap):
                continue
            bx, by, bsz = head[b]
            d = np.hypot(ax - bx, ay - by)
            if d > params.track_max_distance_px:
                continue
            if min(asz, bsz) / max(asz, bsz) < 0.5:
                continue
            if d < bestd:
                bestd, best = d, b
        if best is not None and root(best) != root(a):
            remap[best] = a

    if remap:
        for t in range(n):
            f = tracked[t]
            for b in list(remap):
                rb = root(b)
                if rb != b:
                    f[f == b] = rb
    return tracked


def run(masks, params, frame_idx=None, progress=None):
    T = masks.shape[0]
    if frame_idx is None:
        frame_idx = list(range(T))
    tracked_list, info = unified_tracking([masks[t] for t in range(T)], params, progress)
    tracked_list = close_gaps(tracked_list, params)
    tracked = np.stack(tracked_list).astype(np.uint16)

    os.makedirs(params.out_dir, exist_ok=True)
    path = os.path.join(params.out_dir, "tracked_mask.tiff")
    tifffile.imwrite(path, tracked)

    ids = sorted({int(i) for t in range(T) for i in np.unique(tracked[t]) if i})
    # trajectories (x,y) per track per frame; 0,0 when absent
    total_coordinate = []
    rows = []
    for i in ids:
        traj = np.zeros((T, 2), int)
        present = np.zeros(T, bool)
        for t in range(T):
            ys, xs = np.where(tracked[t] == i)
            if xs.size:
                traj[t] = [int(xs.mean()), int(ys.mean())]
                present[t] = True
        total_coordinate.append(traj)
        fr = np.where(present)[0]
        rows.append((i, int(frame_idx[fr[0]]), int(frame_idx[fr[-1]]),
                     int(present.sum()), round(float(present.mean()), 3)))

    summary_path = os.path.join(params.out_dir, "track_summary.csv")
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track_id", "first_frame", "last_frame", "n_frames", "coverage"])
        w.writerows(rows)

    np.save(os.path.join(params.out_dir, "traced_result_2D.npy"),
            {"total_coordinate": total_coordinate,
             "total_traced_cell_mask": [tracked[t] for t in range(T)],
             "track_ids": ids, "tracking_info": info}, allow_pickle=True)

    stats = dict(n_tracks=len(ids),
                 n_new=int(sum(len(v) for v in info["new_cells"].values())),
                 full_coverage=int(sum(1 for r in rows if r[4] == 1.0)))
    return tracked, path, stats
