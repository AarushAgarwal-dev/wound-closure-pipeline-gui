"""
config.py
=========

All tunable parameters for the pipeline in one dataclass, so the GUI and the
command line share exactly the same knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Params:
    # ---- input / output ----
    data_dir: str = "Wound"
    pattern: str = "*.tif"
    out_dir: str = "pipeline_out"
    frame_start: int = 0
    frame_end: int = -1           # -1 = last frame
    max_frames: int = 0           # 0 = no cap (else subsample to this many)

    # ---- calibration (auto-read from TIFF tags; these are fallbacks) ----
    px_size_um: float = 0.3448
    dt_s: float = 31.09

    # ---- Phase 1: segmentation ----
    backend: str = "cellpose"     # "cellpose" | "watershed"
    cp_model: str = "cpsam"       # cellpose model name
    cp_diameter: float = 30.0     # expected cell diameter (px); 0 -> auto
    cp_flow_threshold: float = 0.4
    cp_cellprob_threshold: float = 0.0
    cp_gpu: bool = False
    cp_isolate: bool = True       # run Cellpose in a subprocess (crash-safe, frees RAM)
    # watershed fallback knobs
    ws_h: float = 0.02
    ws_smooth: float = 2.0

    # ---- Phase 1: mask cleaning ----
    min_cell_area_px: int = 60
    fill_holes: bool = True
    remove_border_cells: bool = False
    smooth_boundaries: bool = True

    # ---- Step 3: tracking (bipartite matching, after Linlin Li) ----
    track_method: str = "hungarian"       # "hungarian" (optimal) | "greedy"
    track_min_iou: float = 0.1            # min IoU to consider a match
    track_pixel_threshold: int = 50       # min absolute pixel overlap to accept
    track_max_distance_px: int = 50       # max centroid-centroid distance
    track_start_frame: int = 0            # reference frame; tracks fwd + bwd
    track_gap_frames: int = 2             # close gaps up to this many frames
    track_w_overlap: float = 0.5          # score weights (sum to 1)
    track_w_size: float = 0.3
    track_w_centroid: float = 0.2

    # ---- visualisation / GIFs ----
    make_gifs: bool = True
    gif_duration: float = 0.25            # seconds per frame

    # ---- intensity branch (after Linlin Li) ----
    run_intensity: bool = True
    bg_ring_px: int = 10                  # background ring around each cell bbox
    intensity_min_frames: int = 5         # min frames present for trajectory plots
    norm_ref_frame: int = 0               # normalise fold-change to this frame

    # ---- Step 4: morphology ----
    min_cell_pixels: int = 50             # skip cells smaller than this
    shape_use_convexhull: bool = True     # boundary = convex hull (their method)
    neighbor_method: str = "touch"        # "touch" (4-conn) | "dilate" (gap-tolerant)
    neighbor_dist_px: int = 3             # gap tolerance when method = "dilate"

    # ---- Step 5: edge kinematics ----
    n_edge_points: int = 100      # points sampled around a cell boundary
    edge_cell_id: int = -1        # tracked-cell id to analyse; -1 = most persistent
    velocity_smooth_frames: float = 1.0

    def resolve_frames(self, n_total):
        end = n_total - 1 if self.frame_end < 0 else min(self.frame_end, n_total - 1)
        idx = list(range(self.frame_start, end + 1))
        if self.max_frames and len(idx) > self.max_frames > 0:
            step = len(idx) / self.max_frames
            idx = [idx[int(i * step)] for i in range(self.max_frames)]
        return idx

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        keep = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**keep)
