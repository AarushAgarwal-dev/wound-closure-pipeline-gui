"""
run.py  --  pipeline orchestrator
=================================

Runs the whole workflow in diagram order and returns a results dict.  Used by
both the command line (``python -m pipeline.run``) and the Streamlit GUI.

    PHASE 1  segment -> mask.tiff -> clean -> cleaned_mask.tiff
    PHASE 2  track (Step 3) | morphology (Step 4) | kinematics (Step 5)
"""

from __future__ import annotations

import json
import os

from . import config, segment, clean, track, morphology, kinematics, viz, intensity


def run_pipeline(params, progress=None, steps=("track", "morphology", "kinematics")):
    """Execute the pipeline. ``progress(frac, msg)`` is an optional callback."""
    def prog(stage_lo, stage_hi):
        if progress is None:
            return None
        return lambda f, m: progress(stage_lo + (stage_hi - stage_lo) * f, m)

    res = {"params": params.to_dict()}

    # ---- load ----
    if progress: progress(0.0, "Loading frames ...")
    stack, frame_idx, tl = segment.load_stack(params)
    res["n_frames"] = len(frame_idx)
    res["frame_idx"] = frame_idx
    res["px_size_um"] = params.px_size_um
    res["dt_s"] = params.dt_s

    # ---- PHASE 1: segment ----
    masks, mask_path, backend = segment.run(stack, params, prog(0.02, 0.55))
    res["mask_path"] = mask_path
    res["backend"] = backend
    res["n_cells_raw"] = int(masks.max())

    # ---- PHASE 1: clean ----
    cleaned, clean_path = clean.run(masks, params, prog(0.55, 0.65))
    res["cleaned_path"] = clean_path

    # ---- PHASE 2 / Step 3: track ----
    tracked, tracked_path, tstats = track.run(cleaned, params, frame_idx,
                                              progress=prog(0.65, 0.74))
    res["tracked_path"] = tracked_path
    res["n_tracks"] = tstats["n_tracks"]
    res["track_stats"] = tstats

    outputs = {"stack": stack, "masks": masks, "cleaned": cleaned, "tracked": tracked,
               "frame_idx": frame_idx}

    # ---- tracking visualisations: GIF (per frame) + trajectory + counts ----
    times_min = [t * params.dt_s / 60.0 for t in range(len(frame_idx))]
    if params.make_gifs:
        if progress: progress(0.75, "Rendering tracking GIF ...")
        gif_path, _ = viz.save_tracking_gif(tracked, params.out_dir, params.gif_duration)
        res["tracking_gif"] = gif_path
        res["overlay_gif"] = viz.save_overlay_gif(stack, tracked, params.out_dir,
                                                  params.gif_duration)
    res["trajectory_plot"] = viz.trajectory_plot(tracked, params.out_dir)
    res["cells_per_frame_plot"] = viz.cells_per_frame_plot(tracked, times_min, params.out_dir)

    # ---- PHASE 2 / Step 4: morphology ----
    if "morphology" in steps:
        rows, morph_path = morphology.run(cleaned, params, frame_idx, prog(0.78, 0.9))
        res["morphology_path"] = morph_path
        res["n_morph_rows"] = len(rows)
        outputs["morphology_rows"] = rows

    # ---- intensity branch (per-cell fluorescence over time) ----
    if params.run_intensity:
        irows, ipath, ispath, ifig = intensity.run(stack, tracked, params, frame_idx,
                                                    progress=prog(0.9, 0.95))
        res["intensity_path"] = ipath
        res["intensity_summary_path"] = ispath
        res["intensity_plot"] = ifig
        res["n_intensity_rows"] = len(irows)

    # ---- PHASE 2 / Step 5: kinematics ----
    if "kinematics" in steps:
        krows, kpath, cid, contours, present = kinematics.run(
            tracked, params, frame_idx, prog(0.95, 0.99))
        res["edge_velocity_path"] = kpath
        res["edge_cell_id"] = cid
        res["edge_frames_present"] = len(present)
        kfig = kinematics.plot(tracked, stack, contours, present, cid, params, frame_idx)
        res["edge_velocity_plot"] = kfig
        outputs["edge_contours"] = contours
        outputs["edge_frames_present"] = present

    # ---- manifest ----
    os.makedirs(params.out_dir, exist_ok=True)
    with open(os.path.join(params.out_dir, "manifest.json"), "w") as fh:
        json.dump({k: v for k, v in res.items() if k != "frame_idx"}, fh, indent=2)
    if progress: progress(1.0, "Done.")
    return res, outputs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Wound")
    ap.add_argument("--out", default="pipeline_out")
    ap.add_argument("--backend", default="watershed", choices=["cellpose", "watershed"])
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    pr = config.Params(data_dir=a.data, out_dir=a.out, backend=a.backend,
                       max_frames=a.max_frames)
    res, _ = run_pipeline(pr, progress=lambda f, m: print(f"[{f*100:5.1f}%] {m}"))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("params", "frame_idx")}, indent=2, default=str))
