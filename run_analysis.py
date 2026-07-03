#!/usr/bin/env python
"""
run_analysis.py
===============

End-to-end EMBRIO Design-Challenge pipeline for the zebrafish tailfin
wound-closure time-lapse in ``Wound/``.  Runs all three objectives, writes
every figure to ``results/`` and a text report to ``results/report.txt``.

Usage
-----
    python run_analysis.py                 # uses ./Wound and ./results
    python run_analysis.py --data Wound --out results
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from wound_analysis import (io_utils, detection, edge_velocity,
                            segmentation, intercalation)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="Wound", help="folder of TIFF frames")
    ap.add_argument("--out", default="results", help="output folder")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(args.out, exist_ok=True)

    print("=" * 62)
    print("EMBRIO Design Challenge — wound-closure analysis")
    print("=" * 62)

    # ---- load -------------------------------------------------------------
    tl = io_utils.load_wound(args.data)
    print(f"Loaded {tl.n_frames} frames {tl.images.shape[1:]} | "
          f"{tl.px_size_um:.4f} µm/px | {tl.dt_s:.1f} s/frame | "
          f"{tl.times_min()[-1]:.1f} min total\n")

    # ---- shared geometry --------------------------------------------------
    tissue, _ = detection.segment_tissue(tl.images)
    centers = detection.track_center(tl.images)
    geo = detection.radial_edges(stack=tl.images, centers=centers,
                                 px_size_um=tl.px_size_um, dt_s=tl.dt_s)

    report = []

    # ---- Objective 1 ------------------------------------------------------
    ev = edge_velocity.analyze(geo, tl.times_min())
    print(edge_velocity.summary_text(ev), "\n")
    report.append(edge_velocity.summary_text(ev))
    edge_velocity.plot_all(geo, ev, tl.images, outdir=args.out)

    # ---- Objective 2 ------------------------------------------------------
    seg = segmentation.analyze(tl.images, geo=geo, tissue=tissue,
                               px_size_um=tl.px_size_um, dt_s=tl.dt_s)
    print(segmentation.summary_text(seg, open_window=ev.open_window), "\n")
    report.append(segmentation.summary_text(seg, open_window=ev.open_window))
    segmentation.plot_all(seg, tl.images, tl.times_min(), outdir=args.out,
                          open_window=ev.open_window)

    # ---- Objective 3 ------------------------------------------------------
    inter = intercalation.analyze(seg, geo, tl.times_min(),
                                  closure_window=ev.open_window)
    print(intercalation.summary_text(inter), "\n")
    report.append(intercalation.summary_text(inter))
    intercalation.plot_all(inter, seg, geo, tl.images, outdir=args.out)

    # ---- report -----------------------------------------------------------
    header = (f"EMBRIO wound-closure analysis report\n"
              f"data: {args.data}  |  {tl.n_frames} frames  |  "
              f"{tl.px_size_um:.4f} µm/px  |  {tl.dt_s:.1f} s/frame\n\n")
    with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(header + "\n\n".join(report) + "\n")

    figs = sorted(f for f in os.listdir(args.out) if f.endswith(".png"))
    print("=" * 62)
    print(f"Done.  {len(figs)} figures + report.txt written to {args.out}/")
    for f in figs:
        print("   ", f)


if __name__ == "__main__":
    main()
