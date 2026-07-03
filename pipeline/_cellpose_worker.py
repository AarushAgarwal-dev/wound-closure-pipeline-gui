"""
_cellpose_worker.py  --  isolated Cellpose segmentation subprocess
==================================================================

Runs Cellpose-SAM in its **own process** and writes ``mask.tiff``.  Cellpose on
CPU/PyTorch can segfault or run out of memory (see the dynamics.py sparse-tensor
warning); isolating it here means such a crash kills only this child, not the
Streamlit app, and frees the ~1.1 GB model when the process exits.

Usage:  python -m pipeline._cellpose_worker <params.json>
Prints  ``PROG <frac> <msg>`` lines on stdout for the parent's progress bar.
"""

from __future__ import annotations

import json
import os
import sys


def main():
    params_json = sys.argv[1]
    with open(params_json) as fh:
        d = json.load(fh)
    from pipeline.config import Params
    params = Params.from_dict(d)

    # CPU stability: single-threaded torch avoids the multi-thread segfaults
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        torch = None

    import numpy as np
    import tifffile
    from wound_analysis import io_utils, detection

    tl = io_utils.load_wound(params.data_dir, params.pattern)
    idx = params.resolve_frames(tl.n_frames)
    stack = tl.images[idx]

    from cellpose import models
    model = models.CellposeModel(gpu=params.cp_gpu)
    diam = params.cp_diameter if params.cp_diameter and params.cp_diameter > 0 else None

    masks = np.zeros(stack.shape, np.uint16)
    ctx = torch.inference_mode() if torch is not None else _nullctx()
    with ctx:
        for t in range(stack.shape[0]):
            img = detection.normalize(stack[t]).astype(np.float32)
            m, _flows, _styles = model.eval(
                img, diameter=diam,
                flow_threshold=params.cp_flow_threshold,
                cellprob_threshold=params.cp_cellprob_threshold,
            )
            masks[t] = m.astype(np.uint16)
            print(f"PROG {(t + 1) / stack.shape[0]:.4f} Cellpose frame "
                  f"{t + 1}/{stack.shape[0]}", flush=True)

    os.makedirs(params.out_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(params.out_dir, "mask.tiff"), masks)
    print("PROG 1.0 segmentation done", flush=True)


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
