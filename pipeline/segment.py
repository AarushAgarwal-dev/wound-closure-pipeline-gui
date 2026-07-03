"""
segment.py  --  PHASE 1 (segmentation)
======================================

Input : original membrane TIFF tiles (one per time point).
Output: ``mask.tiff`` -- a (T, Y, X) uint16 label image, one integer per cell.

Two backends:
  * ``cellpose``  -- Cellpose-SAM deep-learning segmentation (default).
  * ``watershed`` -- the classic h-minima + watershed fallback (fast, no GPU,
    no model download), reused from the original toolkit.

Cellpose is imported lazily so the rest of the pipeline works without it.
"""

from __future__ import annotations

import os

import numpy as np
import tifffile

from wound_analysis import io_utils, detection


def load_stack(params):
    """Load frames selected by ``params`` -> (stack, frame_idx, Timelapse)."""
    tl = io_utils.load_wound(params.data_dir, params.pattern)
    idx = params.resolve_frames(tl.n_frames)
    stack = tl.images[idx]
    # honour auto-read calibration unless the user overrode the default
    params.px_size_um = tl.px_size_um
    params.dt_s = tl.dt_s
    return stack, idx, tl


# --------------------------------------------------------------------------- #
# Cellpose backend
# --------------------------------------------------------------------------- #
_CP_MODEL = None


def _get_cellpose_model(params):
    global _CP_MODEL
    if _CP_MODEL is None:
        from cellpose import models
        _CP_MODEL = models.CellposeModel(gpu=params.cp_gpu)
    return _CP_MODEL


def segment_cellpose(stack, params, progress=None):
    # single-threaded torch is much less segfault-prone on CPU
    try:
        import torch
        torch.set_num_threads(1)
        ctx = torch.inference_mode()
    except Exception:
        import contextlib
        ctx = contextlib.nullcontext()
    model = _get_cellpose_model(params)
    diam = params.cp_diameter if params.cp_diameter and params.cp_diameter > 0 else None
    masks = np.zeros(stack.shape, np.uint16)
    with ctx:
        for t in range(stack.shape[0]):
            img = detection.normalize(stack[t]).astype(np.float32)
            m, _flows, _styles = model.eval(
                img, diameter=diam,
                flow_threshold=params.cp_flow_threshold,
                cellprob_threshold=params.cp_cellprob_threshold,
            )
            masks[t] = m.astype(np.uint16)
            if progress:
                progress((t + 1) / stack.shape[0], f"Cellpose frame {t + 1}/{stack.shape[0]}")
    return masks


def segment_cellpose_isolated(params, progress=None):
    """Run Cellpose in a subprocess and read back mask.tiff.  A crash (segfault
    / OOM) kills only the child; we raise a clear error instead of dying."""
    import json
    import subprocess
    import sys
    import tifffile

    os.makedirs(params.out_dir, exist_ok=True)
    pjson = os.path.join(params.out_dir, "_seg_params.json")
    with open(pjson, "w") as fh:
        json.dump(params.to_dict(), fh)

    cmd = [sys.executable, "-u", "-m", "pipeline._cellpose_worker", pjson]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("PROG "):
            try:
                _, frac, msg = line.split(" ", 2)
                if progress:
                    progress(float(frac), msg)
            except ValueError:
                pass
        else:
            tail.append(line)
            if len(tail) > 15:
                tail.pop(0)
    proc.wait()
    mask_path = os.path.join(params.out_dir, "mask.tiff")
    if proc.returncode != 0 or not os.path.exists(mask_path):
        detail = "\n".join(tail[-8:])
        raise RuntimeError(
            f"Cellpose segmentation failed (exit code {proc.returncode}). This is "
            f"usually out-of-memory or a PyTorch/CPU crash on this dataset.\n"
            f"Try: the 'watershed' backend, fewer 'Max frames', or a smaller cell "
            f"diameter.\n--- worker output ---\n{detail}")
    return tifffile.imread(mask_path).astype(np.uint16)


# --------------------------------------------------------------------------- #
# Watershed backend (fallback)
# --------------------------------------------------------------------------- #
def segment_watershed(stack, params, progress=None):
    tissue, _ = detection.segment_tissue(stack)
    masks = np.zeros(stack.shape, np.uint16)
    from wound_analysis.segmentation import segment_frame
    for t in range(stack.shape[0]):
        labels, _ = segment_frame(stack[t], tissue, h=params.ws_h, smooth=params.ws_smooth)
        masks[t] = labels.astype(np.uint16)
        if progress:
            progress((t + 1) / stack.shape[0], f"Watershed frame {t + 1}/{stack.shape[0]}")
    return masks


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def cellpose_available():
    try:
        import cellpose  # noqa: F401
        return True
    except Exception:
        return False


def run(stack, params, progress=None):
    """Segment ``stack`` with the configured backend; save ``mask.tiff``."""
    backend = params.backend
    if backend == "cellpose" and not cellpose_available():
        backend = "watershed"
    if backend == "cellpose":
        if getattr(params, "cp_isolate", True):
            masks = segment_cellpose_isolated(params, progress)
        else:
            masks = segment_cellpose(stack, params, progress)
    else:
        masks = segment_watershed(stack, params, progress)
    os.makedirs(params.out_dir, exist_ok=True)
    path = os.path.join(params.out_dir, "mask.tiff")
    tifffile.imwrite(path, masks)
    return masks, path, backend
