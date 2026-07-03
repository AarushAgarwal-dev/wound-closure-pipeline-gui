"""
io_utils.py
===========

Load the exported confocal wound time-lapse and recover its physical
calibration (microns per pixel, seconds per frame) straight from the ImageJ
TIFF tags, so every downstream measurement comes out in real units.

The folder ``Wound/`` holds one single-channel (membrane / 561 nm) frame per
file, named ``..._t<NNNN>.tif``.  We sort by that time index and stack to
``(T, Y, X)`` uint16.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import tifffile


@dataclass
class Timelapse:
    images: np.ndarray      # (T, Y, X) uint16
    px_size_um: float       # microns per pixel
    dt_s: float             # seconds per frame
    files: list             # source file paths, in time order

    @property
    def n_frames(self) -> int:
        return self.images.shape[0]

    @property
    def dt_min(self) -> float:
        return self.dt_s / 60.0

    def times_min(self) -> np.ndarray:
        return np.arange(self.n_frames) * self.dt_min


def _frame_index(path: str) -> int:
    m = re.search(r"_t(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _read_calibration(path: str):
    """Return (px_size_um, dt_s) from ImageJ/TIFF tags, with sane fallbacks."""
    px_size_um, dt_s = 0.3448, 31.09  # measured fallbacks for this dataset
    try:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            tags = {t.name: t.value for t in page.tags.values()}
            xres = tags.get("XResolution")
            if xres:
                num, den = xres
                if num:
                    px_size_um = den / num  # pixels-per-unit -> unit-per-pixel
            desc = tags.get("ImageDescription", "") or ""
            m = re.search(r"finterval=([\d.]+)", desc)
            if m:
                dt_s = float(m.group(1))
    except Exception:
        pass
    return px_size_um, dt_s


def load_wound(folder: str = "Wound", pattern: str = "*.tif") -> Timelapse:
    """Load the wound time-lapse from ``folder`` into a :class:`Timelapse`."""
    files = sorted(glob.glob(os.path.join(folder, pattern)), key=_frame_index)
    if not files:
        raise FileNotFoundError(f"no TIFFs matching {pattern!r} in {folder!r}")
    px_size_um, dt_s = _read_calibration(files[0])
    stack = np.stack([tifffile.imread(f) for f in files]).astype(np.uint16)
    return Timelapse(images=stack, px_size_um=px_size_um, dt_s=dt_s, files=files)


if __name__ == "__main__":
    tl = load_wound()
    print(f"frames={tl.n_frames} shape={tl.images.shape[1:]} "
          f"px={tl.px_size_um:.4f} um/px  dt={tl.dt_s:.2f} s "
          f"({tl.dt_min:.3f} min)  total={tl.times_min()[-1]:.1f} min")
