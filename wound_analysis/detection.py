"""
detection.py
============

Tissue masking, wound-centre tracking, and **radial edge detection** of the
wound margin.

Why radial?  On this membrane-label thumbnail the wound is a small dark hole
ringed by a bright actin cable; intercellular membrane gaps are also dark, so a
plain binary threshold is unreliable.  Instead we exploit the geometry that the
wound is *roughly star-convex about its centre*: from the wound centre we cast
rays at many angles and locate the actin-cable intensity peak along each ray.
That peak radius ``r(theta, t)`` is the wound edge.  This is robust to noise,
gives sub-pixel edges, and feeds the whole action plan
(edge detection -> sampling -> windows -> intensity -> tracking -> velocity).

Public API
----------
``segment_tissue(stack)``           -> (tissue_mask, interior_mask)
``track_center(stack, roi)``        -> (T, 2) smoothed wound centres (y, x)
``radial_edges(stack, centers,...)``-> :class:`WoundGeometry`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects, binary_erosion, disk


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def normalize(frame, lo=1.0, hi=99.7):
    """Percentile contrast-stretch a frame to [0, 1] float."""
    f = frame.astype(np.float64)
    a, b = np.percentile(f, (lo, hi))
    return np.clip((f - a) / (b - a + 1e-9), 0, 1)


# --------------------------------------------------------------------------- #
# tissue mask
# --------------------------------------------------------------------------- #
def segment_tissue(stack, erode_px=18, rel_thr=0.22):
    """Tissue band mask (and an eroded interior) from the temporal mean."""
    mean = stack.astype(np.float64).mean(0)
    g = gaussian(normalize(mean), 12)
    tissue = g > g.max() * rel_thr
    tissue = ndi.binary_fill_holes(tissue)
    tissue = remove_small_objects(tissue, 8000)
    interior = binary_erosion(tissue, disk(erode_px))
    return tissue, interior


# --------------------------------------------------------------------------- #
# wound-centre tracking
# --------------------------------------------------------------------------- #
def track_center(stack, roi=None, bright_pct=99.3, smooth_frames=2.0):
    """Track the wound centre as the centroid of the bright actin signal.

    The actin cable is the brightest structure near the wound; its centroid is
    a stable centre once the wound forms.  Centres are smoothed in time.
    """
    T, H, W = stack.shape
    if roi is None:
        roi = np.zeros((H, W), bool)
        roi[int(0.28 * H):int(0.66 * H), int(0.28 * W):int(0.68 * W)] = True
    cy = np.zeros(T)
    cx = np.zeros(T)
    for t in range(T):
        f = gaussian(normalize(stack[t]), 3)
        f = np.where(roi, f, 0)
        thr = np.percentile(f[roi], bright_pct)
        bright = f > thr
        bright = remove_small_objects(bright, 30)
        if bright.sum() == 0:
            cy[t], cx[t] = ndi.center_of_mass(roi)
        else:
            # largest bright blob, to avoid stray speckle
            lab, n = ndi.label(bright)
            sizes = ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1))
            big = (np.argmax(sizes) + 1)
            cy[t], cx[t] = ndi.center_of_mass(lab == big)
    cy = gaussian_filter1d(cy, smooth_frames, mode="nearest")
    cx = gaussian_filter1d(cx, smooth_frames, mode="nearest")
    return np.column_stack([cy, cx])


# --------------------------------------------------------------------------- #
# radial edge detection
# --------------------------------------------------------------------------- #
@dataclass
class WoundGeometry:
    """Per-frame radial wound geometry, all radii in pixels."""

    angles: np.ndarray        # (A,) ray angles, radians
    radius: np.ndarray        # (T, A) edge radius per angle (px)
    edge_intensity: np.ndarray  # (T, A) actin-cable intensity in the edge window
    centers: np.ndarray       # (T, 2) wound centre (y, x)
    px_size_um: float
    dt_s: float

    # ------- derived geometry -------
    def edge_xy(self, t):
        """(A, 2) edge points (x, y) for frame ``t``."""
        cy, cx = self.centers[t]
        x = cx + self.radius[t] * np.cos(self.angles)
        y = cy + self.radius[t] * np.sin(self.angles)
        return np.column_stack([x, y])

    def area_px(self):
        """Wound area per frame (px^2) via the shoelace formula on the contour."""
        A = np.zeros(self.radius.shape[0])
        for t in range(self.radius.shape[0]):
            xy = self.edge_xy(t)
            x, y = xy[:, 0], xy[:, 1]
            A[t] = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return A

    def area_um2(self):
        return self.area_px() * self.px_size_um ** 2

    def equiv_radius_um(self):
        """Equivalent-circle radius from wound area (µm)."""
        return np.sqrt(self.area_um2() / np.pi)


def _ray_profile(img, cy, cx, ang, radii):
    """Sample ``img`` along one ray (sub-pixel, bilinear)."""
    ys = cy + radii * np.sin(ang)
    xs = cx + radii * np.cos(ang)
    return map_coordinates(img, [ys, xs], order=1, mode="nearest")


def robust_center(centers, lo=3, hi=26):
    """A single fixed wound centre = mean of tracked centres over a stable
    window.  The wound closes toward a point, so a fixed tissue-frame centre is
    the correct reference for measuring closure and asymmetry (a *moving*
    centre would cancel the radial closure signal)."""
    hi = min(hi, len(centers))
    return np.asarray(centers[lo:hi]).mean(0)


def radial_edges(
    stack,
    centers,
    px_size_um,
    dt_s,
    n_angles=120,
    r_max_px=46,
    smooth_img=2.0,
    azimuth_smooth=8.0,
    fixed_center=True,
):
    """Detect the wound margin per frame by radial half-max edge finding.

    Along each ray from the (fixed) wound centre the intensity rises from the
    dark hole floor up to the bright actin cable.  The wound edge is the inner
    point where intensity crosses half-way between the hole floor and the cable
    peak -- i.e. the dark->bright transition that bounds the open wound.  The
    cable peak intensity at that ray is stored for the intensity/velocity
    correlation.  Returns a :class:`WoundGeometry`.
    """
    T, H, W = stack.shape
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    radii = np.arange(0, r_max_px, 0.5)
    radius = np.zeros((T, n_angles))
    edge_I = np.zeros((T, n_angles))

    centers = np.asarray(centers, float)
    if fixed_center:
        c = robust_center(centers)
        # refine: re-centre on the wound-contour centroid (1 pass), so the
        # radial origin sits inside the hole rather than on the bright cable
        c = _refine_center(stack, c, angles, radii, smooth_img)
        centers_used = np.tile(c, (T, 1))
    else:
        centers_used = centers

    for t in range(T):
        f = gaussian(normalize(stack[t]), smooth_img)
        cy, cx = centers_used[t]
        for a, ang in enumerate(angles):
            p = _ray_profile(f, cy, cx, ang, radii)
            rp = int(np.argmax(p))            # cable peak index
            peak = p[rp]
            floor = p[: rp + 1].min()
            half = 0.5 * (floor + peak)
            # inner crossing: last radius (going outward) still below half-max
            e = rp
            for r in range(rp, -1, -1):
                if p[r] < half:
                    e = r
                    break
            # linear sub-pixel between samples e and e+1
            if 0 <= e < rp:
                y0, y1 = p[e], p[e + 1]
                frac = (half - y0) / (y1 - y0) if abs(y1 - y0) > 1e-9 else 0.0
                radius[t, a] = radii[e] + np.clip(frac, 0, 1) * 0.5
            else:
                radius[t, a] = radii[e]
            edge_I[t, a] = peak

        radius[t] = _circular_smooth(radius[t], azimuth_smooth)
        edge_I[t] = _circular_smooth(edge_I[t], azimuth_smooth)

    return WoundGeometry(
        angles=angles,
        radius=radius,
        edge_intensity=edge_I,
        centers=centers_used,
        px_size_um=px_size_um,
        dt_s=dt_s,
    )


def _edge_one_frame(f, cy, cx, angles, radii):
    """Inner half-max edge radius per angle for a single (already smoothed)
    frame -- shared by the refinement pass and the main loop."""
    rad = np.zeros(angles.size)
    for a, ang in enumerate(angles):
        p = _ray_profile(f, cy, cx, ang, radii)
        rp = int(np.argmax(p))
        peak = p[rp]
        floor = p[: rp + 1].min()
        half = 0.5 * (floor + peak)
        e = rp
        for r in range(rp, -1, -1):
            if p[r] < half:
                e = r
                break
        rad[a] = radii[e]
    return rad


def _refine_center(stack, c, angles, radii, smooth_img, frames=range(2, 14)):
    """Recompute the wound centre as the mean contour centroid over the early
    open-wound frames, starting from ``c``."""
    cy, cx = c
    cents = []
    for t in frames:
        if t >= stack.shape[0]:
            break
        f = gaussian(normalize(stack[t]), smooth_img)
        rad = _edge_one_frame(f, cy, cx, angles, radii)
        rad = _circular_smooth(rad, 8.0)
        x = cx + rad * np.cos(angles)
        y = cy + rad * np.sin(angles)
        cents.append([y.mean(), x.mean()])
    return np.asarray(cents).mean(0) if cents else np.asarray(c)


def _circular_smooth(x, sigma):
    """Gaussian smooth a periodic 1-D signal."""
    if sigma <= 0:
        return x
    return gaussian_filter1d(x, sigma, mode="wrap")


if __name__ == "__main__":
    from . import io_utils

    tl = io_utils.load_wound()
    tissue, interior = segment_tissue(tl.images)
    centers = track_center(tl.images)
    geo = radial_edges(centers=centers, stack=tl.images,
                       px_size_um=tl.px_size_um, dt_s=tl.dt_s)
    r_um = geo.equiv_radius_um()
    print("equivalent wound radius (um) per frame:")
    print(np.round(r_um, 2))
