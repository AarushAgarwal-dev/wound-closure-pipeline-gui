"""
edge_velocity.py  --  Objective 1
=================================

Quantify wound-edge velocity and overall closure speed, and decide:
  * symmetric vs. asymmetric closure, and
  * constant vs. time-varying closure velocity.

Implements the challenge's action plan on top of the radial
:class:`~wound_analysis.detection.WoundGeometry`:

  edge detection      -> detection.radial_edges (half-max actin-cable margin)
  edge sampling       -> n_angles rays give edge points r(theta, t)
  define edge window  -> per-angle arc window; cable intensity sampled there
  quantify intensity  -> geo.edge_intensity (cable peak per ray)
  edge point tracking -> r(theta, t) followed frame to frame (fixed centre)
  velocity            -> v(theta, t) = -d r/dt  (inward-positive, um/min)
  intensity/velocity  -> correlate cable intensity with local closure speed
  plotting            -> closure curve, kymograph, polar symmetry, scatter
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from . import plotting


@dataclass
class EdgeVelocityResult:
    times_min: np.ndarray
    area_um2: np.ndarray
    equiv_radius_um: np.ndarray
    radius_um: np.ndarray            # (T, A) per-angle edge radius
    velocity: np.ndarray            # (T, A) inward edge velocity, um/min
    edge_intensity: np.ndarray      # (T, A)
    angles: np.ndarray
    open_window: tuple              # (t_start, t_close) frame indices
    closure_rate_um_per_min: float  # mean radial closure speed over window
    closure_rate_area: float        # mean -dA/dt over window, um^2/min
    sector_speed: dict              # mean inward speed per named sector
    sector_ratio: float             # fastest / slowest sector speed
    asymmetry_index: float          # 0 = isotropic, 1 = fully one-sided
    asymmetric: bool                # overall verdict
    velocity_constant: bool         # True if closure speed ~ constant in time
    decel_pvalue: float             # slope significance of speed-vs-time
    iv_correlation: float           # Pearson r between cable intensity & speed
    iv_pvalue: float


def _find_open_window(equiv_r_um, min_r=1.2):
    """Frames spanning the open wound: from the radius peak to first closure."""
    t_peak = int(np.argmax(equiv_r_um[: max(4, len(equiv_r_um) // 2)]))
    t_close = t_peak + 1
    while t_close < len(equiv_r_um) and equiv_r_um[t_close] > min_r:
        t_close += 1
    t_close = min(t_close, len(equiv_r_um) - 1)
    return t_peak, t_close


def analyze(geo, times_min):
    """Compute every Objective-1 metric from a ``WoundGeometry``."""
    px = geo.px_size_um
    dt_min = geo.dt_s / 60.0
    radius_um = geo.radius * px                       # (T, A)
    area = geo.area_um2()
    equiv_r = geo.equiv_radius_um()

    # inward edge velocity: -d r/dt along time, per angle
    drdt = np.gradient(radius_um, dt_min, axis=0)     # um/min
    velocity = -drdt

    t0, t1 = _find_open_window(equiv_r)
    win = slice(t0, t1 + 1)

    # overall closure rate (radial + area) across the open window
    closure_rate = (equiv_r[t0] - equiv_r[t1]) / max((t1 - t0) * dt_min, 1e-6)
    closure_rate_area = (area[t0] - area[t1]) / max((t1 - t0) * dt_min, 1e-6)

    # ---- symmetry: per-angle mean inward speed over the open window --------
    per_angle_speed = velocity[win].mean(0)           # (A,)
    ang = geo.angles
    # image convention: angle 0 = +x (right), +pi/2 = +y (down)
    cos45 = np.cos(np.pi / 4)
    sector_speed = {
        "right": float(per_angle_speed[np.cos(ang) > cos45].mean()),
        "left": float(per_angle_speed[np.cos(ang) < -cos45].mean()),
        "down": float(per_angle_speed[np.sin(ang) > cos45].mean()),
        "up": float(per_angle_speed[np.sin(ang) < -cos45].mean()),
    }
    # asymmetry index from the directional (vector) mean of edge displacement:
    # if the wound closes evenly the displacement vectors cancel; a net vector
    # means one side advances faster.  Normalise by the mean speed magnitude.
    vx = np.mean(per_angle_speed * np.cos(ang))
    vy = np.mean(per_angle_speed * np.sin(ang))
    net = np.hypot(vx, vy)
    asym = float(net / (np.mean(np.abs(per_angle_speed)) + 1e-9))
    pos = [v for v in sector_speed.values() if np.isfinite(v) and v > 0]
    sector_ratio = float(max(pos) / min(pos)) if len(pos) >= 2 and min(pos) > 0 else np.inf
    asymmetric = bool(asym > 0.2 or sector_ratio > 2.0)

    # ---- constant vs. time-varying closure speed --------------------------
    speed_t = velocity[win].mean(1)                   # spatial-mean speed / frame
    tt = times_min[win]
    slope, p_slope = _linfit_pvalue(tt, speed_t)
    velocity_constant = bool(p_slope > 0.05)          # no significant trend

    # ---- intensity / velocity correlation ---------------------------------
    I = geo.edge_intensity[win].ravel()
    V = velocity[win].ravel()
    r_iv, p_iv = _pearson(I, V)

    return EdgeVelocityResult(
        times_min=times_min,
        area_um2=area,
        equiv_radius_um=equiv_r,
        radius_um=radius_um,
        velocity=velocity,
        edge_intensity=geo.edge_intensity,
        angles=geo.angles,
        open_window=(t0, t1),
        closure_rate_um_per_min=float(closure_rate),
        closure_rate_area=float(closure_rate_area),
        sector_speed=sector_speed,
        sector_ratio=sector_ratio,
        asymmetry_index=asym,
        asymmetric=asymmetric,
        velocity_constant=velocity_constant,
        decel_pvalue=float(p_slope),
        iv_correlation=float(r_iv),
        iv_pvalue=float(p_iv),
    )


# --------------------------------------------------------------------------- #
# small stats helpers (avoid a scipy.stats dependency surprise)
# --------------------------------------------------------------------------- #
def _pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return 0.0, 1.0
    r = np.corrcoef(x, y)[0, 1]
    n = x.size
    # t -> two-sided p via survival of normal approx (good enough for n large)
    if abs(r) >= 1:
        return float(r), 0.0
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2))
    return float(r), float(p)


def _linfit_pvalue(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3:
        return 0.0, 1.0
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = coef[0]
    yhat = A @ coef
    resid = y - yhat
    s2 = (resid @ resid) / max(x.size - 2, 1)
    sxx = ((x - x.mean()) ** 2).sum()
    se = np.sqrt(s2 / sxx) if sxx > 0 else np.inf
    if se == 0 or not np.isfinite(se):
        return float(slope), 1.0
    t = slope / se
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2))
    return float(slope), float(p)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def plot_all(geo, res, stack, outdir="results"):
    from . import detection
    plotting.apply_style()
    paths = []
    t0, t1 = res.open_window
    tmin = res.times_min

    # 1) closure curve: radius + area vs time -------------------------------
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(tmin, res.equiv_radius_um, "o-", color="#1f77b4", label="equiv. radius")
    ax1.axvspan(tmin[t0], tmin[t1], color="orange", alpha=0.12, label="closure window")
    ax1.set_xlabel("time (min)")
    ax1.set_ylabel("equivalent wound radius (µm)", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(tmin, res.area_um2, "s--", color="#d62728", alpha=0.6, label="area")
    ax2.set_ylabel("wound area (µm²)", color="#d62728")
    ax2.grid(False)
    ax1.set_title(f"Wound closure  •  mean rate {res.closure_rate_um_per_min:.2f} µm/min"
                  f"  •  {'CONSTANT' if res.velocity_constant else 'TIME-VARYING'} speed")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.86), fontsize=9)
    paths.append(plotting.save(fig, "obj1_closure_curve.png", outdir))

    # 2) kymograph of edge radius r(theta, t) -------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(res.radius_um.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[tmin[0], tmin[-1], 0, 360])
    ax.set_xlabel("time (min)")
    ax.set_ylabel("edge angle (deg)")
    ax.set_title("Edge-radius kymograph  r(θ, t)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="radius (µm)")
    paths.append(plotting.save(fig, "obj1_kymograph.png", outdir))

    # 3) polar symmetry: mean inward speed per angle ------------------------
    win = slice(t0, t1 + 1)
    per_angle = res.velocity[win].mean(0)
    fig, ax = plt.subplots(figsize=(5.6, 5.6), subplot_kw={"projection": "polar"})
    th = res.angles
    ax.plot(np.r_[th, th[0]], np.r_[per_angle, per_angle[0]], "-", color="#2ca02c", lw=2)
    ax.fill(np.r_[th, th[0]], np.r_[per_angle, per_angle[0]], color="#2ca02c", alpha=0.15)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)  # image y-down -> clockwise
    ax.set_title(f"Inward edge speed by direction (µm/min)\n"
                 f"asym index={res.asymmetry_index:.2f}, "
                 f"fast/slow={res.sector_ratio:.1f}×  →  "
                 f"{'ASYMMETRIC' if res.asymmetric else 'symmetric'}",
                 fontsize=10.5)
    paths.append(plotting.save(fig, "obj1_symmetry_polar.png", outdir))

    # 4) intensity vs velocity scatter --------------------------------------
    I = geo.edge_intensity[win].ravel()
    V = res.velocity[win].ravel()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(I, V, s=8, alpha=0.3, color="#9467bd")
    if np.isfinite(I).sum() > 2:
        m, b = np.polyfit(I[np.isfinite(I)], V[np.isfinite(I)], 1)
        xs = np.linspace(np.nanmin(I), np.nanmax(I), 50)
        ax.plot(xs, m * xs + b, "k--", lw=1.5)
    ax.set_xlabel("actin-cable edge intensity (a.u.)")
    ax.set_ylabel("inward edge velocity (µm/min)")
    ax.set_title(f"Edge intensity vs. velocity  •  r = {res.iv_correlation:.2f} "
                 f"(p = {res.iv_pvalue:.1e})")
    paths.append(plotting.save(fig, "obj1_intensity_velocity.png", outdir))

    # 5) edge overlay montage on the raw frames -----------------------------
    sel = np.linspace(t0, t1, 8).astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    for ax, t in zip(axes.ravel(), sel):
        ax.imshow(detection.normalize(stack[t]), cmap="gray")
        xy = geo.edge_xy(t)
        xy = np.vstack([xy, xy[0]])
        ax.plot(xy[:, 0], xy[:, 1], "-", color="lime", lw=1.4)
        ax.plot(geo.centers[t, 1], geo.centers[t, 0], "r+", ms=9)
        cy, cx = geo.centers[t]
        ax.set_xlim(cx - 95, cx + 95)
        ax.set_ylim(cy + 95, cy - 95)
        ax.set_title(f"t={tmin[t]:.1f} min  r={res.equiv_radius_um[t]:.1f}µm", fontsize=10)
        ax.axis("off")
    fig.suptitle("Detected wound edge (radial half-max margin)", fontweight="bold")
    paths.append(plotting.save(fig, "obj1_edge_overlay.png", outdir))
    return paths


def summary_text(res):
    t0, t1 = res.open_window
    lines = [
        "OBJECTIVE 1  —  Wound-edge velocity & closure dynamics",
        "-" * 58,
        f"Open-wound window           : frames {t0}-{t1} "
        f"({res.times_min[t0]:.1f}-{res.times_min[t1]:.1f} min)",
        f"Initial / final radius      : {res.equiv_radius_um[t0]:.2f} -> "
        f"{res.equiv_radius_um[t1]:.2f} µm",
        f"Mean radial closure speed   : {res.closure_rate_um_per_min:.2f} µm/min",
        f"Mean area closure rate      : {res.closure_rate_area:.1f} µm²/min",
        f"Closure velocity over time  : "
        f"{'CONSTANT' if res.velocity_constant else 'CHANGES (accelerates/decelerates)'} "
        f"(slope p={res.decel_pvalue:.2g})",
        f"Per-direction inward speed  : "
        + ", ".join(f"{k}={v:.2f}" for k, v in res.sector_speed.items()) + " µm/min",
        f"Asymmetry index / ratio     : {res.asymmetry_index:.2f} / "
        f"{res.sector_ratio:.1f}x  -> "
        f"{'ASYMMETRIC' if res.asymmetric else 'approximately SYMMETRIC'} closure",
        f"Cable-intensity vs velocity : r={res.iv_correlation:.2f} (p={res.iv_pvalue:.1e})",
    ]
    return "\n".join(lines)
