"""
app.py  --  Streamlit GUI for the wound-closure pipeline
========================================================

Change parameters in the sidebar, hit RUN, and the whole diagram executes:

    PHASE 1  segment (Cellpose / watershed) -> mask -> clean -> cleaned_mask
    PHASE 2  Step 3 track | Step 4 morphology | Step 5 edge kinematics

Run with:   streamlit run app.py
"""

from __future__ import annotations

import os
import glob

import numpy as np
import streamlit as st

from pipeline import config, run as prun, segment as seg_mod
from wound_analysis import detection

st.set_page_config(page_title="Wound-Closure Pipeline", page_icon="🔬", layout="wide")

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def label_rgb(labels, base=None):
    """Colour a label image with stable random colours, optionally over base."""
    from skimage.color import label2rgb
    if base is not None:
        return label2rgb(labels, image=base, bg_label=0, alpha=0.45,
                         image_alpha=1.0, kind="overlay")
    return label2rgb(labels, bg_label=0)


def norm_rgb(frame):
    f = detection.normalize(frame)
    return np.dstack([f, f, f])


@st.cache_data(show_spinner=False)
def load_csv(path, _mtime):
    """Cached CSV read (re-reads only when the file's mtime changes), so
    stepping frames doesn't re-read every table on every rerun."""
    import pandas as pd
    return pd.read_csv(path)


def csv_cached(path):
    try:
        return load_csv(path, os.path.getmtime(path))
    except Exception:
        import pandas as pd
        return pd.read_csv(path)


def save_cleaned_to_disk(out, params_obj):
    """Persist the (possibly hand-edited) cleaned masks to cleaned_mask.tiff."""
    import tifffile
    os.makedirs(params_obj.out_dir, exist_ok=True)
    path = os.path.join(params_obj.out_dir, "cleaned_mask.tiff")
    tifffile.imwrite(path, out["cleaned"].astype(np.uint16))
    return path


def masks_to_tiff_bytes(arr):
    """Encode a label stack to in-memory TIFF bytes for download (multi-page)."""
    import io
    import tifffile
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr.astype(np.uint16))
    return buf.getvalue()


def masks_to_zip_bytes(arr, prefix="cleaned_mask"):
    """Pack a label stack as a .zip of one TIFF per frame (all masks, separate files)."""
    import io
    import zipfile
    import tifffile
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for t in range(arr.shape[0]):
            fbuf = io.BytesIO()
            tifffile.imwrite(fbuf, arr[t].astype(np.uint16))
            zf.writestr(f"{prefix}_t{t:03d}.tif", fbuf.getvalue())
    return zbuf.getvalue()


def _to_u8(img):
    """Coerce a float [0,1] or arbitrary image to uint8 RGB."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.dstack([img, img, img])
    return img[..., :3]


def read_uploaded_tiff_stack(files):
    """Read uploaded TIFF(s) into a (T, Y, X) stack. Accepts several single-frame
    files (sorted by name) or a single multi-page TIFF."""
    import io
    import tifffile
    if not isinstance(files, (list, tuple)):
        files = [files]
    arrs = []
    for f in sorted(files, key=lambda x: x.name):
        a = tifffile.imread(io.BytesIO(f.read()))
        if a.ndim == 3:
            arrs.extend(list(a))
        else:
            arrs.append(a)
    return np.stack(arrs, axis=0)


def align_intensity_stack(arr, frame_idx, ref_hw):
    """Match an uploaded intensity stack to the analysed frames. Accepts a stack
    with exactly len(frame_idx) frames, or a full timelapse we sub-sample by
    frame_idx. Raises ValueError on a size mismatch."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    T = len(frame_idx)
    if arr.shape[0] == T:
        sub = arr
    elif arr.shape[0] > max(frame_idx):
        sub = arr[list(frame_idx)]
    else:
        raise ValueError(f"Uploaded {arr.shape[0]} frames; need {T} analysed "
                         f"frames (or the full {max(frame_idx) + 1}+ timelapse).")
    if tuple(sub.shape[1:]) != tuple(ref_hw):
        raise ValueError(f"Frame size {tuple(sub.shape[1:])} doesn't match the "
                         f"data {tuple(ref_hw)}.")
    return sub.astype(np.float32)


def frames_to_gif_bytes(frames, fps=4):
    """Encode a list of uint8 RGB frames to in-memory GIF bytes."""
    import io
    import imageio.v2 as imageio
    frames = [_to_u8(f) for f in frames]
    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", duration=1.0 / max(fps, 1), loop=0)
    return buf.getvalue()


def gif_download_widget(label, key, build_frames, fps=4, filename=None, out_dir=None):
    """Default save/download control for an auto-play viewer.

    One click: build the GIF from the viewer's frames (``build_frames`` is a
    0-arg callable), SAVE it to ``out_dir``, and expose a download button.
    """
    filename = filename or f"{key}.gif"
    if st.button(f"💾 Save & Download {label} GIF", key=f"build_{key}",
                 help="Builds the GIF over all frames, saves it to the output "
                      "folder, and offers a download"):
        with st.spinner(f"Building {label} GIF..."):
            try:
                gifb = frames_to_gif_bytes(build_frames(), fps)
                st.session_state[f"gifbytes_{key}"] = gifb
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                    p = os.path.join(out_dir, filename)
                    with open(p, "wb") as fh:
                        fh.write(gifb)
                    st.session_state[f"gifpath_{key}"] = p
            except Exception as e:
                st.error(f"GIF build failed: {e}")
    gifb = st.session_state.get(f"gifbytes_{key}")
    if gifb:
        saved = st.session_state.get(f"gifpath_{key}")
        if saved:
            st.success(f"Saved → {saved}")
        st.image(gifb, caption=f"{label} GIF", width='stretch')
        st.download_button(f"⬇ Download {label} GIF", gifb, filename,
                           "image/gif", key=f"dl_{key}", width='stretch')


def boundary_intensity_heatmap(int_df, key_prefix, n_frames=None):
    """Render the per-boundary-point intensity heatmap (Point ID × Timeframe)
    PLUS a mean ± std line plot over time, with a colourmap picker, editable
    title and PNG download. The x-axis spans ALL ``n_frames`` frames (gaps shown
    where no wound boundary was detected). Shared by the Wound Boundary and
    Intensity tabs."""
    if int_df is None or getattr(int_df, "empty", True):
        st.info("Run **▶ Run Wound Boundary Analysis** in the ⑤ Wound Boundary "
                "tab first — that produces the per-boundary-point intensity.")
        return
    st.caption("Per boundary-point fluorescence over time — same style as the "
               "F-actin / Myosin maps. Upload a channel above to change what's "
               "measured, then pick a matching colourmap. Frames with no detected "
               "wound edge (e.g. after closure) appear as gaps.")
    ih_cmap = st.selectbox(
        "Colourmap", ["viridis (F-actin style)", "inferno (Myosin style)",
                      "plasma", "magma", "turbo"], key=f"{key_prefix}_int_hm_cmap")
    cmap_name = ih_cmap.split(" ")[0]
    ih_title = st.text_input("Heatmap title", "Boundary intensity",
                             key=f"{key_prefix}_int_hm_title")
    import io as _io_hm
    import matplotlib.pyplot as _hmplt

    piv = int_df.pivot_table(index="point_id", columns="timeframe",
                             values="intensity").sort_index()
    # span every frame (0..n_frames-1), leaving NaN gaps where no wound edge
    if n_frames:
        piv = piv.reindex(columns=range(int(n_frames)))
    x_lo, x_hi = (0, int(n_frames) - 1) if n_frames else \
        (piv.columns.min(), piv.columns.max())

    cmap = _hmplt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white")
    masked = np.ma.masked_invalid(piv.values)

    fig, ax = _hmplt.subplots(figsize=(10, 4))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, origin="lower",
                   extent=[x_lo, x_hi, piv.index.min(), piv.index.max()])
    ax.set_xlabel("Timeframe")
    ax.set_ylabel("Point ID")
    _nf = int(n_frames) if n_frames else int(int_df["timeframe"].nunique())
    ax.set_title(f"{ih_title}  (n={_nf} frames)")
    fig.colorbar(im, ax=ax, label="Intensity (a.u.)")
    fig.tight_layout()
    st.pyplot(fig)
    buf = _io_hm.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    _hmplt.close(fig)
    st.download_button("⬇ intensity_heatmap.png", buf.getvalue(),
                       "boundary_intensity_heatmap.png", "image/png",
                       key=f"{key_prefix}_dl_int_hm")

    # ---- companion line plot: mean ± std over all frames -------------------
    mean_f = int_df.groupby("timeframe")["intensity"].mean()
    std_f = int_df.groupby("timeframe")["intensity"].std().fillna(0)
    if n_frames:
        idx = range(int(n_frames))
        mean_f = mean_f.reindex(idx)
        std_f = std_f.reindex(idx)
    fig2, ax2 = _hmplt.subplots(figsize=(10, 3))
    xv = np.array(list(mean_f.index), dtype=float)
    ax2.fill_between(xv, (mean_f - std_f).values, (mean_f + std_f).values,
                     color="mediumpurple", alpha=0.22)
    ax2.plot(xv, mean_f.values, "o-", color="mediumpurple", lw=2, ms=3)
    ax2.set_xlabel("Timeframe")
    ax2.set_ylabel("Mean boundary\nintensity (a.u.)")
    ax2.set_title(f"{ih_title} over time (mean ± std)")
    if n_frames:
        ax2.set_xlim(0, int(n_frames) - 1)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)
    buf2 = _io_hm.BytesIO()
    fig2.savefig(buf2, format="png", dpi=120, bbox_inches="tight")
    _hmplt.close(fig2)
    st.download_button("⬇ intensity_lineplot.png", buf2.getvalue(),
                       "boundary_intensity_lineplot.png", "image/png",
                       key=f"{key_prefix}_dl_int_line")


def rerun_downstream(out, res, params_obj, stack, progress=None):
    """Re-run Step 3-5 + intensity + GIFs from the current cleaned masks, so a
    correction (manual edit or uploaded file) propagates everywhere at once."""
    from pipeline import track, morphology, kinematics, viz, intensity
    f_idx = out["frame_idx"]
    cleaned = out["cleaned"]
    times_min = [t * params_obj.dt_s / 60.0 for t in range(len(f_idx))]

    tracked, tracked_path, tstats = track.run(cleaned, params_obj, f_idx)
    res["tracked_path"] = tracked_path
    res["n_tracks"] = tstats["n_tracks"]
    res["track_stats"] = tstats
    out["tracked"] = tracked
    if params_obj.make_gifs:
        res["tracking_gif"], _ = viz.save_tracking_gif(tracked, params_obj.out_dir, params_obj.gif_duration)
        res["overlay_gif"] = viz.save_overlay_gif(stack, tracked, params_obj.out_dir, params_obj.gif_duration)
    res["trajectory_plot"] = viz.trajectory_plot(tracked, params_obj.out_dir)
    res["cells_per_frame_plot"] = viz.cells_per_frame_plot(tracked, times_min, params_obj.out_dir)

    rows, morph_path = morphology.run(cleaned, params_obj, f_idx)
    res["morphology_path"] = morph_path
    out["morphology_rows"] = rows

    if params_obj.run_intensity:
        irows, ipath, ispath, ifig = intensity.run(stack, tracked, params_obj, f_idx)
        res["intensity_path"] = ipath
        res["intensity_summary_path"] = ispath
        res["intensity_plot"] = ifig

    krows, kpath, cid, contours, present = kinematics.run(tracked, params_obj, f_idx)
    res["edge_velocity_path"] = kpath
    res["edge_cell_id"] = cid
    res["edge_frames_present"] = len(present)
    res["edge_velocity_plot"] = kinematics.plot(tracked, stack, contours, present, cid, params_obj, f_idx)
    out["edge_contours"] = contours
    out["edge_frames_present"] = present
    # also persist the cleaned masks that produced these results
    save_cleaned_to_disk(out, params_obj)
    return res, out


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #
st.title("🔬 Zebrafish Wound-Closure Pipeline")
st.caption("Cellpose segmentation → mask cleaning → tracking · morphology · edge kinematics "
           "— EMBRIO Team 5")

cp_ok = seg_mod.cellpose_available()

# --------------------------------------------------------------------------- #
# sidebar parameters
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Parameters")

    with st.expander("📁 Input / output", expanded=True):
        data_dir = st.text_input("Data folder", "Wound")
        pattern = st.text_input("File pattern", "*.tif")
        out_dir = st.text_input("Output folder", "pipeline_out")
        n_avail = len(glob.glob(os.path.join(data_dir, pattern)))
        st.caption(f"{n_avail} frames found")
        max_frames = st.number_input("Max frames (0 = all)", 0, 1000, 8, step=1,
                                     help="Cellpose ≈15 s/frame on CPU — cap while experimenting.")

    with st.expander("① Segmentation", expanded=True):
        backend_opts = ["cellpose", "watershed"]
        backend = st.radio("Backend", backend_opts,
                           index=0 if cp_ok else 1,
                           help=None if cp_ok else "cellpose not installed — using watershed")
        if backend == "cellpose" and not cp_ok:
            st.warning("Cellpose unavailable; will fall back to watershed.")
        if backend == "cellpose":
            cp_diameter = st.slider("Cell diameter (px, 0=auto)", 0, 80, 30)
            cp_flow = st.slider("Flow threshold", 0.0, 3.0, 0.4, 0.1)
            cp_prob = st.slider("Cellprob threshold", -6.0, 6.0, 0.0, 0.5)
            cp_gpu = st.checkbox("Use GPU", False)
            cp_isolate = st.checkbox("Run Cellpose in a crash-safe subprocess", True,
                                     help="Isolates Cellpose so a segfault/OOM can't take "
                                          "down the app, and frees the model after each run.")
            ws_h, ws_smooth = 0.02, 2.0
        else:
            ws_h = st.slider("Watershed seed depth (h)", 0.005, 0.1, 0.02, 0.005)
            ws_smooth = st.slider("Smoothing σ (px)", 0.5, 4.0, 2.0, 0.5)
            cp_diameter, cp_flow, cp_prob, cp_gpu, cp_isolate = 30, 0.4, 0.0, False, True

    with st.expander("② Mask cleaning"):
        min_area = st.slider("Min cell area (px)", 0, 400, 60, 10)
        fill_holes = st.checkbox("Fill holes", True)
        remove_border = st.checkbox("Remove border cells", False)
        smooth_bound = st.checkbox("Smooth boundaries", True)

    with st.expander("③ Tracking (bipartite matching)"):
        track_iou = st.slider("Min IoU to link cells", 0.0, 0.9, 0.1, 0.05)
        track_pix = st.slider("Min pixel overlap", 0, 400, 50, 10)
        track_dist = st.slider("Max centroid distance (px)", 10, 150, 50, 5)
        track_start = st.number_input("Reference frame (tracks fwd+bwd)", 0, 1000, 0)
        track_method = st.radio("Assignment", ["hungarian", "greedy"],
                                help="hungarian = optimal one-to-one (more accurate); greedy = original")
        track_gap = st.slider("Gap-closing frames", 0, 5, 2,
                              help="re-link a track that vanishes for up to N frames (fewer ID switches)")

    with st.expander("④ Morphology"):
        neigh_method = st.radio("Neighbour detection", ["touch", "dilate"],
                                help="touch = 4-connectivity shared border (their method); "
                                     "dilate = expand labels first (tolerates Cellpose gaps)")
        neigh_dist = st.slider("Gap tolerance (px) [dilate]", 0, 10, 3,
                               disabled=(neigh_method == "touch"))
        min_px = st.slider("Min cell pixels", 0, 300, 50, 10)
        use_hull = st.checkbox("Convex-hull boundary", True)

    with st.expander("⑤ Edge kinematics"):
        n_edge = st.slider("Edge points per cell", 20, 200, 100, 10)
        edge_cell = st.number_input("Cell id (-1 = most persistent)", -1, 100000, -1)

    with st.expander("⑥ Intensity & GIFs"):
        run_intensity = st.checkbox("Run intensity analysis", True)
        bg_ring = st.slider("Background ring (px)", 0, 30, 10)
        intensity_minf = st.slider("Min frames for trajectory", 1, 30, 5)
        norm_ref = st.number_input("Normalise to frame", 0, 1000, 0)
        make_gifs = st.checkbox("Make tracking GIFs", True)
        gif_dur = st.slider("GIF seconds/frame", 0.05, 1.0, 0.25, 0.05)

    run_clicked = st.button("▶  RUN PIPELINE", type="primary", width='stretch')


def build_params():
    return config.Params(
        data_dir=data_dir, pattern=pattern, out_dir=out_dir, max_frames=int(max_frames),
        backend=backend, cp_diameter=float(cp_diameter), cp_flow_threshold=float(cp_flow),
        cp_cellprob_threshold=float(cp_prob), cp_gpu=bool(cp_gpu),
        cp_isolate=bool(cp_isolate),
        ws_h=float(ws_h), ws_smooth=float(ws_smooth),
        min_cell_area_px=int(min_area), fill_holes=fill_holes,
        remove_border_cells=remove_border, smooth_boundaries=smooth_bound,
        track_min_iou=float(track_iou), track_pixel_threshold=int(track_pix),
        track_max_distance_px=int(track_dist), track_start_frame=int(track_start),
        track_method=track_method, track_gap_frames=int(track_gap),
        neighbor_method=neigh_method, neighbor_dist_px=int(neigh_dist),
        min_cell_pixels=int(min_px), shape_use_convexhull=bool(use_hull),
        n_edge_points=int(n_edge), edge_cell_id=int(edge_cell),
        run_intensity=bool(run_intensity), bg_ring_px=int(bg_ring),
        intensity_min_frames=int(intensity_minf), norm_ref_frame=int(norm_ref),
        make_gifs=bool(make_gifs), gif_duration=float(gif_dur),
    )


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
if run_clicked:
    params = build_params()
    bar = st.progress(0.0, text="Starting ...")

    def cb(frac, msg):
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)

    try:
        with st.spinner("Running pipeline ..."):
            res, outputs = prun.run_pipeline(params, progress=cb)
        bar.progress(1.0, text="Done.")
        st.session_state["res"] = res
        st.session_state["outputs"] = outputs
        st.success(f"Pipeline finished — backend: {res['backend']} · "
                   f"{res['n_frames']} frames · {res['n_tracks']} tracked cells")
    except Exception as e:
        bar.empty()
        st.error("Pipeline failed — the app is still running. See details below.")
        st.exception(e)
        st.info("If this was the Cellpose backend, try **watershed** (Segmentation → "
                "Backend), reduce **Max frames**, or lower the **cell diameter**.")

# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
if "res" not in st.session_state:
    st.info("Set parameters on the left and press **RUN PIPELINE**. "
            "Tip: start with a small *Max frames* if using Cellpose on CPU.")
    st.stop()

res = st.session_state["res"]
out = st.session_state["outputs"]
stack = out["stack"]
T = stack.shape[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Frames", res["n_frames"])
c2.metric("Cells (raw)", res["n_cells_raw"])
c3.metric("Tracked cells", res["n_tracks"])
c4.metric("Backend", res["backend"])

def frame_slider_with_buttons(label, max_val, key, default=0):
    if key not in st.session_state:
        st.session_state[key] = default
        
    def step_frame(delta):
        st.session_state[key] = max(0, min(max_val, st.session_state[key] + delta))
        
    c1, c2, c3 = st.columns([1, 1, 8])
    with c1:
        st.button("⬅️", key=f"prev_{key}", on_click=step_frame, args=(-1,), width='stretch', help="Previous frame")
    with c2:
        st.button("➡️", key=f"next_{key}", on_click=step_frame, args=(1,), width='stretch', help="Next frame")
    with c3:
        return st.slider(label, 0, max_val, key=key, label_visibility="collapsed")

tab1, tab2, tab3, tab4, tab8, tab7, tab6 = st.tabs(
    ["① Segmentation", "② Manual Cleaning", "③ Tracking", "④ Morphology",
     "⑤ Wound Boundary", "⑥ Intensity", "📦 Files"])

# ---- Phase 1 ----
with tab1:
    import time
    # Non-blocking auto-play: advance ONE frame per rerun (a blocking for/sleep
    # loop freezes the script thread and drops the WebSocket -> "disconnects").
    play_seg = st.toggle("▶ Auto-play", key="play_seg")
    if play_seg:
        st.session_state["seg_frame"] = (st.session_state.get("seg_frame", 0) + 1) % T
    f = frame_slider_with_buttons("Frame", T - 1, "seg_frame", 0)

    a, b, c = st.columns(3)
    a.image(norm_rgb(stack[f]), caption=f"original tiff (frame {f})", width='stretch')
    b.image(label_rgb(out["masks"][f], detection.normalize(stack[f])),
            caption=f"mask.tiff (segmentation) (frame {f})", width='stretch')
    c.image(label_rgb(out["cleaned"][f], detection.normalize(stack[f])),
            caption=f"cleaned_mask.tiff (frame {f})", width='stretch')

    def _seg_frames():
        out_frames = []
        for i in range(T):
            base = detection.normalize(stack[i])
            orig = _to_u8(norm_rgb(stack[i]))
            mask = _to_u8(label_rgb(out["masks"][i], base))
            clean = _to_u8(label_rgb(out["cleaned"][i], base))
            out_frames.append(np.hstack([orig, mask, clean]))
        return out_frames
    gif_download_widget("segmentation", "seg", _seg_frames, fps=4,
                        filename="segmentation.gif", out_dir=res["params"]["out_dir"])

    if play_seg:
        time.sleep(0.25)
        st.rerun()

# ---- Phase 1: Manual Cleaning ----
with tab2:
    st.caption("Manual mask cleaning: Click on cells to edit the mask.")
    
    col_mc_action, col_mc_slider = st.columns([2, 2])
    with col_mc_action:
        action = st.radio("Action Mode", ["🧽 Erase", "➕ Add / Recover", "🔗 Merge", "✂️ Split", "🖌️ Brush Draw"], horizontal=True, key="mc_action")
    with col_mc_slider:
        f_mc = frame_slider_with_buttons("Frame", T - 1, "mc_frame", 0)
        
    # Clear merge target if mode changes
    if action != "🔗 Merge" and "merge_target" in st.session_state:
        del st.session_state["merge_target"]
    # Clear split state if mode changes
    if action != "✂️ Split" and "split_pending" in st.session_state:
        del st.session_state["split_pending"]
        
    # -- Compatibility shim for streamlit-drawable-canvas on Streamlit >=1.38 --
    # The canvas package calls st_image.image_to_url(img, width_int, clamp, channels, fmt, id)
    # but that function moved to streamlit.elements.lib.image_utils and now expects a
    # LayoutConfig object instead of an int width. We provide a thin wrapper.
    import streamlit.elements.image as st_image
    if not hasattr(st_image, "image_to_url"):
        from streamlit.elements.lib.image_utils import image_to_url as _real_image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig
        def _compat_image_to_url(image, width, clamp, channels, output_format, image_id):
            layout_cfg = _LayoutConfig(width=width if isinstance(width, str) else "content")
            return _real_image_to_url(image, layout_cfg, clamp, channels, output_format, image_id)
        st_image.image_to_url = _compat_image_to_url
    from streamlit_drawable_canvas import st_canvas
    from streamlit_image_coordinates import streamlit_image_coordinates
    from PIL import Image

    canvas_key = f"mc_coords_{f_mc}_{st.session_state.get('mc_update', 0)}"
    value = st.session_state.get(canvas_key) if action != "🖌️ Brush Draw" else None

    bg_img_array = label_rgb(out["cleaned"][f_mc], detection.normalize(stack[f_mc]))
    bg_img_array_uint8 = (bg_img_array * 255).astype(np.uint8)

    # Highlight cells based on action
    if value is not None:
        x, y = value["x"], value["y"]
        current_cleaned = out["cleaned"][f_mc]
        current_raw = out["masks"][f_mc]
        
        if 0 <= y < current_cleaned.shape[0] and 0 <= x < current_cleaned.shape[1]:
            cid = current_cleaned[y, x]
            
            if action == "🧽 Erase":
                if cid > 0:
                    bg_img_array_uint8[current_cleaned == cid] = [255, 0, 0] # Red
            
            elif action == "➕ Add / Recover":
                if cid == 0:
                    raw_cid = current_raw[y, x]
                    if raw_cid > 0:
                        bg_img_array_uint8[current_raw == raw_cid] = [0, 255, 0] # Green
                    else:
                        from skimage.segmentation import flood
                        from skimage.filters import gaussian
                        from scipy.ndimage import binary_fill_holes
                        img = detection.normalize(stack[f_mc])
                        smoothed = gaussian(img, sigma=2.0)
                        filled_mask = flood(smoothed, (y, x), tolerance=0.1)
                        filled_mask = binary_fill_holes(filled_mask)
                        bg_img_array_uint8[filled_mask & (current_cleaned == 0)] = [0, 255, 0] # Green
                        
            elif action == "🔗 Merge":
                target = st.session_state.get("merge_target")
                if target is not None:
                    bg_img_array_uint8[current_cleaned == target] = [0, 150, 255] # Blue for target
                if cid > 0:
                    if target is None or cid == target:
                        bg_img_array_uint8[current_cleaned == cid] = [0, 150, 255] # Blue for target selection
                    else:
                        bg_img_array_uint8[current_cleaned == cid] = [255, 255, 0] # Yellow for cell to merge
                else:
                    if target is not None:
                        raw_cid = current_raw[y, x]
                        if raw_cid > 0:
                            bg_img_array_uint8[current_raw == raw_cid] = [255, 255, 0] # Yellow
                        else:
                            from skimage.segmentation import flood
                            from skimage.filters import gaussian
                            from scipy.ndimage import binary_fill_holes
                            img = detection.normalize(stack[f_mc])
                            smoothed = gaussian(img, sigma=2.0)
                            filled_mask = flood(smoothed, (y, x), tolerance=0.1)
                            filled_mask = binary_fill_holes(filled_mask)
                            bg_img_array_uint8[filled_mask & (current_cleaned == 0)] = [255, 255, 0] # Yellow

    bg_img = Image.fromarray(bg_img_array_uint8)

    col_img, col_actions = st.columns([3, 1])
    
    brush_size = 5
    with col_actions:
        if action == "🖌️ Brush Draw":
            st.write("### Brush Settings")
            brush_size = st.slider("Brush Size", 1, 30, 5)

    with col_img:
        if action == "🖌️ Brush Draw":
            st.caption("Draw on the image to create a new cell mask. Existing cells will **NOT** be overwritten.")
            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 0, 0.5)",
                stroke_width=brush_size,
                stroke_color="rgba(0, 255, 0, 1.0)",
                background_image=bg_img,
                update_streamlit=True,
                height=bg_img.height,
                width=bg_img.width,
                drawing_mode="freedraw",
                key=f"brush_{canvas_key}",
            )
        elif action == "✂️ Split":
            st.caption("Draw a line across a cell to split it into two. The line gap separates the halves.")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 0, 0.0)",
                stroke_width=st.session_state.get("split_width", 3),
                stroke_color="rgba(255, 255, 0, 1.0)",
                background_image=bg_img,
                update_streamlit=True,
                height=bg_img.height,
                width=bg_img.width,
                drawing_mode="line",
                key=f"split_{canvas_key}",
            )
        else:
            value = streamlit_image_coordinates(bg_img, key=canvas_key)
        
    with col_actions:
        if action == "✂️ Split":
            st.write("### Split Settings")
            split_width = st.slider("Line thickness (px)", 1, 10, 3, key="split_width")
            if canvas_result is not None and canvas_result.image_data is not None:
                split_line_mask = canvas_result.image_data[:, :, 3] > 0
                if split_line_mask.any():
                    current_cleaned = out["cleaned"][f_mc]
                    # Find which cell(s) the line crosses
                    touched_ids = set(np.unique(current_cleaned[split_line_mask])) - {0}
                    if touched_ids:
                        st.write(f"**Line crosses cell(s):** {', '.join(str(i) for i in sorted(touched_ids))}")
                        if st.button("Apply Split", type="primary", width='stretch'):
                            from skimage.morphology import dilation, disk
                            # Dilate the line slightly for a clean gap
                            line_dilated = dilation(split_line_mask.astype(np.uint8),
                                                   disk(max(1, split_width // 2)))
                            for cid in touched_ids:
                                cell_mask = (current_cleaned == cid)
                                # Zero out pixels under the split line
                                current_cleaned[cell_mask & (line_dilated > 0)] = 0
                            # Relabel disconnected components to assign new IDs
                            from skimage.measure import label as sk_label
                            remaining = current_cleaned.copy()
                            # Only relabel the touched cells
                            for cid in touched_ids:
                                cell_region = (out["cleaned"][f_mc] == cid) & (current_cleaned != 0)
                                if not cell_region.any():
                                    continue
                                sub_labels = sk_label(cell_region.astype(np.uint8))
                                unique_sub = [u for u in np.unique(sub_labels) if u > 0]
                                if len(unique_sub) > 1:
                                    # First component keeps original ID, rest get new IDs
                                    for j, sub_id in enumerate(unique_sub):
                                        if j == 0:
                                            remaining[sub_labels == sub_id] = cid
                                        else:
                                            new_id = remaining.max() + 1
                                            remaining[sub_labels == sub_id] = new_id
                            out["cleaned"][f_mc] = remaining
                            st.session_state["outputs"] = out
                            st.session_state["mc_update"] = st.session_state.get("mc_update", 0) + 1
                            st.rerun()
                    else:
                        st.info("Draw a line across a cell to split it.")

        elif action == "🖌️ Brush Draw":
            if canvas_result is not None and canvas_result.image_data is not None:
                drawn_mask = canvas_result.image_data[:, :, 3] > 0
                if drawn_mask.any():
                    st.write("**Drawn stroke detected.**")
                    if st.button("Apply Brush Stroke", type="primary", width='stretch'):
                        current_cleaned = out["cleaned"][f_mc]
                        new_id = current_cleaned.max() + 1 if current_cleaned.max() > 0 else 1
                        
                        # Only apply mask to pixels that are currently empty (0)!
                        current_cleaned[drawn_mask & (current_cleaned == 0)] = new_id
                        
                        out["cleaned"][f_mc] = current_cleaned
                        st.session_state["outputs"] = out
                        st.session_state["mc_update"] = st.session_state.get("mc_update", 0) + 1
                        st.rerun()
                        
        elif value is not None:
            x, y = value["x"], value["y"]
            current_cleaned = out["cleaned"][f_mc]
            current_raw = out["masks"][f_mc]
            
            if 0 <= y < current_cleaned.shape[0] and 0 <= x < current_cleaned.shape[1]:
                cid = current_cleaned[y, x]
                
                if action == "🧽 Erase":
                    if cid > 0:
                        st.write(f"**Selected cell ID:** {cid}")
                        if st.button("Erase this cell", type="primary", width='stretch'):
                            current_cleaned[current_cleaned == cid] = 0
                            out["cleaned"][f_mc] = current_cleaned
                            st.session_state["outputs"] = out
                            st.session_state["mc_update"] = st.session_state.get("mc_update", 0) + 1
                            st.rerun()
                    else:
                        st.info("No cell found at clicked location.")
                        
                elif action == "➕ Add / Recover":
                    if cid == 0:
                        raw_cid = current_raw[y, x]
                        if raw_cid > 0:
                            st.write(f"**Found lost cell** (raw ID {raw_cid})")
                            if st.button("Recover this cell", type="primary", width='stretch'):
                                new_id = current_cleaned.max() + 1 if current_cleaned.max() > 0 else 1
                                current_cleaned[current_raw == raw_cid] = new_id
                                out["cleaned"][f_mc] = current_cleaned
                                st.session_state["outputs"] = out
                                st.session_state["mc_update"] = st.session_state.get("mc_update", 0) + 1
                                st.rerun()
                        else:
                            st.write("**Empty space selected**")
                            if st.button("Add new cell via flood-fill", type="primary", width='stretch'):
                                from skimage.segmentation import flood
                                from skimage.filters import gaussian
                                from scipy.ndimage import binary_fill_holes
                                img = detection.normalize(stack[f_mc])
                                smoothed = gaussian(img, sigma=2.0)
                                filled_mask = flood(smoothed, (y, x), tolerance=0.1)
                                filled_mask = binary_fill_holes(filled_mask)
                                new_id = current_cleaned.max() + 1 if current_cleaned.max() > 0 else 1
                                current_cleaned[filled_mask & (current_cleaned == 0)] = new_id
                                out["cleaned"][f_mc] = current_cleaned
                                st.session_state["outputs"] = out
                                st.session_state["mc_update"] = st.session_state.get("mc_update", 0) + 1
                                st.rerun()
                    else:
                        st.info("Click on empty background to add or recover a cell.")
                        
                elif action == "🔗 Merge":
                    target = st.session_state.get("merge_target")
                    if target is None:
                        if cid > 0:
                            st.write(f"**Target cell ID:** {cid}")
                            if st.button("Set as Target", type="primary", width='stretch'):
                                st.session_state["merge_target"] = cid
                                st.rerun()
                        else:
                            st.info("Click a cell to select it as the target for merging.")
                    else:
                        st.write(f"**Target cell ID:** {target}")
                        if cid > 0 and cid != target:
                            st.write(f"Merge cell {cid} into target?")
                            colA, colB = st.columns(2)
                            if colA.button("Merge", type="primary", width='stretch'):
                                current_cleaned[current_cleaned == cid] = target
                                out["cleaned"][f_mc] = current_cleaned
                                st.session_state["outputs"] = out
                                st.session_state["mc_update"] = st.session_state.get("mc_update", 0) + 1
                                st.rerun()
                            if colB.button("Cancel", width='stretch'):
                                st.session_state["merge_target"] = None
                                st.rerun()
                        elif cid == target:
                            if st.button("Clear Target", width='stretch'):
                                st.session_state["merge_target"] = None
                                st.rerun()
                        else:
                            raw_cid = current_raw[y, x]
                            if raw_cid > 0:
                                st.write(f"Merge recovered cell into target {target}?")
                                colA, colB = st.columns(2)
                                if colA.button("Merge", type="primary", width='stretch'):
                                    current_cleaned[current_raw == raw_cid] = target
                                    out["cleaned"][f_mc] = current_cleaned
                                    st.session_state["outputs"] = out
                                    st.session_state["mc_update"] = st.session_state.get("mc_update", 0) + 1
                                    st.rerun()
                                if colB.button("Cancel", width='stretch'):
                                    st.session_state["merge_target"] = None
                                    st.rerun()
                            else:
                                st.write(f"Merge flood-filled area into target {target}?")
                                colA, colB = st.columns(2)
                                if colA.button("Merge", type="primary", width='stretch'):
                                    from skimage.segmentation import flood
                                    from skimage.filters import gaussian
                                    from scipy.ndimage import binary_fill_holes
                                    img = detection.normalize(stack[f_mc])
                                    smoothed = gaussian(img, sigma=2.0)
                                    filled_mask = flood(smoothed, (y, x), tolerance=0.1)
                                    filled_mask = binary_fill_holes(filled_mask)
                                    current_cleaned[filled_mask & (current_cleaned == 0)] = target
                                    out["cleaned"][f_mc] = current_cleaned
                                    st.session_state["outputs"] = out
                                    st.session_state["mc_update"] = st.session_state.get("mc_update", 0) + 1
                                    st.rerun()
                                if colB.button("Cancel", width='stretch'):
                                    st.session_state["merge_target"] = None
                                    st.rerun()
            
        st.markdown("---")
        st.write("**Save / export your corrections**")
        params_obj = build_params()
        _T = out["cleaned"].shape[0]
        sc1, sc2, sc3 = st.columns(3)
        if sc1.button("💾 Save all permanently", key="save_cleaned_mc",
                      width='stretch', help="Writes the edited cleaned_mask.tiff to the output folder"):
            p = save_cleaned_to_disk(out, params_obj)
            st.success(f"Saved → {p}")
        sc2.download_button(f"⬇ All {_T} masks (.zip)",
                            data=masks_to_zip_bytes(out["cleaned"]),
                            file_name="cleaned_masks.zip", mime="application/zip",
                            width='stretch', key="dl_cleaned_zip",
                            help=f"One TIFF per frame ({_T} files), zipped")
        sc3.download_button("⬇ Stack (.tiff)",
                            data=masks_to_tiff_bytes(out["cleaned"]),
                            file_name="cleaned_mask.tiff", mime="image/tiff",
                            width='stretch', key="dl_cleaned_mc",
                            help=f"Single multi-page TIFF with all {_T} frames")

        st.markdown("---")
        st.write("After manual cleaning, re-run downstream steps to update everything.")
        if st.button("🔄 Update everything (tracking · morphology · intensity · edges)",
                     key="update_downstream_mc", type="primary", width='stretch'):
            with st.spinner("Re-running tracking, morphology, intensity, kinematics + GIFs..."):
                res2, out2 = rerun_downstream(out, res, params_obj, stack)
                st.session_state["res"] = res2
                st.session_state["outputs"] = out2
            st.success("All downstream results updated from your corrections.")
            import time
            time.sleep(1)
            st.rerun()

# ---- Step 3 ----
with tab3:
    import time
    ts = res.get("track_stats", {})
    st.caption(f"tracked_mask.tiff — each cell keeps its id across time")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tracked cells", res["n_tracks"])
    m2.metric("Full-coverage tracks", ts.get("full_coverage", "—"),
              help="present in every frame")
    m3.metric("New-cell events", ts.get("n_new", "—"))

    # ---- upload corrected masks -> auto re-run everything --------------------
    with st.expander("⬆ Upload corrected masks (re-runs tracking + all downstream)"):
        st.caption("Upload a corrected label TIFF (same T×Y×X as the data, e.g. an "
                   "edited cleaned_mask.tiff). Tracking, morphology, intensity and "
                   "edge-kinematics are recomputed automatically.")
        up = st.file_uploader("Corrected mask stack (.tif/.tiff)", type=["tif", "tiff"],
                              key="upload_corrected")
        if up is not None and st.button("Apply uploaded masks", type="primary", key="apply_upload"):
            import tifffile, io
            try:
                arr = tifffile.imread(io.BytesIO(up.read()))
                if arr.ndim == 2:
                    arr = arr[None, ...]
                if arr.shape != out["cleaned"].shape:
                    st.error(f"Shape {arr.shape} doesn't match the data "
                             f"{out['cleaned'].shape}. Upload a stack with one label "
                             f"image per frame.")
                else:
                    out["cleaned"] = arr.astype(np.uint16)
                    with st.spinner("Recomputing tracking · morphology · intensity · edges..."):
                        res2, out2 = rerun_downstream(out, res, build_params(), stack)
                        st.session_state["res"] = res2
                        st.session_state["outputs"] = out2
                    st.success("Uploaded masks applied — everything updated.")
                    import time
                    time.sleep(1)
                    st.rerun()
            except Exception as e:
                st.exception(e)

    # Animated tracking GIF (one GIF-frame per movie-frame, colour = identity)
    gif = res.get("tracking_gif")
    ovl = res.get("overlay_gif")
    if gif and os.path.exists(gif):
        g1, g2 = st.columns(2)
        g1.image(gif, caption="tracking_result.gif — consistent colour = same cell", width='stretch')
        if ovl and os.path.exists(ovl):
            g2.image(ovl, caption="tracking_overlay.gif — outlines on raw frames", width='stretch')

    # trajectory + cells-per-frame
    p1, p2 = st.columns(2)
    if res.get("trajectory_plot") and os.path.exists(res["trajectory_plot"]):
        p1.image(res["trajectory_plot"], width='stretch')
    if res.get("cells_per_frame_plot") and os.path.exists(res["cells_per_frame_plot"]):
        p2.image(res["cells_per_frame_plot"], width='stretch')

    st.markdown("**Frame-by-frame viewer**  ·  _(the GIF above already animates)_")
    play_trk = st.toggle("▶ Auto-play", key="play_trk")
    if play_trk:
        st.session_state["trk_frame"] = (st.session_state.get("trk_frame", 0) + 1) % T
    f = frame_slider_with_buttons("Frame", T - 1, "trk_frame", min(1, T - 1))
    a, b = st.columns(2)
    a.image(norm_rgb(stack[f]), caption=f"frame {f}", width='stretch')
    b.image(label_rgb(out["tracked"][f]), caption=f"tracked ids (colour = id) (frame {f})",
            width='stretch')

    def _trk_frames():
        return [np.hstack([_to_u8(norm_rgb(stack[i])), _to_u8(label_rgb(out["tracked"][i]))])
                for i in range(T)]
    gif_download_widget("tracking", "trk", _trk_frames, fps=4,
                        filename="tracking_viewer.gif", out_dir=res["params"]["out_dir"])

    if play_trk:
        time.sleep(0.25)
        st.rerun()

# ---- Step 4 ----
@st.cache_data(show_spinner=False)
def _morph_map(labimg, base, items, metric):
    from pipeline import viz
    val = dict(items)
    if metric == "n_neighbors":
        rgb, _, _ = viz.neighbor_count_rgb(labimg, val)
    else:
        rgb, _, _ = viz.metric_rgb(labimg, val)
    base3 = np.dstack([base] * 3)
    blend = 0.3 * base3 + 0.7 * rgb
    blend[labimg == 0] = base3[labimg == 0]
    return np.clip(blend, 0, 1)


@st.cache_data(show_spinner=False)
def _neighbor_topology_png(labimg, base, counts_items, draw_edges, neigh_method,
                           neigh_dist, title):
    """Neighbour-topology map: each cell shaded by its #neighbours, with a
    colourbar; optionally overlay the neighbour-adjacency graph (cf. the team's
    visualize_neighbors / neighbour-count map)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import io
    from pipeline import viz

    counts = dict(counts_items)
    rgb, norm, cmo = viz.neighbor_count_rgb(labimg, counts)
    base3 = np.dstack([base] * 3)
    blend = 0.25 * base3 + 0.75 * rgb
    blend[labimg == 0] = base3[labimg == 0]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.imshow(np.clip(blend, 0, 1))
    ax.axis("off")
    ax.set_title(title)

    from skimage.measure import regionprops
    cen = {rp.label: (rp.centroid[1], rp.centroid[0]) for rp in regionprops(labimg)}

    if draw_edges:
        from wound_analysis.intercalation import adjacency_edges
        img = labimg
        if neigh_method == "dilate" and neigh_dist > 0:
            from skimage.segmentation import expand_labels
            img = expand_labels(labimg, distance=neigh_dist)
        for a, b in adjacency_edges(img):
            if a in cen and b in cen:
                ax.plot([cen[a][0], cen[b][0]], [cen[a][1], cen[b][1]],
                        "-", color="#00ffd5", lw=0.5, alpha=0.6)

    # number on each cell = its neighbour count
    for lab_id, (cx, cy) in cen.items():
        n = counts.get(int(lab_id))
        if n is None:
            continue
        ax.text(cx, cy, str(int(n)), color="black", fontsize=6.5, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.65))

    sm = cm.ScalarMappable(norm=norm, cmap=cmo)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="number of neighbours")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


with tab4:
    df = csv_cached(res["morphology_path"])
    st.caption("morphology.csv — per cell, per frame")
    st.dataframe(df, width='stretch', height=260)
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        st.markdown("**Area (µm²)**")
        st.bar_chart(np.histogram(df["area_um2"], bins=20)[0])
    with cc2:
        st.markdown("**Neighbour count**")
        st.bar_chart(df["n_neighbors"].value_counts().sort_index())
    with cc3:
        st.markdown("**Circularity**")
        st.bar_chart(np.histogram(df["circularity"].dropna(), bins=20)[0])
    with cc4:
        st.markdown("**Shape index** (SI*≈3.81)")
        st.bar_chart(np.histogram(df["shape_index"].dropna(), bins=20)[0])
    st.download_button("⬇ morphology.csv", df.to_csv(index=False),
                       "morphology.csv", "text/csv")

    # spatial maps: colour cells by a chosen metric + neighbour count
    st.markdown("**Spatial maps** (cells coloured by value)")
    metric = st.selectbox("Metric", ["area_um2", "circularity", "shape_index",
                                     "aspect_ratio", "elongation", "n_neighbors"])
    mf = frame_slider_with_buttons("Frame", T - 1, "morph_map_frame", 0)
    sub = df[df["frame"] == out["frame_idx"][mf]]
    labimg = out["cleaned"][mf]
    base = detection.normalize(stack[mf])
    items = tuple((int(r.cell_id), float(getattr(r, metric))) for r in sub.itertuples())
    blend = _morph_map(labimg, base, items, metric)   # cached -> cheap on reruns
    im1, im2 = st.columns(2)
    im1.image(blend, caption=f"{metric} — frame {mf}", width='stretch')
    im2.image(label_rgb(labimg, base), caption="cleaned mask", width='stretch')

    # ---- neighbour topology map (cells shaded by #neighbours) ---------------
    st.markdown("**Neighbour topology** — each cell shaded by + labelled with its number of neighbours")
    draw_edges = st.checkbox("Show adjacency graph", True, help="overlay the cell-adjacency network")
    ntf = frame_slider_with_buttons("Frame", T - 1, "ntopo_frame", 0)
    nsub = df[df["frame"] == out["frame_idx"][ntf]]
    nlab = out["cleaned"][ntf]
    nbase = detection.normalize(stack[ntf])
    counts_items = tuple((int(r.cell_id), int(r.n_neighbors)) for r in nsub.itertuples())
    pa = res.get("params", {})
    png = _neighbor_topology_png(nlab, nbase, counts_items, bool(draw_edges),
                                 pa.get("neighbor_method", "touch"),
                                 int(pa.get("neighbor_dist_px", 3)),
                                 f"Neighbour count — frame {ntf}")
    tt1, tt2 = st.columns([3, 2])
    tt1.image(png, width='stretch')
    if len(nsub):
        tt2.metric("Mean neighbours", f"{nsub['n_neighbors'].mean():.1f}")
        tt2.markdown("**Polygon class distribution**")
        tt2.bar_chart(nsub["n_neighbors"].value_counts().sort_index())

# ---- Step 6: Intensity ----
with tab7:
    st.caption("cell_intensity_per_frame.csv — per-cell fluorescence over time "
               "(measured on the tracked masks)")

    # ---- Upload a raw intensity TIFF stack (e.g. a different fluor channel) --
    with st.expander("⬆ Upload intensity TIFF stack (used for cell + boundary intensity)",
                     expanded=("intensity_stack" not in st.session_state)):
        st.caption("Upload the raw fluorescence frames (one multi-page TIFF, or "
                   "many single-frame TIFFs). This channel then drives BOTH the "
                   "per-cell intensity here AND the wound-boundary intensity. "
                   "Frame count must match the analysed frames (or be the full timelapse).")
        if "intensity_stack" in st.session_state:
            st.info(f"Active custom intensity stack: "
                    f"{st.session_state['intensity_stack'].shape} "
                    "— cell + boundary intensity use this.")
        up_int = st.file_uploader("Intensity TIFF(s)", type=["tif", "tiff"],
                                  accept_multiple_files=True, key="upload_intensity_tiff")
        cua, cub = st.columns(2)
        if cua.button("Apply intensity stack", type="primary", key="apply_int_tiff",
                      disabled=not up_int):
            try:
                raw = read_uploaded_tiff_stack(up_int)
                istack = align_intensity_stack(raw, out["frame_idx"], stack.shape[1:])
                st.session_state["intensity_stack"] = istack
                from pipeline import intensity as _intensity_mod, boundary as _bnd_mod
                with st.spinner("Recomputing cell intensity from uploaded stack..."):
                    _ir, _ip, _isp, _ifig = _intensity_mod.run(
                        istack, out["tracked"], build_params(), out["frame_idx"])
                    res["intensity_path"] = _ip
                    res["intensity_summary_path"] = _isp
                    res["intensity_plot"] = _ifig
                    # refresh boundary intensity too, if boundary was already run
                    if "bnd_results" in st.session_state:
                        _br = st.session_state["bnd_results"]
                        _br["int_df"] = _bnd_mod.sample_boundary_intensity(
                            istack, _br["boundary_df"], 5)
                        st.session_state["bnd_results"] = _br
                    st.session_state["res"] = res
                st.success(f"Applied {istack.shape[0]}-frame intensity stack — "
                           "cell + boundary intensity updated.")
                import time as _t
                _t.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Could not apply intensity stack: {e}")
        if cub.button("Reset to default channel", key="reset_int_tiff",
                      disabled=("intensity_stack" not in st.session_state)):
            st.session_state.pop("intensity_stack", None)
            from pipeline import intensity as _intensity_mod
            with st.spinner("Restoring default intensity..."):
                _ir, _ip, _isp, _ifig = _intensity_mod.run(
                    stack, out["tracked"], build_params(), out["frame_idx"])
                res["intensity_path"] = _ip
                res["intensity_summary_path"] = _isp
                res["intensity_plot"] = _ifig
                st.session_state["res"] = res
            st.success("Reverted to the default channel. Re-run boundary analysis "
                       "to refresh its intensity.")
            import time as _t
            _t.sleep(1)
            st.rerun()

    st.markdown("---")
    if res.get("intensity_plot") and os.path.exists(res["intensity_plot"]):
        st.image(res["intensity_plot"], width='stretch')
        idf = csv_cached(res["intensity_path"])
        st.dataframe(idf.head(300), width='stretch', height=240)
        st.download_button("⬇ cell_intensity_per_frame.csv", idf.to_csv(index=False),
                           "cell_intensity_per_frame.csv", "text/csv")
    else:
        st.info("Intensity analysis was not run. Enable it in the sidebar "
                "(⑥ Intensity & GIFs) and re-run.")

    # ---- Boundary intensity heatmap (same as in the ⑤ Wound Boundary tab) ----
    st.markdown("---")
    st.markdown("### 🌈 Boundary Intensity Heatmap")
    _bnd_int_df = None
    if "bnd_results" in st.session_state:
        _bnd_int_df = st.session_state["bnd_results"].get("int_df")
    boundary_intensity_heatmap(_bnd_int_df, key_prefix="int", n_frames=T)

    # ---- Custom intensity CSV upload ----
    st.markdown("---")
    with st.expander("⬆ Upload custom intensity dataset (CSV)"):
        st.caption("Upload your own intensity CSV to override the pipeline-generated data. "
                   "The CSV should have columns: frame, time_min, track_id, mean_intensity "
                   "(same format as cell_intensity_per_frame.csv).")
        custom_csv = st.file_uploader("Custom intensity CSV", type=["csv"],
                                      key="custom_intensity_csv")
        if custom_csv is not None:
            import pandas as pd
            try:
                custom_df = pd.read_csv(custom_csv)
                st.success(f"Loaded custom dataset: {len(custom_df)} rows, "
                           f"columns: {', '.join(custom_df.columns)}")
                st.dataframe(custom_df.head(300), width='stretch', height=240)
                # If it has the right columns, plot it
                if "mean_intensity" in custom_df.columns:
                    plot_col = "mean_intensity"
                elif "intensity" in custom_df.columns:
                    plot_col = "intensity"
                else:
                    plot_col = custom_df.columns[-1]
                    st.caption(f"No 'mean_intensity' column found; plotting '{plot_col}'")
                time_col = None
                for tc in ["time_min", "timeframe", "frame", "t_index"]:
                    if tc in custom_df.columns:
                        time_col = tc
                        break
                if time_col:
                    st.line_chart(custom_df.groupby(time_col)[plot_col].mean())
                st.download_button("⬇ Download custom dataset",
                                   custom_df.to_csv(index=False),
                                   "custom_intensity.csv", "text/csv",
                                   key="dl_custom_intensity")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

# ---- Step 5: Wound Boundary ----
with tab8:
    st.markdown("### 🩹 Wound Boundary Analysis")
    st.caption("Cell cluster inner boundary (wound edge) detection, tracking, "
               "velocity, cell layers, and intensity analysis.")

    # ---- Sidebar-like controls within the tab ----
    with st.expander("⚙️ Boundary analysis parameters", expanded=True):
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            bnd_n_points = st.slider("Seed vertices per frame", 5, 100, 10,
                                     key="bnd_n_points")
        with bc2:
            bnd_min_wound = st.slider("Min wound area (px)", 10, 2000, 50,
                                      key="bnd_min_wound",
                                      help="Smaller = detects the wound in more "
                                           "(later/smaller) frames")
        with bc3:
            bnd_arrow_scale = st.slider("Arrow scale", 1, 20, 8,
                                        key="bnd_arrow_scale")
        with bc4:
            bnd_ring_px = st.slider("Intensity ring (px)", 1, 30, 5,
                                    key="bnd_ring_px")

    run_boundary = st.button("▶ Run Wound Boundary Analysis", type="primary",
                             key="run_boundary")

    if run_boundary:
        from pipeline import boundary as bnd_mod
        params_bnd = build_params()
        # Attach boundary-specific params
        params_bnd.boundary_n_points = bnd_n_points
        params_bnd.boundary_min_wound_area = bnd_min_wound
        params_bnd.boundary_ring_px = bnd_ring_px

        # use an uploaded intensity stack for boundary intensity if provided
        raw_for_bnd = st.session_state.get("intensity_stack", stack)
        with st.spinner("Running wound boundary analysis..."):
            bnd_results = bnd_mod.run(
                out["cleaned"], raw_for_bnd, params_bnd
            )
        st.session_state["bnd_results"] = bnd_results
        st.success(f"Boundary analysis complete — "
                   f"{len(bnd_results['boundary_df'])} boundary points, "
                   f"{len(bnd_results['vel_df'])} velocity records")

    if "bnd_results" in st.session_state:
        bnd_res = st.session_state["bnd_results"]
        bdf = bnd_res["boundary_df"]
        vel_df = bnd_res["vel_df"]
        wa_df = bnd_res["wa_df"]
        layer_df = bnd_res["layer_df"]
        layer_cache = bnd_res["layer_cache"]
        int_df = bnd_res["int_df"]

        # ---- Metrics ----
        bm1, bm2, bm3, bm4 = st.columns(4)
        n_wound_frames = len(bdf["timeframe"].unique()) if not bdf.empty else 0
        bm1.metric("Wound frames", n_wound_frames)
        bm2.metric("Boundary points", len(bdf))
        bm3.metric("Velocity records", len(vel_df))
        bm4.metric("Layer assignments", len(layer_df))

        # ---- 1. Wound Boundary Viewer (playable) ----
        st.markdown("### 🔴 Wound Boundary Viewer")
        import time as _time
        play_bnd = st.toggle("▶ Auto-play", key="play_bnd")
        if play_bnd:
            st.session_state["bnd_frame"] = (st.session_state.get("bnd_frame", 0) + 1) % T
        fbnd = frame_slider_with_buttons("Frame", T - 1, "bnd_frame", 0)

        from pipeline import boundary as bnd_mod
        bnd_col1, bnd_col2 = st.columns(2)
        bnd_rgb = bnd_mod.render_boundary_frame(
            out["cleaned"][fbnd], stack[fbnd], bdf, fbnd, bnd_n_points)
        bnd_col1.image(bnd_rgb, caption=f"Wound contour + seed vertices (frame {fbnd})",
                       width='stretch')
        bnd_col2.image(label_rgb(out["cleaned"][fbnd], detection.normalize(stack[fbnd])),
                       caption=f"Cleaned mask (frame {fbnd})", width='stretch')

        def _bnd_frames():
            return [bnd_mod.render_boundary_frame(out["cleaned"][i], stack[i], bdf,
                                                  i, bnd_n_points) for i in range(T)]
        gif_download_widget("wound boundary", "bnd", _bnd_frames, fps=4,
                            filename="wound_boundary.gif", out_dir=res["params"]["out_dir"])

        if play_bnd:
            _time.sleep(0.3)
            st.rerun()

        # ---- 2. Wound Area Over Time ----
        st.markdown("### 📉 Wound Area Over Time")
        wa_plot = bnd_res.get("wound_area_plot")
        if wa_plot and os.path.exists(wa_plot):
            st.image(wa_plot, width='stretch')
        elif not wa_df.empty:
            st.line_chart(wa_df.set_index("timeframe")["wound_area_px"])
        else:
            st.info("No wound detected in any frame.")

        # ---- 3. Velocity Heatmap ----
        st.markdown("### 🌡️ Velocity Heatmap")
        vel_heatmap = bnd_res.get("velocity_heatmap")
        if vel_heatmap and os.path.exists(vel_heatmap):
            st.image(vel_heatmap, width='stretch')
        elif not vel_df.empty:
            st.info("Velocity heatmap not generated.")

        # ---- 3b. Boundary Intensity Heatmap (Point ID × Timeframe) ----------
        if not int_df.empty:
            st.markdown("### 🌈 Boundary Intensity Heatmap")
            boundary_intensity_heatmap(int_df, key_prefix="bnd", n_frames=T)

        # ---- 4. Velocity Vector Overlay (playable) ----
        if not vel_df.empty:
            st.markdown("### ➡️ Velocity Vector Overlay")
            play_vel = st.toggle("▶ Auto-play", key="play_vel")
            vel_frames = sorted(vel_df["timeframe_from"].unique())
            if vel_frames:
                if play_vel:
                    cur_idx = st.session_state.get("vel_frame_idx", 0)
                    st.session_state["vel_frame_idx"] = (cur_idx + 1) % len(vel_frames)
                vel_fi = st.slider("Transition", 0, len(vel_frames) - 1,
                                   key="vel_frame_idx",
                                   label_visibility="collapsed")
                vel_t = vel_frames[vel_fi]
                vel_png = bnd_mod.render_velocity_frame(
                    out["cleaned"][vel_t], stack[vel_t], bdf, vel_df,
                    vel_t, bnd_arrow_scale)
                st.image(vel_png,
                         caption=f"Velocity arrows (frame {vel_t}, scale ×{bnd_arrow_scale})",
                         width='stretch')

                def _vel_frames():
                    import imageio.v2 as _imageio
                    out_f = []
                    for vt in vel_frames:
                        png = bnd_mod.render_velocity_frame(
                            out["cleaned"][vt], stack[vt], bdf, vel_df, vt, bnd_arrow_scale)
                        out_f.append(np.asarray(_imageio.imread(png)))
                    return out_f
                gif_download_widget("velocity overlay", "vel", _vel_frames, fps=4,
                                    filename="velocity_overlay.gif", out_dir=res["params"]["out_dir"])

                if play_vel:
                    _time.sleep(0.3)
                    st.rerun()

        # ---- 5. Cell Layer Map (playable) ----
        if not layer_df.empty:
            st.markdown("### 🧅 Cell Layer Map")
            play_lay = st.toggle("▶ Auto-play", key="play_lay")
            if play_lay:
                st.session_state["lay_frame"] = (st.session_state.get("lay_frame", 0) + 1) % T
            flay = frame_slider_with_buttons("Frame", T - 1, "lay_frame", 0)
            lay_col1, lay_col2 = st.columns(2)
            lay_rgb = bnd_mod.render_layer_frame(
                out["cleaned"][flay], layer_cache, flay)
            lay_col1.image(lay_rgb, caption=f"Cell layers from wound (frame {flay})",
                          width='stretch')
            lay_col2.image(label_rgb(out["cleaned"][flay], detection.normalize(stack[flay])),
                          caption=f"Cleaned mask (frame {flay})", width='stretch')

            def _lay_frames():
                return [bnd_mod.render_layer_frame(out["cleaned"][i], layer_cache, i)
                        for i in range(T)]
            gif_download_widget("cell layers", "lay", _lay_frames, fps=4,
                                filename="cell_layers.gif", out_dir=res["params"]["out_dir"])

            if play_lay:
                _time.sleep(0.3)
                st.rerun()

        # ---- 6. Boundary intensity over time ------------------------------------
        if not int_df.empty:
            st.markdown("### 💡 Boundary Fluorescence Intensity Over Time")
            import matplotlib.pyplot as _bplt
            fig_bi, ax_bi = _bplt.subplots(figsize=(10, 4))
            mean_int = int_df.groupby("timeframe")["intensity"].mean()
            std_int  = int_df.groupby("timeframe")["intensity"].std().fillna(0)
            t_bi = mean_int.index.values.astype(float)
            ax_bi.fill_between(t_bi, (mean_int - std_int).values,
                               (mean_int + std_int).values,
                               alpha=0.25, color="mediumpurple", label="±1 std")
            ax_bi.plot(t_bi, mean_int.values, "o-", color="mediumpurple",
                       lw=2, ms=4, label="Mean boundary intensity")
            ax_bi.set_xlabel("Frame")
            ax_bi.set_ylabel("Mean fluorescence intensity")
            ax_bi.set_title("Boundary Fluorescence Intensity Over Time")
            ax_bi.legend(); ax_bi.grid(alpha=0.3)
            fig_bi.tight_layout()
            st.pyplot(fig_bi)
            _bplt.close(fig_bi)
            st.download_button("⬇ boundary_intensity.csv", int_df.to_csv(index=False),
                               "boundary_intensity.csv", "text/csv", key="dl_bnd_int2")

        # ---- 6b. Cell intensity in boundary layers ---------------------------
        if (not layer_df.empty and res.get("intensity_path")
                and os.path.exists(res["intensity_path"])):
            st.markdown("### 🔬 Cell Intensity by Layer")
            import pandas as _pd2
            _idf = csv_cached(res["intensity_path"])
            # join layer info onto cell intensity
            _last_frame = layer_df["timeframe"].max()
            _layer_last = layer_df[layer_df["timeframe"] == _last_frame][
                ["cell_label", "layer"]].rename(columns={"cell_label": "track_id"})
            if "track_id" in _idf.columns and not _layer_last.empty:
                _merged_i = _idf.merge(_layer_last, on="track_id", how="inner")
                if not _merged_i.empty and "mean_intensity" in _merged_i.columns:
                    import matplotlib.pyplot as _lplt
                    _tc = "time_min" if "time_min" in _merged_i.columns else "frame"
                    fig_li, ax_li = _lplt.subplots(figsize=(10, 4))
                    _cmap_li = _lplt.cm.viridis
                    _layers = sorted(_merged_i["layer"].unique())
                    for lyr in _layers:
                        _sub = _merged_i[_merged_i["layer"] == lyr]
                        _g = _sub.groupby(_tc)["mean_intensity"]
                        _m = _g.mean(); _s = _g.std().fillna(0)
                        _col = _cmap_li(lyr / max(_layers))
                        ax_li.plot(_m.index, _m.values, lw=1.8, ms=3,
                                   marker="o", color=_col, label=f"Layer {lyr}")
                    ax_li.set_xlabel("Time (min)" if _tc == "time_min" else "Frame")
                    ax_li.set_ylabel("Mean cell intensity")
                    ax_li.set_title("Per-cell Fluorescence by Wound Layer")
                    ax_li.legend(fontsize=7, ncol=4); ax_li.grid(alpha=0.3)
                    fig_li.tight_layout()
                    st.pyplot(fig_li)
                    _lplt.close(fig_li)

        # ---- 6+7. Combined closure dynamics (menu-driven) -------------------
        st.markdown("### 📈 Closure Dynamics — speed · intensity · wound area")
        st.caption("All traces normalised 0–1 so they share one axis. Pick any to "
                   "overlay. Speed, boundary intensity and cell intensity show the "
                   "whole spread: every point/cell (faint), the min–max band, and "
                   "the bold mean. 'Cell intensity (all cells)' uses every tracked "
                   "cell, not just the boundary.")
        import matplotlib.pyplot as _plt
        _spd_col = "speed"
        _series = {}
        if not vel_df.empty:
            _series["Migration speed"] = \
                vel_df.groupby("timeframe_from")[_spd_col].mean()
        if not int_df.empty:
            _series["Boundary fluorescence intensity"] = \
                int_df.groupby("timeframe")["intensity"].mean()
        # per-cell intensity over ALL cells (not just boundary points)
        _cell_idf = None
        if res.get("intensity_path") and os.path.exists(res["intensity_path"]):
            _tmp_ci = csv_cached(res["intensity_path"])
            if {"mean_intensity", "frame", "track_id"}.issubset(_tmp_ci.columns):
                _f2t = {int(f): i for i, f in enumerate(out["frame_idx"])}
                _cell_idf = _tmp_ci.copy()
                _cell_idf["timeframe"] = _cell_idf["frame"].map(_f2t)
                _cell_idf = _cell_idf.dropna(subset=["timeframe"])
                _cell_idf["timeframe"] = _cell_idf["timeframe"].astype(int)
                if not _cell_idf.empty:
                    _series["Cell intensity (all cells)"] = \
                        _cell_idf.groupby("timeframe")["mean_intensity"].mean()
        if not wa_df.empty:
            _series["Wound area"] = wa_df.set_index("timeframe")["wound_area_px"]

        if _series:
            _opts = list(_series.keys())
            _chosen = st.multiselect("Traces to plot", _opts, default=_opts,
                                     key="closure_traces")
            _colors = {"Migration speed": "green",
                       "Boundary fluorescence intensity": "mediumpurple",
                       "Cell intensity (all cells)": "crimson",
                       "Wound area": "steelblue"}
            _style = st.radio("Spread style", ["Mean ± std", "All points"],
                              horizontal=True, key="closure_spread_style",
                              help="'Mean ± std' = clean shaded band (slide style); "
                                   "'All points' = every individual point faint")
            if _chosen:
                fig_cd, ax_cd = _plt.subplots(figsize=(10, 4))

                def _plot_all_points_and_mean(ax, x, piv_raw, mean_raw, col, label):
                    """Normalise a wide pivot + its mean, then render either a clean
                    mean ± std band or every individual column faint, overlaid with
                    the bold mean line."""
                    lo = float(np.nanmin(piv_raw.values))
                    hi = float(np.nanmax(piv_raw.values))
                    rng = (hi - lo) or 1.0
                    piv_n = (piv_raw - lo) / rng
                    mean_n = ((mean_raw - lo) / rng).reindex(piv_raw.index)
                    if _style == "Mean ± std":
                        sd = piv_n.std(axis=1).fillna(0)
                        ax.fill_between(x, (mean_n - sd).values,
                                        (mean_n + sd).values, color=col, alpha=0.22)
                        lab = f"{label} (mean ± std)"
                    else:
                        for col_id in piv_n.columns:
                            ax.plot(x, piv_n[col_id].values, color=col,
                                    alpha=0.18, lw=0.7)
                        ax.fill_between(x, piv_n.min(axis=1).values,
                                        piv_n.max(axis=1).values,
                                        color=col, alpha=0.10)
                        lab = f"{label} (mean · {len(piv_n.columns)} pts)"
                    ax.plot(x, mean_n.values, color=col, lw=2.5, marker="o",
                            ms=4, label=lab)

                for name in _chosen:
                    s = _series[name].sort_index()
                    col = _colors.get(name, "gray")
                    if name.startswith("Migration speed"):
                        piv = vel_df.pivot_table(index="timeframe_from",
                                                 columns="point_id",
                                                 values=_spd_col).sort_index()
                        _plot_all_points_and_mean(ax_cd,
                            piv.index.values.astype(float), piv, s, col, name)
                    elif name.startswith("Boundary fluorescence"):
                        # pivot: one column per boundary point id
                        piv = int_df.pivot_table(index="timeframe",
                                                 columns="point_id",
                                                 values="intensity").sort_index()
                        _plot_all_points_and_mean(ax_cd,
                            piv.index.values.astype(float), piv, s, col, name)
                    elif name.startswith("Cell intensity"):
                        # pivot: one column per cell (track_id) -> all cells shown
                        piv = _cell_idf.pivot_table(index="timeframe",
                                                    columns="track_id",
                                                    values="mean_intensity").sort_index()
                        _plot_all_points_and_mean(ax_cd,
                            piv.index.values.astype(float), piv, s, col, name)
                    else:
                        # Wound area — one value per frame, plot as scatter + line
                        lo, hi = float(s.min()), float(s.max())
                        rng = (hi - lo) or 1.0
                        xw = s.index.values.astype(float)
                        yn = ((s - lo) / rng).values
                        ax_cd.scatter(xw, yn, color=col, s=20, zorder=5, alpha=0.6)
                        ax_cd.plot(xw, yn, color=col, lw=2.5, marker="o",
                                   ms=4, label=name)

                ax_cd.set_xlabel("Frame")
                ax_cd.set_ylabel("Normalised value (0–1)")
                ax_cd.set_ylim(-0.05, 1.15)
                ax_cd.grid(alpha=0.3)
                ax_cd.legend(loc="upper right", fontsize=8)
                fig_cd.tight_layout()
                st.pyplot(fig_cd)
                _plt.close(fig_cd)
            else:
                st.info("Select at least one trace to plot.")
        else:
            st.info("No speed / intensity / wound-area data to plot yet.")

        # ---- 8. Multi-frame Preview ----
        if not bdf.empty:
            st.markdown("### 🖼️ Multi-frame Preview")
            show_frames = sorted(bdf["timeframe"].unique())[:6]
            cols = st.columns(len(show_frames))
            for col, t_prev in zip(cols, show_frames):
                prev_rgb = bnd_mod.render_boundary_frame(
                    out["cleaned"][t_prev], stack[t_prev], bdf, t_prev,
                    bnd_n_points)
                col.image(prev_rgb, caption=f"t={t_prev}", width='stretch')

        # ---- Downloads ----
        st.markdown("---")
        st.markdown("**Downloads**")
        dc1, dc2, dc3, dc4 = st.columns(4)
        if not bdf.empty:
            dc1.download_button("⬇ boundary_points.csv",
                                bdf.to_csv(index=False),
                                "cluster_boundary_points.csv", "text/csv",
                                key="dl_bnd_pts")
        if not vel_df.empty:
            dc2.download_button("⬇ boundary_velocity.csv",
                                vel_df.to_csv(index=False),
                                "cluster_boundary_velocity.csv", "text/csv",
                                key="dl_bnd_vel")
        if not layer_df.empty:
            dc3.download_button("⬇ cell_layers.csv",
                                layer_df.to_csv(index=False),
                                "cell_layers.csv", "text/csv",
                                key="dl_bnd_layers")
        if not int_df.empty:
            dc4.download_button("⬇ boundary_intensity.csv",
                                int_df.to_csv(index=False),
                                "boundary_intensity.csv", "text/csv",
                                key="dl_bnd_int")
    else:
        st.info("Press **Run Wound Boundary Analysis** above to detect wound edges, "
                "compute velocities, cell layers, and generate interactive plots.")

# ---- Files ----
with tab6:
    st.caption(f"All outputs written to `{res['params']['out_dir']}/`")
    od = res["params"]["out_dir"]
    files = [res.get(k) for k in ("mask_path", "cleaned_path", "tracked_path",
                                  "morphology_path", "edge_velocity_path",
                                  "edge_velocity_plot", "tracking_gif", "overlay_gif",
                                  "trajectory_plot", "cells_per_frame_plot",
                                  "intensity_path", "intensity_summary_path",
                                  "intensity_plot")]
    files += [os.path.join(od, "track_summary.csv"),
              os.path.join(od, "traced_result_2D.npy")]
    # Include boundary analysis outputs if they exist
    for bnd_file in ["cluster_boundary_points.csv", "cluster_boundary_velocity.csv",
                     "cell_layers.csv", "boundary_intensity.csv", "wound_area.csv",
                     "wound_area_plot.png", "velocity_heatmap.png",
                     "speed_vs_intensity.png"]:
        bp = os.path.join(od, bnd_file)
        if os.path.exists(bp):
            files.append(bp)
    for fp in files:
        if fp and os.path.exists(fp):
            with open(fp, "rb") as fh:
                st.download_button(f"⬇ {os.path.basename(fp)}", fh.read(),
                                   os.path.basename(fp), key=fp)
