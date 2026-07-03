# 🔬 Wound-Closure Analysis GUI

A self-contained **Streamlit GUI** for quantifying cell dynamics in time-lapse
microscopy of zebrafish tailfin wound closure — segmentation, tracking,
morphology, edge kinematics, fluorescence intensity, and wound-boundary
kinematics — with one-click CSV / TIFF / GIF export.

Built for the EMBRIO Design Challenge (Team 5). Runs on **any folder of TIFF
frames**, on a laptop (CPU) or a GPU box (e.g. [Vast.ai](https://vast.ai)).

<!-- Add a screenshot/GIF here once you have one:
![screenshot](docs/screenshot.png)
-->

---

## ✨ What it does

| Stage | Output |
|-------|--------|
| **Segmentation** — Cellpose-SAM (GPU) or watershed (CPU fallback) | `mask.tiff`, `cleaned_mask.tiff` |
| **Manual cleaning** — erase / add / merge / split / brush by hand | corrected masks |
| **Tracking** — Hungarian bipartite matching + gap-closing | `tracked_mask.tiff`, `track_summary.csv`, tracking GIFs |
| **Morphology** — area, perimeter, circularity, shape index, aspect ratio, neighbours | `morphology.csv` + spatial maps |
| **Edge kinematics** — 100-point boundary sampling → velocity (u, v) | `edge_velocity.csv` + plots |
| **Intensity** — per-cell fluorescence over time | `cell_intensity_per_frame.csv` + heatmaps |
| **Wound boundary** — inner-contour tracking, closure speed, per-vertex velocity | boundary CSVs + overlays |

All of it is exposed in tabbed GUI with live progress, frame viewers,
auto-play, and downloads.

---

## 🚀 Quick start (local, laptop)

Requires **Python 3.10+**.

```bash
git clone <your-repo-url>
cd wound-closure-gui

pip install -r requirements.txt      # Cellpose is optional — see notes below
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. The app starts pointed at the
bundled sample dataset in [`Wound/`](Wound/) — press **▶ RUN PIPELINE** to see
it end-to-end immediately.

> **No GPU?** Leave the backend on **watershed** (fast, no download) or set
> **Max frames = 8** while experimenting with Cellpose on CPU (~15 s/frame).

---

## 📁 Run it on *your own* data (any folder)

The GUI is not tied to the sample data. In the **left sidebar → Input/Output**:

1. **Data folder** — path to a folder of your TIFF frames. Absolute
   (`D:/experiments/wound_A`) or relative to where you launched the app.
2. **File pattern** — glob for the frames, e.g. `*.tif` or `MAX_*.tif`.
3. **Output folder** — where results are written (default `pipeline_out`).

Then press **▶ RUN PIPELINE**. That's it — nothing is hard-coded.

**Data format**
- Single-channel TIFF, **one file per time point** (membrane / fluorescence marker).
- Filenames must sort in time order, e.g. `frame_0001.tif`, `frame_0002.tif`, …
- For 3-D stacks, **max- or sum-project to 2-D first**.
- Pixel size (µm/px) and frame interval (s) are read automatically from ImageJ
  TIFF tags when present; otherwise the sidebar fallbacks
  (`0.3448 µm/px`, `31.09 s/frame`) are used — edit them to match your microscope.

You can also **upload** TIFFs directly in the Tracking / Manual-cleaning tabs
without touching the folder path.

📖 A full, non-programmer, tab-by-tab walkthrough is in **[GUIDE.md](GUIDE.md)**.

---

## ☁️ Run on Vast.ai (GPU, for Cellpose)

Cellpose-SAM is much faster and more accurate on a GPU. See the dedicated
**[VASTAI.md](VASTAI.md)** for the full walkthrough. Short version:

1. **Rent an instance** on [Vast.ai](https://vast.ai) using a **PyTorch NGC**
   image (e.g. `nvidia/pytorch:24.xx-py3`). A single RTX 3090/4090/5060 is plenty.
2. **Get the code onto the box** (from the instance shell):
   ```bash
   cd /workspace
   git clone <your-repo-url>
   cd wound-closure-gui
   ```
3. **Install** (keeps the image's CUDA PyTorch, adds Cellpose without swapping it):
   ```bash
   bash setup_vast.sh
   ```
4. **Launch the GUI** on a port your SSH tunnel / Vast port-map exposes:
   ```bash
   streamlit run app.py --server.port 8080 --server.address 0.0.0.0
   ```
   or get an instant public HTTPS link via a Cloudflare tunnel:
   ```bash
   bash serve_web.sh 8501
   ```
5. **Open it** at `http://localhost:8080` (via `ssh -L 8080:localhost:8080 …`)
   or the printed `https://<name>.trycloudflare.com` link.
6. In the app: **Segmentation → tick "Use GPU"**, set **Max frames = 0** (all).

---

## 🖥️ Command-line (no GUI)

```bash
# diagram-driven pipeline
python -m pipeline.run --backend watershed --max-frames 8 --out pipeline_out
python -m pipeline.run --backend cellpose                       # GPU / Cellpose

# original objective-oriented analysis (reads ./Wound, writes ./results)
python run_analysis.py --data Wound --out results
```

---

## 🗂️ Repository layout

```
wound-closure-gui/
├── app.py                 # Streamlit GUI (entry point)
├── pipeline/              # diagram-driven Phase 1/2 workflow
│   ├── segment.py         #   Cellpose-SAM / watershed segmentation
│   ├── clean.py           #   mask cleaning
│   ├── track.py           #   Hungarian tracking + gap-closing
│   ├── morphology.py      #   shape metrics + neighbours
│   ├── kinematics.py      #   edge-point velocity
│   ├── intensity.py       #   per-cell fluorescence
│   ├── boundary.py        #   wound-boundary tracking
│   ├── viz.py / run.py / config.py
├── wound_analysis/        # objective-oriented analysis + shared I/O
│   ├── io_utils.py        #   TIFF loading + µm/px & s/frame calibration
│   ├── detection.py       #   tissue mask, wound-centre, radial edge
│   ├── segmentation.py    #   watershed shape analysis
│   ├── edge_velocity.py / intercalation.py / plotting.py
├── run_analysis.py        # CLI driver for wound_analysis
├── Wound/                 # ✅ bundled sample dataset (44 frames, 512×512)
├── .streamlit/config.toml # Streamlit server settings
├── setup_vast.sh          # one-shot GPU-box setup
├── serve_web.sh           # public HTTPS tunnel helper
├── requirements.txt
├── GUIDE.md               # non-programmer GUI walkthrough
└── VASTAI.md              # detailed Vast.ai instructions
```

---

## 🧾 Sample dataset

[`Wound/`](Wound/) contains a real confocal membrane time-lapse:
**44 frames · 512×512 · 0.3448 µm/px · 31.1 s/frame (≈22 min)**, single-channel
(561 nm), MAX-projected. It ships so the app works out-of-the-box — delete it or
point the sidebar elsewhere once you're using your own data.

---

## ⚙️ Notes & troubleshooting

- **Cellpose is optional.** If `cellpose` isn't installed the app automatically
  falls back to watershed. On first GPU use Cellpose-SAM downloads a ~1.1 GB model.
- **App disconnects / freezes** → lower **Max frames**; CPU Cellpose is slow.
- **No cells detected** → adjust **Cell diameter**, or switch to **watershed**.
- **Wound not detected** → raise **Min wound area**; the wound must be fully
  interior (not touching the image border).

More fixes in [GUIDE.md → Troubleshooting](GUIDE.md#troubleshooting).
