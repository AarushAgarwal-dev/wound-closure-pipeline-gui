# 🔬 EMBRIO Wound-Closure Pipeline — Researcher's Guide

A step-by-step guide for using the Zebrafish Wound-Closure Analysis tool.
No programming experience required.

---

## What Does This Tool Do?

This tool analyses time-lapse microscopy images of zebrafish tailfin wound closure. It automatically:

1. **Segments** individual cells in each image frame
2. **Cleans** the segmentation to remove noise and artefacts
3. **Tracks** cells across time (giving each cell a persistent ID)
4. **Measures cell shapes** — area, perimeter, circularity, neighbour count
5. **Measures wound-edge velocity** — how fast the wound edge moves
6. **Measures fluorescence intensity** — per-cell brightness over time
7. **Detects the wound boundary** — tracks the wound edge and computes closure speed
8. Produces downloadable **CSV tables, plots, and animated GIFs**

---

## Installation (One-Time Setup)

### Step 1: Install Python

Download Python 3.10 or newer from [python.org](https://www.python.org/downloads/).

During installation, **check the box that says "Add Python to PATH"** — this is important!

### Step 2: Download the Tool

Download or clone this folder (the `EMBRIO` folder) to your computer. Place it somewhere easy to find, like your Desktop or Documents folder.

### Step 3: Install Dependencies

1. Open a **Terminal** (Mac/Linux) or **Command Prompt** (Windows)
2. Navigate to the EMBRIO folder:
   ```
   cd path/to/EMBRIO
   ```
   For example, if it's on your Desktop:
   - **Windows:** `cd C:\Users\YourName\Desktop\EMBRIO`
   - **Mac:** `cd ~/Desktop/EMBRIO`

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   This may take a few minutes. You'll see text scrolling — that's normal.

> **Note:** The Cellpose segmentation engine (deep learning) is optional. If it fails to install, the tool automatically falls back to a simpler watershed method that works without it.

---

## Launching the Tool

1. Open your Terminal / Command Prompt
2. Navigate to the EMBRIO folder (same as above)
3. Run:
   ```
   streamlit run app.py
   ```
4. A browser window will open automatically showing the tool. If it doesn't, open your browser and go to `http://localhost:8501`

---

## Preparing Your Data

### What format do I need?

- **Single-channel TIFF images** (one file per time point)
- Named so they sort in time order (e.g., `frame_001.tif`, `frame_002.tif`, ...)
- If you have 3D stacks, max-project them to 2D first

### Where do I put my data?

Place your TIFF files in a folder. By default, the tool looks for a folder called `Wound` inside the EMBRIO directory. You can change this in the sidebar.

---

## Using the Tool — Tab by Tab

### ⚙️ Sidebar (Left Panel)

The sidebar contains all adjustable parameters. The most important ones:

| Parameter | What it does | Recommended |
|-----------|-------------|-------------|
| **Data folder** | Where your TIFF images are | `Wound` (default) |
| **Max frames** | Limits how many frames to process | Start with 8 to test, then set to 0 for all |
| **Start frame** | First naturally sorted raw frame to load | `0`, or `1` if masks begin at T001 |
| **Backend** | Segmentation method | `cellpose` (more accurate) or `watershed` (faster) |
| **Cell diameter** | Expected cell size in pixels | 30 (adjust if cells are much larger/smaller) |

After setting parameters, click **▶ RUN PIPELINE**.

Files containing `_T###` or `_t###` are sorted by that number. For example,
`surface_max_T000.tif` through `surface_max_T049.tif` should use the file
pattern `surface_max_T*.tif`. **Max frames** takes the first N sequential files;
use `0` when every raw frame has a corresponding mask frame. If a 49-frame mask
stack corresponds to `T001–T049`, set **Start frame** to `1` and **Max frames**
to `49`; for `T000–T048`, use Start frame `0`.

### Skip Cellpose and begin with an existing segmented mask

If you already have label masks, open **Start at Step 2 — load segmented masks
for manual editing** on the main page. Upload a multi-page TIFF stack or several
single-frame TIFF masks, then click **Load masks into Step 2**. The number and
size of mask frames must match the selected raw images. The masks are loaded
unchanged; automatic cleaning and all downstream analyses are skipped.

If you want this GUI to perform segmentation but stop before cleaning and
tracking, click **Segment only → Step 2** in the sidebar. After it finishes,
download `mask.tiff` from the Segmentation tab or continue directly with the
manual drawing tools.

In **② Manual Cleaning**, use the erase, recover, merge, split, and brush tools.
Click **Stack (.tiff)** to download the edited mask stack. Use **Update
everything** only if you also want to continue with downstream analyses.

---

### ① Segmentation Tab

Shows three images side by side for each frame:
- **Original** — your raw microscopy image
- **Mask** — the computer's first attempt at finding cells
- **Cleaned mask** — the refined segmentation

Use the **frame slider** or **◀ ▶ buttons** to step through frames.
Toggle **▶ Auto-play** to watch all frames like a movie.

---

### ② Manual Cleaning Tab

If the automatic segmentation made mistakes, you can fix them here by hand.

**Tools available:**

| Tool | What it does |
|------|-------------|
| 🧽 **Erase** | Click a cell to remove it from the mask |
| ➕ **Add / Recover** | Click empty space to recover a missed cell |
| 🔗 **Merge** | Combine two cell fragments into one (click target first, then the cell to merge) |
| ✂️ **Split** | Draw a line across a cell to split it into two separate cells |
| 🖌️ **Brush Draw** | Free-draw a new cell shape |

**How to split a cell:**
1. Select ✂️ Split mode
2. Draw a line across the cell you want to divide
3. Adjust the line thickness if needed
4. Click "Apply Split"

The drawn split line is extended through the cell(s) it touches, so the drag
does not need to reach the exact outer boundary. Use **Cut thickness** for
difficult shapes and **Clear split line** to redraw before applying. Split
always creates exactly two labels: the line band and any small fragments are
merged into one of the two sides rather than retained as extra cells.

Brush Draw remains a freehand click-and-drag tool. Increase **Brush Size** for
larger corrections, enable **Paint over existing cells** when the new cell must
replace existing labels, or use **Clear brush stroke** to start over.

5. The cell becomes two cells with different colours

**Saving your corrections:**
- 💾 **Save all permanently** — writes to disk
- ⬇ **All masks (.zip)** — downloads one TIFF per frame, zipped
- ⬇ **Stack (.tiff)** — downloads a single multi-frame TIFF

The Segmentation tab has a separate **Raw segmented stack (.tiff)** download.
This saves `mask.tiff` before manual cleaning. You can later upload that file
through **Start at Step 2** and continue editing without rerunning Cellpose.

- 🔄 **Update everything** — re-runs tracking and analysis with your corrections

The save and download controls are below the editor. Large images are reduced
to fit the browser while editing, but all edits and downloaded TIFFs retain the
original mask dimensions. You can safely switch between Brush Draw, Split, and
the click-based tools; each tool uses its own editor state.

Use **Editor display size (%)** to shrink or enlarge the manual editor from 50%
to 200% of its fitted size. Expanded views automatically use the full page
width. The display is never enlarged past the source image resolution, and the
size control does not change edit coordinates, brush size, split thickness, or
the dimensions of saved/downloaded masks.

---

### ③ Tracking Tab

Shows how cells are tracked across time. Each cell keeps a consistent colour/ID.

- **Tracking GIF** — animated view of tracked cells
- **Trajectory plot** — lines showing where each cell moved
- **Cells per frame** — graph of how many cells were found per frame

You can also **upload corrected masks** here if you edited them externally.

---

### ④ Morphology Tab

Quantitative measurements of cell shapes:

| Metric | What it measures |
|--------|-----------------|
| **Area** | Cell size in µm² |
| **Circularity** | How round the cell is (1.0 = perfect circle) |
| **Shape index** | Perimeter/√Area (≈3.81 for a regular hexagon) |
| **Aspect ratio** | How elongated the cell is |
| **Neighbour count** | How many other cells touch this one |

**Spatial maps** show these metrics colour-coded on the image.
**Neighbour topology** shows the cell-adjacency network.

Download the full data table as `morphology.csv`.

---

### ⑤ Edge Kinematics Tab

Measures how a cell's edge moves over time:
- Samples 100 points around a chosen cell's boundary
- Tracks each point frame-to-frame
- Computes velocity vectors (speed and direction)

The most persistent cell is chosen automatically, or you can specify a cell ID in the sidebar.

---

### ⑥ Intensity Tab

Measures fluorescence brightness per cell over time:
- **Population dynamics** — mean ± SD intensity across all cells
- **Per-cell trajectories** — individual cell brightness traces
- **Intensity heatmap** — all cells × all frames as a colour map
- **Spatial intensity map** — cells coloured by brightness

**Upload custom data:** Use the expandable section at the bottom to upload your own intensity CSV if you have measurements from another tool.

---

### ⑦ Wound Boundary Tab

Detects and tracks the **wound edge** (inner boundary of the cell cluster):

1. **Set parameters:**
   - **Seed vertices** — how many tracking points on the wound edge
   - **Min wound area** — minimum hole size to be considered a wound
   - **Arrow scale** — visual size of velocity arrows
   - **Intensity ring** — how far around each vertex to sample brightness

2. **Click "Run Wound Boundary Analysis"**

3. **Interactive viewers** (all have ▶ Auto-play):
   - 🔴 **Wound Boundary Viewer** — contour + seed vertices on each frame
   - 📉 **Wound Area Over Time** — how the wound shrinks
   - 🌡️ **Velocity Heatmap** — per-vertex speed over time
   - ➡️ **Velocity Vector Overlay** — arrows showing closure direction/speed
   - 🧅 **Cell Layer Map** — cells coloured by distance from wound
   - 💡 **Boundary Intensity** — fluorescence at the wound edge
   - 📊 **Speed vs Intensity** — do brighter edges close faster?
   - 🖼️ **Multi-frame Preview** — first 6 frames with tracked boundary

4. **Download** all data as CSV files.

---

### 📦 Files Tab

One-click download of every output file the pipeline produced:
- TIFF mask stacks
- CSV data tables
- PNG plots and GIF animations
- Boundary analysis outputs

---

## Troubleshooting

### The drawing canvas does not load

If Streamlit says it is having trouble loading `streamlit_drawable_canvas`,
reinstall the canvas frontend from the project folder and restart the GUI:

```bash
pip install --force-reinstall --no-deps streamlit-drawable-canvas==0.9.3
streamlit run app.py
```

This replaces missing or corrupted component frontend assets without changing
Streamlit or the rest of the scientific Python environment.

| Problem | Solution |
|---------|----------|
| **"cellpose not installed"** | The tool will automatically use the simpler watershed method instead. This is fine for most uses. |
| **Cellpose will not run on this computer** | Segment elsewhere, then use **Start at Step 2** to upload and clean the label masks directly. |
| **Cellpose is very slow** | Check the device caption in Segmentation. CPU-only processing can take about 45 seconds per 512×512 frame; reduce **Max frames** while testing. |
| **App disconnects or freezes** | Reduce **Max frames** in the sidebar. Cellpose is slow on CPU (~15 seconds/frame). |
| **No cells detected** | Try adjusting the **Cell diameter** slider, or switch to the **watershed** backend. |
| **Wound not detected** | Increase the **Min wound area** slider, or check that the wound is fully interior (not touching the image edge). |
| **Browser shows error** | Refresh the page. If the error persists, check the terminal for error messages. |
| **Installation fails** | Make sure Python 3.10+ is installed and "Add to PATH" was checked. Try `pip install --upgrade pip` first. |

---

## Glossary

| Term | Meaning |
|------|---------|
| **Segmentation** | The process of identifying individual cells in an image |
| **Mask** | A label image where each cell has a unique number (0 = background) |
| **Tracking** | Following the same cell across multiple time frames |
| **Cellpose** | A deep-learning tool for cell segmentation (more accurate, slower) |
| **Watershed** | A classical image-processing method for segmentation (faster, less accurate) |
| **Morphology** | Quantitative measurements of cell shape |
| **Circularity** | 4π × Area / Perimeter² — 1.0 for a perfect circle |
| **IoU** | Intersection over Union — measures overlap between cells in adjacent frames |
| **BFS layers** | Breadth-first search layers — concentric rings of cells outward from the wound |
| **µm** | Micrometre — one millionth of a metre |
| **px** | Pixel — the smallest unit of a digital image |

---

## Getting Help

If you encounter issues not covered here, check:
1. The terminal window where you launched `streamlit run app.py` — it often shows helpful error messages
2. The [README.md](README.md) file for technical details
3. Contact the EMBRIO Team 5 developers
