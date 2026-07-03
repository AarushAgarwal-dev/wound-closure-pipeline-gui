# ☁️ Running the GUI on Vast.ai (GPU)

This guide walks through renting a GPU box on [Vast.ai](https://vast.ai) and
running the wound-closure GUI there so **Cellpose-SAM** segmentation runs on a
real GPU (seconds/frame instead of ~15 s/frame on a laptop CPU).

You do **not** need a GPU to use this tool — the watershed backend runs anywhere.
Use Vast.ai only when you want fast, accurate Cellpose segmentation on a full
time-lapse.

---

## 1. Rent an instance

1. Sign in at [vast.ai](https://vast.ai) and add credit.
2. Go to **Console → Search / Rent**.
3. **Image / template:** choose a **PyTorch NGC** image so CUDA PyTorch is
   pre-installed and matches the GPU:
   ```
   nvidia/pytorch:24.10-py3        (or any recent nvidia/pytorch:*-py3)
   ```
   > Using the NGC PyTorch image matters — `setup_vast.sh` installs Cellpose
   > **without** reinstalling PyTorch, so the GPU build already in the image is
   > preserved (important for newer GPUs like the RTX 50-series / Blackwell).
4. **GPU:** any single modern card is plenty — RTX 3090 / 4090 / 5060 / A10, etc.
   ~12 GB VRAM is comfortable for 512×512 frames.
5. **Disk:** 20 GB+ (Cellpose downloads a ~1.1 GB model on first use).
6. **Ports:** make sure at least one TCP port is exposed/mapped (e.g. `8080`).
   Vast shows the external mapping under the instance's **"Open Ports"**.
7. Click **Rent**. Wait for the instance to show **Running**.

---

## 2. Connect to the box

From the instance card, copy the **SSH** command. Two common ways to reach the
Streamlit UI from your laptop:

**Option A — SSH tunnel (simplest, private):**
```bash
ssh -L 8080:localhost:8080 -p <PORT> root@<HOST>
```
This forwards your laptop's `localhost:8080` to the box.

**Option B — Vast port mapping:** use the external `host:port` Vast assigns to
the container's `8080` (shown in the instance's port list). No tunnel needed.

---

## 3. Get the code onto the box

In the instance shell:

```bash
cd /workspace
git clone <your-repo-url>
cd wound-closure-gui
```

No GitHub? Upload the folder instead — from your **laptop**:
```bash
scp -P <PORT> -r wound-closure-gui root@<HOST>:/workspace/
```

The bundled `Wound/` sample data comes with it, so you can test immediately.

---

## 4. Install dependencies

```bash
bash setup_vast.sh
```

This script:
- verifies the GPU + existing CUDA PyTorch (and does **not** replace it),
- installs Cellpose with `--no-deps` (so pip can't swap in a CPU torch),
- installs the remaining Python deps (scikit-image, streamlit, pandas, …),
- runs a tiny CUDA matmul to prove the GPU kernel works,
- prints the exact launch command at the end.

If CUDA reports unavailable at the end, re-check that you picked an NGC PyTorch
image and that the instance actually has a GPU attached.

---

## 5. Launch the GUI

**Direct (matches an SSH tunnel / port map on 8080):**
```bash
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```
Then open `http://localhost:8080` on your laptop (Option A) or the mapped Vast
URL (Option B).

**Or get an instant public HTTPS link** (Cloudflare quick tunnel — no port
mapping needed):
```bash
bash serve_web.sh 8501
```
Watch the output for a line like:
```
https://<random-name>.trycloudflare.com
```
Open that in any browser. `serve_web.sh` starts Streamlit if it isn't already
running and keeps it alive even if the tunnel drops.

> **Tip:** run the server in `tmux` / `screen` (or `nohup`) so it survives an SSH
> disconnect:
> ```bash
> tmux new -s app
> streamlit run app.py --server.port 8080 --server.address 0.0.0.0
> # detach with Ctrl-b then d; reattach with: tmux attach -t app
> ```

---

## 6. Use the GPU in the app

Once the UI is open:

1. Sidebar → **Segmentation**: tick **Use GPU**, backend = **cellpose**.
2. Sidebar → **Input/Output**: set **Max frames = 0** to process all frames
   (or point **Data folder** at your uploaded dataset).
3. Press **▶ RUN PIPELINE**.

First run downloads the Cellpose-SAM model (~1.1 GB) — subsequent runs are fast.

---

## 7. Get your results off the box

- **In-app:** every tab and the **📦 Files** tab have download buttons
  (CSV / TIFF / GIF) — easiest for individual outputs.
- **Bulk copy** the whole output folder to your laptop:
  ```bash
  scp -P <PORT> -r root@<HOST>:/workspace/wound-closure-gui/pipeline_out ./
  ```

---

## 8. Stop the instance 💸

Vast bills while the instance runs. When finished, **Stop** or **Destroy** it
from the Vast console. Copy your results off first (step 7) — a destroyed
instance's disk is gone.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `torch.cuda.is_available()` is `False` after setup | You didn't rent an NGC PyTorch image, or the box has no GPU. Re-rent with `nvidia/pytorch:*-py3`. |
| Cloudflare tunnel keeps failing | The host may block Cloudflare — use the SSH tunnel or Vast port mapping instead. |
| Streamlit "connection refused" from laptop | Ensure `--server.address 0.0.0.0` and that the port is tunneled/mapped. |
| Out-of-memory during Cellpose | Lower **Max frames**, or rent a GPU with more VRAM. |
| Model re-downloads every run | The model cache lives on the instance; it persists only while the instance exists. |
