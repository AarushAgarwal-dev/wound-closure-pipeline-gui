#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_vast.sh — set up the EMBRIO wound pipeline on a Vast.ai PyTorch NGC box
#
#   RTX 5060 = Blackwell (sm_120) -> needs the CUDA torch that ships in the NGC
#   image. We install Cellpose with --no-deps so pip never swaps that torch for
#   a CPU build. Run this from the project root (where app.py lives).
# ---------------------------------------------------------------------------
set -e

echo "==> GPU / driver"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo "==> Existing PyTorch (must stay CUDA — do NOT reinstall torch)"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"

echo "==> Installing Cellpose WITHOUT touching torch/torchvision"
pip install --no-deps "cellpose>=4.0"

echo "==> Installing remaining dependencies (no torch here)"
pip install \
  scikit-image tifffile imagecodecs streamlit pandas scikit-learn imageio matplotlib \
  natsort fastremap fill-voids roifile segment-anything numba tqdm \
  opencv-python-headless

echo "==> Manual-cleaning UI components (no-deps so they can't downgrade Streamlit)"
pip install --no-deps streamlit-drawable-canvas streamlit-image-coordinates

echo "==> Verifying CUDA is still intact + Blackwell kernel works"
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0), "(Blackwell = (12, 0))")
    # tiny op forces a real CUDA kernel launch — fails loudly if torch lacks sm_120
    x = torch.randn(8, 8, device="cuda"); _ = (x @ x).sum().item()
    print("CUDA matmul OK")
else:
    print("WARNING: CUDA not available — check the NGC image / driver")
PY

echo "==> Verifying project imports"
python -c "import cellpose, skimage, streamlit, pandas, sklearn, tifffile, imageio; print('all imports OK')"

echo
echo "==> DONE. Launch the app (port 8080 matches your SSH -L tunnel):"
echo "    streamlit run app.py --server.port 8080 --server.address 0.0.0.0"
echo "    then open  http://localhost:8080  on your laptop"
echo "    In the app: Segmentation -> check 'Use GPU', set 'Max frames = 0'."
