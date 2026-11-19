#!/bin/bash
# ============================================
# ONE-TIME SETUP ON THE GPU SERVER (/workspace is the persistent drive)
# ============================================
# Run from the unzipped code folder:   bash setup_server.sh
set -e

export TLE_DATA_ROOT=/workspace/tle_data
export TLE_RUN_DIR=/workspace/tle_data/run_server
mkdir -p "$TLE_DATA_ROOT" "$TLE_RUN_DIR"

# 1. Python packages. The container's PyTorch is NVIDIA's own build; the
#    constraints file stops pip from replacing it while installing HD-BET.
pip freeze | grep -i -E "^(torch|torchvision|pytorch-triton)==" > /workspace/constraints.txt || true
pip install -c /workspace/constraints.txt SimpleITK nibabel nilearn PyWavelets networkx umap-learn matplotlib tqdm hd-bet

# 2. Confirm torch is untouched and sees the GPU.
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 3. Download the 219 cohort subjects (T1 + FLAIR + JSON) from OpenNeuro.
python download_subjects.py $(cat subject_list.txt)

echo
echo "Setup done. Data in $TLE_DATA_ROOT, outputs will go to $TLE_RUN_DIR"
