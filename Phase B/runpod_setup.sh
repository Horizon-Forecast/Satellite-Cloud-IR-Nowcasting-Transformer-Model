#!/bin/bash
# runpod_setup.sh
# Run this on RunPod pod after uploading archives.
# Usage: bash runpod_setup.sh

set -e
WORKSPACE=/workspace

echo "=== 1. Clone code ==="
cd $WORKSPACE
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git project
cd project

echo "=== 2. Install dependencies ==="
pip install -r requirements.txt

echo "=== 3. Extract satellite data ==="
mkdir -p data/processed
tar --zstd -xf $WORKSPACE/horizon_sat.tar.zst -C data/processed
echo "sat_npy extracted"

echo "=== 4. Extract ERA5 (if uploaded) ==="
if [ -f "$WORKSPACE/horizon_era5.tar.zst" ]; then
    mkdir -p data
    tar --zstd -xf $WORKSPACE/horizon_era5.tar.zst -C data
    echo "era5_npy extracted"
else
    echo "horizon_era5.tar.zst not found — re-download ERA5 manually"
fi

echo "=== 5. Rebase CSV paths ==="
python runpod_rebase_csv.py --data-root /workspace/project/data/processed

echo "=== 6. Verify setup ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

echo "=== Setup complete ==="
echo "Launch training with:"
echo "python entry_point.py --device-id 0 --precision bf16 --batch-size 12 --grad-accum 3 --max-epochs 30 --num-workers 4 --rollout-max 2 --multihorizon-every 10 --train-csv data/processed/index_train_subset.csv --wandb --wandb-project horizon-forecast --era5-path data/era5_npy --no-cascade"
