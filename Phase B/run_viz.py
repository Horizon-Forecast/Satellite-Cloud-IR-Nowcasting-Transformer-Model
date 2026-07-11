#!/usr/bin/env python
"""
Standalone Inference + Visualization.

Loads a trained checkpoint, runs inference on N samples from the val or test set, and
saves one PNG per sample to viz_output/.

Usage:
  python run_viz.py                                       # best.pt, 5 val samples
  python run_viz.py --ckpt checkpoints/gpu0_best.pt --n 10
  python run_viz.py --split test --n 3
  python run_viz.py --ckpt checkpoints/gpu0_best_rollout.pt --n 5
  python run_viz.py --ckpt checkpoints/gpu0_epoch_0080.pt --n 1
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import HorizonDataset
from src.train.train  import load_model_for_inference
from src.viz.visualize import visualize_forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Horizon Forecast inference visualization")
    parser.add_argument("--ckpt",    default="checkpoints/best.pt",
                        help="Checkpoint path (.pt)")
    parser.add_argument("--split",   default="val", choices=["val", "test"],
                        help="Which index CSV to sample from")
    parser.add_argument("--n",       type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--out",     default="viz_output",
                        help="Output directory for PNGs")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16",    action="store_true", default=True,
                        help="Use FP16 inference (default True)")
    args = parser.parse_args()

    project_root = str(ROOT)
    processed    = ROOT / "data" / "processed"

    # Load artifacts
    with (processed / "norm_stats.json").open() as f:
        norm_stats = {k: tuple(v) for k, v in json.load(f).items()}

    mask_data      = torch.load(processed / "station_mask.pt", weights_only=True)
    station_pixels = [(r, c) for r, c, *_ in mask_data["pixels"]]

    # Load model
    model = load_model_for_inference(args.ckpt, device=args.device, fp16=args.fp16)
    amp_dtype = torch.float16 if args.fp16 else torch.float32

    # Build dataset (no augmentation)
    index_csv = processed / f"index_{args.split}.csv"
    ds = HorizonDataset(
        str(index_csv),
        str(processed / "dem_256.npy"),
        str(processed / "station_mask.pt"),
        norm_stats=norm_stats,
        augment=False,
        project_root=project_root,
    )
    logger.info(f"Dataset: {len(ds)} samples from {args.split}")

    # Pick N random indices
    random.seed(args.seed)
    indices = random.sample(range(len(ds)), min(args.n, len(ds)))

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from torch import autocast
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load DEM for coastline + hillshade overlay (matches training viz)
    dem_arr = np.load(processed / "dem_256.npy")
    dem_tensor = torch.from_numpy(dem_arr).unsqueeze(0)

    for i, idx in enumerate(indices):
        # The dataset normalizes the band order to IR=channel 0 by default,
        # correcting the alternating IR/WV swap in the raw EUMETVIEW TIFs and
        # matching the layout the delivered checkpoints were trained on. Both the
        # model input and the display below therefore use consistent channels.
        sample = ds[idx]
        x          = sample["x"].unsqueeze(0).to(args.device)
        true_rain  = sample["y_rain"][0]       # (H, W)    - rain class, step 0
        timestamp  = ds.index.iloc[idx]["timestamp"]

        with torch.no_grad():
            with autocast(args.device, dtype=amp_dtype):
                drivers, y_cloud_pred, y_rain_logits = model(x)

        true_cloud = sample["y_sat"][0, 0:1]   # (1, H, W) IR, step 0

        save_path = out_dir / f"{args.split}_{idx:06d}.png"
        fig = visualize_forecast(
            x_stacked      = sample["x"],
            pred_wind      = drivers[0, 0],
            pred_temp      = drivers[0, 1],
            pred_cloud     = y_cloud_pred[0, 0],
            pred_rain_cls  = y_rain_logits[0].argmax(dim=0),
            true_cloud     = true_cloud[0],
            true_rain_cls  = true_rain,
            station_pixels = station_pixels,
            norm_stats     = norm_stats,
            dem            = dem_tensor,
            title_suffix   = f"| {timestamp}",
            save_path      = str(save_path),
        )
        plt.close(fig)
        logger.info(f"[{i+1}/{len(indices)}] {save_path}")

    logger.info(f"Done. {len(indices)} PNGs saved to {out_dir}/")


if __name__ == "__main__":
    main()
