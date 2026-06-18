#!/usr/bin/env python
# entry_point.py
# Horizon Forecast - Local Training Entry Point (Windows PC)
# Authors: Or Mordechay Hod, Gilad Boudman | Braude College, CODE: 26-1-R-1
#
# RTX A5000 (Ampere, 24GB) - PRIMARY local GPU (CUDA device 0):
#   precision   = bf16
#   batch_size  = 16
#   grad_accum  = 2      (effective batch = 32)
#   num_workers = 4      (validated stable on Windows; 8+ deadlock on pin_memory)
#   use_compile = False  (Triton on Windows is fragile)
#   device_id   = 0      (PyTorch logical 0 = A5000 on this machine)
#
# RTX 3070 (Ampere, 8GB) - FALLBACK (CUDA device 1):
#   Pass: --device-id 1 --batch-size 2 --grad-accum 8 --precision fp16
#
# Usage:
#   python entry_point.py                    # A5000 full training (defaults)
#   python entry_point.py --smoke            # 2-epoch sanity check
#   python entry_point.py --rollout-max 16   # push to 4h rollout (default 8 = 2h)
#   python entry_point.py --no-cascade       # ablation: sever Stage1→Stage2 fusion

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import random
import torch

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.data.dataset  import get_dataloaders, C_STACKED, N_RAIN_BINS
from src.data.prep     import verify_artifacts
from src.models.model  import HorizonForecastModel
from src.train.train   import HorizonLoss, Trainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="2-epoch sanity check (verifies pipeline end-to-end)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=None,
                        help="Val DataLoader batch size. Defaults to --batch-size. "
                             "Set larger (e.g. 64) to exploit free VRAM during inference "
                             "(no gradients/optimizer states). A5000 24GB: 64-96 safe.")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. 4 measured stable on Windows. "
                             "8 and 12 hang at pin_memory in this environment; "
                             "do not raise without validating end-to-end.")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ckpt-dir", default="checkpoints/")
    parser.add_argument("--device-id", type=int, default=0,
                        help="CUDA device index. On this machine: 0=A5000 24GB, 1=RTX 3070 8GB.")
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16",
                        help="bf16 for A5000/H100; fp16 for RTX 3070")
    parser.add_argument("--rollout-max", type=int, default=8,
                        help="max rollout steps per train step (8=2h, 16=4h)")
    parser.add_argument("--ss-anneal-epochs", type=int, default=40,
                        help="epochs to anneal scheduled-sampling prob 1.0->0.1")
    parser.add_argument("--ssim-blur-threshold", type=float, default=0.25,
                        help="rollout viz stops when SSIM drops below this")
    parser.add_argument("--wandb", action="store_true",
                        help="enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="horizon-forecast",
                        help="W&B project name (default: horizon-forecast)")
    parser.add_argument("--no-cascade", action="store_true",
                        help="ABLATION: zero out drivers fed to ManifestationHead "
                             "(Stage 1 still supervised). Used to answer research "
                             "question vs cascaded run.")
    parser.add_argument("--multihorizon-every", type=int, default=20,
                        help="Run per-horizon eval every N epochs (also tracks "
                             "best_rollout.pt). 0 = disable. ~15-30 min per call. "
                             "Default 20: ~5 runs over 100 epochs.")
    parser.add_argument("--resume-ckpt", default=None,
                        help="Resume training from this checkpoint .pt file. "
                             "Loads model weights and skips epochs already done. "
                             "Optimizer state is NOT preserved (cosine LR is "
                             "fast-forwarded to the right step).")
    parser.add_argument("--resume-weights-only", action="store_true",
                        help="Load model weights from --resume-ckpt but reset epoch "
                             "counter and LR schedule to 1. Use for phase-transfer "
                             "(e.g. H5 Phase1→Phase2) where checkpoint epoch should "
                             "not carry over.")
    parser.add_argument("--era5-path", default=None,
                        help="Path to era5_npy/ directory. If set, uses dense ERA5 "
                             "wind+temp as Stage1 supervision instead of sparse IMS stations.")
    parser.add_argument("--train-csv", default="data/processed/index_train.csv",
                        help="Path to train index CSV. Pass index_train_subset.csv "
                             "after running subsample_train.py to train on a smaller "
                             "fraction (sprint-speed mode).")
    # H5 two-phase frozen training
    parser.add_argument("--freeze-stage", choices=["none", "mani", "encoder_phys"],
                        default="none",
                        help="H5 two-phase: 'mani' freezes ManifestationHead (Phase 1 — "
                             "train encoder+phys_head only). 'encoder_phys' freezes "
                             "SimVPv2Encoder+PhysicsDriverHead (Phase 2 — train mani_head only).")
    parser.add_argument("--lambda-cloud",  type=float, default=None,
                        help="Override lambda_cloud loss weight (default 1.0)")
    parser.add_argument("--lambda-thermo", type=float, default=None,
                        help="Override combined thermo loss weight (replaces split wind/temp)")
    parser.add_argument("--lambda-rain",   type=float, default=None,
                        help="Override lambda_rain loss weight (default 0.5)")
    args = parser.parse_args()

    if args.smoke:
        args.max_epochs = 2
        logger.warning("SMOKE MODE: max_epochs=2, no real training")

    if not torch.cuda.is_available():
        logger.error("CUDA not available. Local training requires NVIDIA GPU.")
        return 1

    n_gpus = torch.cuda.device_count()
    if args.device_id >= n_gpus:
        logger.error(f"--device-id {args.device_id} invalid: only {n_gpus} GPU(s) found.")
        return 1

    # Auto-select GPU with most VRAM if requested device has < 16GB (e.g. after reboot GPU order swaps)
    requested = args.device_id
    if torch.cuda.get_device_properties(requested).total_memory < 16 * 1024**3:
        best = max(range(n_gpus), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
        logger.warning(f"GPU {requested} has <16GB VRAM — auto-switching to GPU {best} (most VRAM)")
        args.device_id = best

    torch.cuda.set_device(args.device_id)
    props = torch.cuda.get_device_properties(args.device_id)
    logger.info(
        f"GPU {args.device_id}: {props.name}  "
        f"VRAM: {props.total_memory/1e9:.1f} GB  "
        f"precision={args.precision}  "
        f"effective_batch={args.batch_size * args.grad_accum}  "
        f"rollout_max={args.rollout_max}"
    )

    verify_artifacts("data/processed", require_test=False)

    with Path("data/processed/norm_stats.json").open() as f:
        norm_stats = {k: tuple(v) for k, v in json.load(f).items()}
    logger.info(f"NORM_STATS loaded: {norm_stats}")

    rain_weights = torch.load("data/processed/rain_weights.pt", weights_only=True)
    logger.info(f"rain_weights: dry={rain_weights[0]:.3f}  "
                f"rain_mean={rain_weights[1:].mean():.3f}")

    logger.info(f"Train CSV: {args.train_csv}")
    if args.era5_path:
        logger.info(f"ERA5 dense supervision: {args.era5_path}")
    train_loader, val_loader = get_dataloaders(
        train_csv   =args.train_csv,
        val_csv     ="data/processed/index_val.csv",
        dem_path    ="data/processed/dem_256.npy",
        mask_path   ="data/processed/station_mask.pt",
        norm_stats  =norm_stats,
        batch_size      =args.batch_size,
        val_batch_size  =args.val_batch_size,
        num_workers     =args.num_workers,
        era5_npy_dir=args.era5_path,
    )
    logger.info(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    model = HorizonForecastModel(
        in_channels=C_STACKED,
        latent_dim =256,
        n_rain_bins=N_RAIN_BINS,
        no_cascade =args.no_cascade,
    )
    if args.no_cascade:
        logger.warning("ABLATION MODE: --no-cascade ON. Drivers zeroed before ManifestationHead.")

    # H5 two-phase freeze logic
    if args.freeze_stage == "mani":
        for p in model.mani_head.parameters():
            p.requires_grad_(False)
        frozen_params  = sum(p.numel() for p in model.mani_head.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"FREEZE Phase 1: mani_head frozen ({frozen_params/1e6:.1f}M). "
                    f"Trainable: {trainable_params/1e6:.1f}M (encoder + phys_head)")
    elif args.freeze_stage == "encoder_phys":
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        for p in model.phys_head.parameters():
            p.requires_grad_(False)
        frozen_params  = (sum(p.numel() for p in model.encoder.parameters()) +
                          sum(p.numel() for p in model.phys_head.parameters()))
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"FREEZE Phase 2: encoder+phys_head frozen ({frozen_params/1e6:.1f}M). "
                    f"Trainable: {trainable_params/1e6:.1f}M (mani_head only)")
    else:
        logger.info(f"Model parameters: {model.n_params/1e6:.1f}M")

    # Loss weights: CLI overrides take precedence; Phase 1 typically zeros cloud+rain,
    # Phase 2 typically zeros thermo.
    lam_cloud  = args.lambda_cloud  if args.lambda_cloud  is not None else 1.0
    lam_rain   = args.lambda_rain   if args.lambda_rain   is not None else 0.5
    if args.lambda_thermo is not None:
        # Unified thermo override (bypasses split wind/temp)
        loss_fn = HorizonLoss(
            rain_weights =rain_weights,
            lambda_cloud =lam_cloud,
            lambda_thermo=args.lambda_thermo,
            lambda_rain  =lam_rain,
        )
    else:
        loss_fn = HorizonLoss(
            rain_weights =rain_weights,
            lambda_cloud =lam_cloud,
            lambda_wind  =1.0,
            lambda_temp  =0.5,
            lambda_rain  =lam_rain,
        )
    logger.info(f"Loss weights: cloud={lam_cloud}  rain={lam_rain}  "
                f"thermo={'unified=' + str(args.lambda_thermo) if args.lambda_thermo is not None else 'wind=1.0 temp=0.5'}")

    trainer = Trainer(
        model                =model,
        loss_fn              =loss_fn,
        train_loader         =train_loader,
        val_loader           =val_loader,
        lr                   =args.lr,
        weight_decay         =1e-2,
        max_epochs           =args.max_epochs,
        grad_accum           =args.grad_accum,
        warmup_epochs        =1 if args.smoke else 5,
        ckpt_dir             =args.ckpt_dir,
        ckpt_prefix          =f"gpu{args.device_id}",
        use_compile          =False,
        precision            =args.precision,
        rollout_max          =args.rollout_max,
        ss_anneal_epochs     =args.ss_anneal_epochs,
        ssim_blur_threshold  =args.ssim_blur_threshold,
        use_wandb            =args.wandb,
        wandb_project        =args.wandb_project,
        multihorizon_every   =0 if args.smoke else args.multihorizon_every,
        resume_ckpt          =args.resume_ckpt,
        resume_weights_only  =args.resume_weights_only,
    )
    trainer.train()
    logger.info("training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
