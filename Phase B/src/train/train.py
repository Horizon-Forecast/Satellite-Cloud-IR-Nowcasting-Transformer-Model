"""
Two-Phase Training Loop (Phase B).

Trainer for the two-phase frozen schedule: mixed-precision (bf16/fp16) autoregressive
rollout with scheduled sampling, cosine LR, checkpointing, and periodic multi-horizon eval.
Also exposes load_model_for_inference() used by the eval / visualization tools.
"""

import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from tqdm.auto import tqdm

from src.data.dataset import N_RAIN_BINS, T_ROLLOUT
from src.models.model import HorizonForecastModel, MaskedMSELoss

logger = logging.getLogger(__name__)


def _ssim_simple(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Fast single-channel SSIM approximation for rollout blur detection.
    pred, target: any shape float tensors (treated as flat distributions).
    Returns scalar in [0, 1].
    """
    mu1 = pred.mean()
    mu2 = target.mean()
    s1  = pred.var()
    s2  = target.var()
    s12 = ((pred - mu1) * (target - mu2)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return float((num / den.clamp(min=1e-8)).clamp(0, 1).item())


# Multi-Objective Loss (§6.6 eq. 2-5)
class HorizonLoss(nn.Module):
    """
    L_total = λ_cloud * L_cloud  +  λ_wind * L_wind  +  λ_temp * L_temp  +  λ_rain * L_rain

    L_cloud  (eq. 3): Dense MSE over full 256×256 grid vs future satellite frame.
                      Acts as spatial consistency constraint — forces cloud motion
                      to follow physically valid fluid flow patterns.

    L_wind   (eq. 4a): wind channel (drivers[:,0]). Masked MSE at IMS station pixels by
                       default, DENSE full-grid MSE when an ERA5 grid is supplied
                       (era5_dense, Phase B). Wind is the primary driver — kept strong.

    L_temp   (eq. 4b): temp channel (drivers[:,1]). Same sparse/dense logic as L_wind.
                       Temperature is less critical for rain detection, so weighted lower.

    L_rain   (eq. 5): 64-bin class-weighted cross-entropy at IMS station pixels, with
                      extreme weights on rain classes to counter 98.84% dry dominance.

    The thermo supervision density is selected per-batch from the dataset flag
    `era5_dense` — see forward().

    Split-thermo weights (e.g. λ_cloud=1.0, λ_wind=1.0, λ_temp=0.2, λ_rain=0.5) keep wind
    strong (the primary driver) and ease temp (less critical for rain detection).
    """

    def __init__(
        self,
        rain_weights:  torch.Tensor,  # (64,) from compute_rain_class_weights()
        lambda_cloud:  float = 1.0,
        lambda_thermo: float = 0.5,   # legacy param — ignored if lambda_wind/lambda_temp set
        lambda_rain:   float = 2.0,
        lambda_wind:   float = None,  # split wind supervision (None = use lambda_thermo)
        lambda_temp:   float = None,  # split temp supervision (None = use lambda_thermo)
    ):
        super().__init__()
        self.masked_mse    = MaskedMSELoss()
        self.lambda_cloud  = lambda_cloud
        self.lambda_rain   = lambda_rain
        # split-thermo mode: separate wind/temp lambdas
        if lambda_wind is not None and lambda_temp is not None:
            self.lambda_wind   = lambda_wind
            self.lambda_temp   = lambda_temp
            self.lambda_thermo = None   # disabled
            self._split_thermo = True
        else:
            self.lambda_thermo = lambda_thermo
            self.lambda_wind   = None
            self.lambda_temp   = None
            self._split_thermo = False
        self.register_buffer("rain_weights", rain_weights)
        assert not torch.isnan(rain_weights).any(), "rain_weights contains NaN"
        assert not torch.isinf(rain_weights).any(), "rain_weights contains Inf"
        assert (rain_weights > 0).all(), "rain_weights must all be positive"

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the multi-objective training loss for one batch.

        Combines three weighted terms and returns them plus the total:
          L_cloud  — dense MSE on the predicted IR+WV cloud field.
          L_thermo — wind/temp driver loss: dense full-grid MSE when ERA5 is loaded,
                     else sparse masked MSE at IMS station pixels (Phase-A regime).
          L_rain   — class-weighted cross-entropy over 64 rain buckets at station pixels.
        Weighting is set by the lambda_* fields. A frozen stage contributes a zero term.

        Returns a dict with each component and `total` (the value to backprop).
        """
        drivers, y_cloud_pred, y_rain_logits = preds

        y_cloud  = batch["y_cloud"]       # (B, 1, H, W)
        y_thermo = batch["y_thermo"]      # (B, 2, H, W)
        mask     = batch["station_mask"]  # (B, H, W) bool
        y_rain   = batch["y_rain"]        # (B, H, W) int64

        # L_cloud: dense MSE — full 256×256 grid (both IR+WV channels)
        l_cloud = F.mse_loss(y_cloud_pred, y_cloud)

        # L_thermo: wind+temp supervision
        # ERA5 mode: dense full-grid MSE (no mask needed — every pixel has ground truth)
        # IMS mode: sparse masked MSE — station pixels only
        era5_dense = batch.get("era5_dense", False)  # True when ERA5 loaded
        if isinstance(era5_dense, torch.Tensor): era5_dense = bool(era5_dense.any())
        if self._split_thermo:
            if era5_dense:
                l_wind = F.mse_loss(drivers[:, 0:1], y_thermo[:, 0:1])
                l_temp = F.mse_loss(drivers[:, 1:2], y_thermo[:, 1:2])
            else:
                l_wind = self.masked_mse(drivers[:, 0:1], y_thermo[:, 0:1], mask)
                l_temp = self.masked_mse(drivers[:, 1:2], y_thermo[:, 1:2], mask)
            l_thermo = l_wind + l_temp  # combined for logging
        else:
            if era5_dense:
                l_thermo = F.mse_loss(drivers, y_thermo)
            else:
                l_thermo = self.masked_mse(drivers, y_thermo, mask)

        # L_rain: class-weighted CE at IMS station pixels only.
        B, C, H, W = y_rain_logits.shape
        # Mask flattening: select only active station pixels before CE computation
        logits_flat = y_rain_logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, 64)
        target_flat = y_rain.reshape(-1)                                   # (B*H*W,)
        mask_flat   = mask.reshape(-1)                                     # (B*H*W,) bool

        active_logits = logits_flat[mask_flat]   # (N_active, 64)
        active_target = target_flat[mask_flat]   # (N_active,)

        if active_logits.numel() > 0:
            l_rain = F.cross_entropy(active_logits, active_target, weight=self.rain_weights)
        else:
            # Rare edge case: batch has no station pixels (e.g., all masked out)
            l_rain = y_rain_logits.sum() * 0.0  # zero loss, keeps compute graph alive

        if self._split_thermo:
            total = (
                self.lambda_cloud * l_cloud
                + self.lambda_wind  * l_wind
                + self.lambda_temp  * l_temp
                + self.lambda_rain  * l_rain
            )
        else:
            total = (
                self.lambda_cloud  * l_cloud
                + self.lambda_thermo * l_thermo
                + self.lambda_rain   * l_rain
            )

        return {
            "total":  total,
            "cloud":  l_cloud.detach(),
            "thermo": l_thermo.detach(),
            "rain":   l_rain.detach(),
        }


# Trainer
class Trainer:
    """
    Training orchestrator for Horizon Forecast.

    Performance choices:
      - Mixed precision: bf16 (no GradScaler, wider exponent range, ~2× over FP32) or
        fp16 (with loss scaling).
      - Optional torch.compile: fuses ops, ~30-40% throughput after a first warmup epoch.
      - Fused AdamW and TF32 matmul/cuDNN on Ampere+.
      - Gradient accumulation for a larger effective batch than fits in a single step.
      - Cosine LR with linear warmup.

    Checkpoints store only the model state_dict (no optimizer, no torch.compile wrapper), so
    load_model_for_inference() can restore them on any device (including CPU) at FP16.
    """

    def __init__(
        self,
        model:         HorizonForecastModel,
        loss_fn:       HorizonLoss,
        train_loader,
        val_loader,
        lr:            float = 2e-4,
        weight_decay:  float = 1e-2,
        max_epochs:    int   = 100,
        grad_accum:    int   = 2,       # effective batch = batch_size * grad_accum
        warmup_epochs: int   = 5,
        ckpt_dir:      str   = "checkpoints/",
        use_compile:   bool  = True,
        precision:     str   = "bf16",   # "bf16" (Ampere+/H100) or "fp16" (older GPUs)
        ckpt_prefix:    str   = "",        # filename prefix for multi-device runs
        use_wandb:      bool  = False,
        wandb_project:  str   = "horizon-forecast",
        resume_ckpt:    Optional[str] = None,
        resume_weights_only: bool = False,  # load model weights only, reset epoch/LR (for phase-transfer)
        norm_stats:     Optional[Dict] = None,
        station_pixels: Optional[list] = None,
        viz_every:      int   = 10,
        eval_every:     int   = 10,
        multihorizon_every: int = 0,   # 0 = disabled. N = run per-horizon eval every N epochs
        # Scheduled-sampling rollout params
        rollout_max:         int   = 8,    # max rollout steps (8 × 15min = 2h, grows to T_ROLLOUT)
        ss_anneal_epochs:    int   = 40,   # epochs to anneal ss_prob from 1.0 → ss_min
        ss_min:              float = 0.1,  # floor for scheduled-sampling real-frame probability
        ssim_blur_threshold: float = 0.25, # stop rollout viz when SSIM drops below this
        rollout_loss_decay:  float = 0.85, # per-step loss weight multiplier (1.0, 0.85, 0.72…)
    ):
        if precision not in ("bf16", "fp16"):
            raise ValueError(f"precision must be 'bf16' or 'fp16', got {precision!r}")

        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grad_accum  = grad_accum
        self.max_epochs  = max_epochs
        self.ckpt_dir    = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_prefix = ckpt_prefix

        # GPU performance flags (Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        torch.backends.cudnn.benchmark         = True

        self.model = model.to(self.device)

        if use_compile and hasattr(torch, "compile"):
            torch._dynamo.config.suppress_errors = True
            self.model = torch.compile(self.model, mode="default")
            logger.info("torch.compile enabled (mode=default). First epoch will be slow (compilation).")

        self.loss_fn      = loss_fn.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        # fused AdamW available on Ampere+ (H100, A100, RTX 30/40 series)
        # Only optimize unfrozen params — supports the two-phase frozen schedule (freeze_stage).
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            fused=torch.cuda.is_available(),
        )

        total_steps  = max_epochs * len(train_loader)
        warmup_steps = warmup_epochs * len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda s: self._lr_schedule(s, warmup_steps, total_steps),
        )

        # Precision selection:
        #   bf16 — wider exponent range, no GradScaler needed (Ampere+/H100)
        #   fp16 — fastest on older tensor cores, requires loss scaling
        self.precision = precision
        if precision == "bf16":
            self.amp_dtype = torch.bfloat16
            self.scaler    = None
        else:
            self.amp_dtype = torch.float16
            self.scaler    = torch.amp.GradScaler(device="cuda")
        logger.info(f"Trainer precision={precision} amp_dtype={self.amp_dtype}")

        self.best_val          = float("inf")
        self.best_rollout      = float("-inf")  # higher = better
        self.start_epoch       = 1
        self.use_wandb         = use_wandb
        self.norm_stats        = norm_stats
        self.station_pixels    = [(r, c) for r, c, *_ in station_pixels] if station_pixels else None
        self.viz_every         = viz_every
        self.eval_every        = eval_every
        self.multihorizon_every = multihorizon_every

        # Rollout / scheduled sampling
        self.rollout_max          = rollout_max
        self.ss_anneal_epochs     = ss_anneal_epochs
        self.ss_min               = ss_min
        self.ssim_blur_threshold  = ssim_blur_threshold
        self.rollout_loss_decay   = rollout_loss_decay

        # Cache DEM on device for rollout frame construction (static — never changes)
        dem_tensor = getattr(train_loader.dataset, "dem", None)
        self.dem = dem_tensor.to(self.device) if dem_tensor is not None else None
        if use_wandb:
            import wandb
            wandb.init(
                project=wandb_project,
                config={
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "max_epochs": max_epochs,
                    "batch_size": train_loader.batch_size,
                    "effective_batch": grad_accum * train_loader.batch_size,
                    "grad_accum": grad_accum,
                    "precision": precision,
                    "lambda_cloud":  loss_fn.lambda_cloud,
                    "lambda_thermo": loss_fn.lambda_thermo if not loss_fn._split_thermo else None,
                    "lambda_wind":   loss_fn.lambda_wind,
                    "lambda_temp":   loss_fn.lambda_temp,
                    "lambda_rain":   loss_fn.lambda_rain,
                },
            )
            logger.info(f"W&B run: {wandb.run.url}")

        self.resume_weights_only = resume_weights_only
        if resume_ckpt is not None:
            self._resume(resume_ckpt)

    def _resume(self, ckpt_path: str) -> None:
        """Restore model (and, unless weights-only phase-transfer, optimizer/scheduler/epoch) from a checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        raw_model = getattr(self.model, "_orig_mod", self.model)
        raw_model.load_state_dict(ckpt["model_state"])
        if self.resume_weights_only:
            # Phase-transfer: load weights only, start fresh from epoch 1.
            # LR scheduler stays at step 0 (warmup from scratch).
            logger.info(
                f"Weights-only load | ckpt={ckpt_path} | start_epoch=1 (epoch/LR reset)"
            )
            return
        done_epoch       = int(ckpt["epoch"])
        self.start_epoch = done_epoch + 1
        self.best_val    = float(ckpt.get("val_loss", float("inf")))
        # Fast-forward LR scheduler to the step count already completed.
        # Pure Python lambda — no GPU ops, takes <1s even at epoch 100.
        steps_done = done_epoch * len(self.train_loader)
        for _ in range(steps_done):
            self.scheduler.step()
        logger.info(
            f"Resumed | ckpt={ckpt_path} | done_epoch={done_epoch} "
            f"| val_loss={self.best_val:.4f} | next_epoch={self.start_epoch}"
        )

    @staticmethod
    def _lr_schedule(step: int, warmup: int, total: int) -> float:
        """Linear warmup → cosine decay to 0."""
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _to_device(self, batch: Dict) -> Dict:
        """Move every tensor in the batch to the training device (async, pinned-memory transfer)."""
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _ss_prob(self, epoch: int) -> float:
        """Scheduled-sampling: prob of using REAL frame. 1.0 → ss_min over ss_anneal_epochs."""
        frac = min(1.0, epoch / max(self.ss_anneal_epochs, 1))
        return max(self.ss_min, 1.0 - (1.0 - self.ss_min) * frac)

    def _curriculum_steps(self, epoch: int) -> int:
        """Curriculum rollout: add 1 step every 5 epochs, cap at rollout_max."""
        return min(1 + epoch // 5, self.rollout_max)

    def _build_next_frame(
        self, y_cloud_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct next 3-channel input frame from model prediction.
        y_cloud_pred: (B, 2, H, W)  — predicted IR+WV
        Returns:      (B, 3, H, W)  — [IR, WV, DEM] ready to append to input window
        """
        B = y_cloud_pred.size(0)
        if self.dem is not None:
            dem_b = self.dem.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 1, H, W)
        else:
            dem_b = torch.zeros(B, 1, *y_cloud_pred.shape[2:], device=self.device)
        return torch.cat([y_cloud_pred, dem_b], dim=1)  # (B, 3, H, W)

    def _step_batch(self, batch: Dict, step: int) -> Dict:
        """Extract single-step targets from multi-step batch for loss computation."""
        sb = {
            "y_cloud":      batch["y_sat"][:, step],       # (B, 2, H, W)
            "y_thermo":     batch["y_thermo"][:, step],    # (B, 2, H, W)
            "station_mask": batch["station_mask"],          # (B, H, W)
            "y_rain":       batch["y_rain"][:, step],       # (B, H, W)
            "era5_dense":   batch.get("era5_dense", False),  # forward dense-thermo flag to the loss
        }
        return sb

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch: forward/loss/backward with grad accumulation, scheduled-sampling
        rollout, and BF16 autocast. Returns the epoch-averaged loss components."""
        self.model.train()
        sums = {"total": 0.0, "cloud": 0.0, "thermo": 0.0, "rain": 0.0}
        t0   = time.perf_counter()

        ss_prob      = self._ss_prob(epoch)
        n_steps      = self._curriculum_steps(epoch)

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"E{epoch:03d} train ss={ss_prob:.2f} rollout={n_steps}",
            leave=False,
        )
        for batch_idx, batch in pbar:
            batch   = self._to_device(batch)
            B       = batch["x"].size(0)
            x_input = batch["x"]                          # (B, 12, H, W) real input
            valid   = batch["valid_steps"].long()         # (B,)

            rollout_loss   = torch.tensor(0.0, device=self.device)
            step_weight    = 1.0
            loss_breakdown = {"cloud": 0.0, "thermo": 0.0, "rain": 0.0}
            n_active_steps = 0

            for rollout_step in range(n_steps):
                # Skip steps where no sample in batch has valid targets
                if (valid > rollout_step).sum() == 0:
                    break

                with autocast("cuda", dtype=self.amp_dtype):
                    preds                            = self.model(x_input)
                    drivers, y_cloud_pred, y_rain_lg = preds
                    sb                               = self._step_batch(batch, rollout_step)
                    losses                           = self.loss_fn(sb, preds)

                # Mask loss to samples that have valid target at this step
                step_valid_mask = (valid > rollout_step).float()
                masked_loss = (losses["total"] * step_valid_mask).sum() / step_valid_mask.sum().clamp(min=1)
                rollout_loss = rollout_loss + step_weight * masked_loss / self.grad_accum
                step_weight *= self.rollout_loss_decay

                for k in ("cloud", "thermo", "rain"):
                    loss_breakdown[k] += losses[k].item() * step_weight
                n_active_steps += 1

                # Build next input via scheduled sampling
                with torch.no_grad():
                    pred_frame = self._build_next_frame(y_cloud_pred.detach())  # (B, 3, H, W)

                    if rollout_step + 1 < n_steps and rollout_step + 1 < T_ROLLOUT:
                        has_real = (valid > rollout_step + 1)              # (B,) bool
                        real_sat = batch["y_sat"][:, rollout_step + 1]     # (B, 2, H, W)
                        dem_b    = (self.dem.unsqueeze(0).expand(B, -1, -1, -1)
                                    if self.dem is not None
                                    else torch.zeros(B, 1, *pred_frame.shape[2:], device=self.device))
                        real_frame = torch.cat([real_sat, dem_b], dim=1)   # (B, 3, H, W)

                        # Per-sample scheduled sampling: use real with prob ss_prob
                        use_real = (
                            (torch.rand(B, device=self.device) < ss_prob) & has_real
                        ).view(B, 1, 1, 1).expand_as(pred_frame)
                        next_frame = torch.where(use_real, real_frame, pred_frame)
                    else:
                        next_frame = pred_frame

                    x_input = torch.cat([x_input[:, 3:], next_frame], dim=1)

            if self.scaler is not None:
                self.scaler.scale(rollout_loss).backward()
            else:
                rollout_loss.backward()

            if (batch_idx + 1) % self.grad_accum == 0:
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_item = rollout_loss.item() * self.grad_accum
            sums["total"]  += total_item
            for k in ("cloud", "thermo", "rain"):
                sums[k] += loss_breakdown[k] / max(n_active_steps, 1)

            pbar.set_postfix({"loss": f"{total_item:.4f}", "steps": n_active_steps})

        n   = len(self.train_loader)
        avg = {k: v / n for k, v in sums.items()}
        dt  = time.perf_counter() - t0
        lr  = self.scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch:03d} | TRAIN  "
            f"total={avg['total']:.4f}  cloud={avg['cloud']:.4f}  "
            f"thermo={avg['thermo']:.4f}  rain={avg['rain']:.4f}  "
            f"| rollout={n_steps}  ss={ss_prob:.2f}  {dt:.0f}s  LR={lr:.2e}"
        )
        if self.use_wandb:
            import wandb
            wandb.log({
                "train/total":          avg["total"],
                "train/cloud":          avg["cloud"],
                "train/thermo":         avg["thermo"],
                "train/rain":           avg["rain"],
                "train/epoch_seconds":  dt,
                "train/lr":             lr,
                "train/rollout_steps":  n_steps,
                "train/ss_prob":        ss_prob,
                "epoch":                epoch,
            })
        return avg

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> float:
        """Run one validation epoch (no grad, eval mode) and return the averaged loss components
        used for best-checkpoint selection."""
        self.model.eval()
        sums = {"total": 0.0, "cloud": 0.0, "thermo": 0.0, "rain": 0.0}
        # IMS monitoring accumulators (ERA5 mode only)
        ims_wind_sq, ims_temp_sq, ims_n = 0.0, 0.0, 0

        for batch in tqdm(self.val_loader, desc=f"E{epoch:03d} val  ", leave=False):
            batch = self._to_device(batch)
            with autocast("cuda", dtype=self.amp_dtype):
                preds  = self.model(batch["x"])
                # Validation loss: step 0 only (t+15min) — comparable across all epochs
                losses = self.loss_fn(self._step_batch(batch, 0), preds)
            for k in sums:
                sums[k] += losses[k].item()

            # IMS RMSE monitoring — compare Stage1 predictions vs real IMS station readings
            # Only active in ERA5 mode (y_thermo_ims present and era5_dense=True)
            era5_dense = batch.get("era5_dense", False)
            if isinstance(era5_dense, torch.Tensor): era5_dense = bool(era5_dense.any())
            if era5_dense and "y_thermo_ims" in batch:
                drivers, _, _ = preds
                mask    = batch["station_mask"]           # (B, H, W) bool
                y_ims   = batch["y_thermo_ims"][:, 0]    # (B, 2, H, W) step=0
                # Only at station pixels where IMS has nonzero data
                ims_mask = mask & (y_ims[:, 0] != 0)     # (B, H, W) — nonzero wind pixels
                if ims_mask.any():
                    pred_w = drivers[:, 0][ims_mask].float()
                    pred_t = drivers[:, 1][ims_mask].float()
                    ims_w  = y_ims[:, 0][ims_mask].float()
                    ims_t  = y_ims[:, 1][ims_mask].float()
                    ims_wind_sq += ((pred_w - ims_w) ** 2).sum().item()
                    ims_temp_sq += ((pred_t - ims_t) ** 2).sum().item()
                    ims_n       += ims_mask.sum().item()

        n   = len(self.val_loader)
        avg = {k: v / n for k, v in sums.items()}

        # Compute IMS RMSE (denormalized)
        rmse_wind_ims = float("nan")
        rmse_temp_ims = float("nan")
        if ims_n > 0:
            wind_norm = self.norm_stats.get("wind", (0.0, 1.0)) if self.norm_stats else (0.0, 1.0)
            temp_norm = self.norm_stats.get("temp", (0.0, 1.0)) if self.norm_stats else (0.0, 1.0)
            rmse_wind_ims = ((ims_wind_sq / ims_n) ** 0.5) * wind_norm[1]  # denormalize ×std
            rmse_temp_ims = ((ims_temp_sq / ims_n) ** 0.5) * temp_norm[1]

        logger.info(
            f"Epoch {epoch:03d} | VAL    "
            f"total={avg['total']:.4f}  cloud={avg['cloud']:.4f}  "
            f"thermo={avg['thermo']:.4f}  rain={avg['rain']:.4f}"
            + (f"  | IMS_RMSE wind={rmse_wind_ims:.3f}m/s  temp={rmse_temp_ims:.3f}C" if ims_n > 0 else "")
        )
        if self.use_wandb:
            import wandb
            log = {
                "val/total":   avg["total"],
                "val/cloud":   avg["cloud"],
                "val/thermo":  avg["thermo"],
                "val/rain":    avg["rain"],
                "epoch":       epoch,
            }
            if ims_n > 0:
                log["val/ims_rmse_wind"] = rmse_wind_ims
                log["val/ims_rmse_temp"] = rmse_temp_ims
            wandb.log(log)
        return avg

    def _save(self, epoch: int, val_loss: float, is_best: bool) -> None:
        """
        Save only model state_dict — NOT the torch.compile wrapper.
        The _orig_mod unwrap ensures the checkpoint loads on any device
        (GPU or CPU) without requiring torch.compile to be available.
        """
        raw_model = getattr(self.model, "_orig_mod", self.model)
        ckpt = {
            "epoch":       epoch,
            "val_loss":    val_loss,
            "model_state": raw_model.state_dict(),
            "model_config": {
                "in_channels": raw_model.in_channels,
                "latent_dim":  raw_model.latent_dim,
                "n_rain_bins": raw_model.n_rain_bins,
                "no_cascade":  getattr(raw_model, "no_cascade", False),
            },
        }
        prefix = f"{self.ckpt_prefix}_" if self.ckpt_prefix else ""
        path = self.ckpt_dir / f"{prefix}epoch_{epoch:04d}.pt"
        torch.save(ckpt, path)
        if is_best:
            torch.save(ckpt, self.ckpt_dir / f"{prefix}best.pt")
            logger.info(f"  ★ New best val_loss={val_loss:.4f} → {prefix}best.pt")
        return ckpt  # caller may want to save under another name (e.g., best_rollout.pt)

    @torch.no_grad()
    def _viz_val_sample(self, epoch: int) -> None:
        """Inference on one val batch → PNG + W&B image log."""
        import matplotlib
        matplotlib.use("Agg")  # headless — no display needed on Colab/server
        import matplotlib.pyplot as plt
        from src.viz.visualize import visualize_forecast

        self.model.eval()
        batch = next(iter(self.val_loader))
        batch = self._to_device(batch)

        with autocast("cuda", dtype=self.amp_dtype):
            drivers, y_cloud_pred, y_rain_logits = self.model(batch["x"])

        save_path = self.ckpt_dir / "viz" / f"epoch_{epoch:04d}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig = visualize_forecast(
            x_stacked      = batch["x"][0],
            pred_wind      = drivers[0, 0],
            pred_temp      = drivers[0, 1],
            pred_cloud     = y_cloud_pred[0, 0],      # IR channel
            pred_rain_cls  = y_rain_logits[0].argmax(dim=0),
            true_cloud     = batch["y_sat"][0, 0, 0], # step 0, IR channel
            true_rain_cls  = batch["y_rain"][0, 0],   # step 0
            station_pixels = self.station_pixels,
            norm_stats     = self.norm_stats,
            dem            = self.val_loader.dataset.dem_raw,
            title_suffix   = f"| Epoch {epoch}",
            save_path      = str(save_path),
        )
        if self.use_wandb:
            import wandb
            wandb.log({"val/forecast_viz": wandb.Image(fig), "epoch": epoch})
        plt.close(fig)
        logger.info(f"Epoch {epoch:03d} | VIZ    saved → {save_path}")

    @torch.no_grad()
    def _viz_rollout_sample(self, epoch: int) -> None:
        """
        Roll model out autoregressively on one val sample (always-predicted, no scheduled
        sampling). At each step record SSIM vs real. Stop when SSIM < ssim_blur_threshold
        or all valid steps exhausted. Save a multi-row comparison figure showing:
          - how many steps the model predicted
          - predicted IR vs real IR per step
          - predicted rain vs real rain per step
          - per-step SSIM score
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.viz.visualize import visualize_rollout

        self.model.eval()
        batch = next(iter(self.val_loader))
        batch = self._to_device(batch)

        x_input     = batch["x"][0:1]       # (1, 12, H, W) — single sample
        valid_steps = int(batch["valid_steps"][0].item())
        max_steps   = min(valid_steps, self.rollout_max, T_ROLLOUT)

        steps_data = []  # list of dicts per step

        for step in range(max_steps):
            with autocast("cuda", dtype=self.amp_dtype):
                drivers, y_cloud_pred, y_rain_logits = self.model(x_input)

            pred_ir       = y_cloud_pred[0, 0].float().cpu()             # (H, W)
            pred_rain_cls = y_rain_logits[0].argmax(dim=0).cpu()         # (H, W)
            true_ir       = batch["y_sat"][0, step, 0].float().cpu()     # (H, W)
            true_rain_cls = batch["y_rain"][0, step].cpu()               # (H, W)

            ssim_val = _ssim_simple(
                pred_ir.to(self.device),
                batch["y_sat"][0, step, 0].to(self.device),
            )

            steps_data.append({
                "step":           step + 1,
                "lead_min":       (step + 1) * 15,
                "pred_ir":        pred_ir,
                "pred_rain_cls":  pred_rain_cls,
                "true_ir":        true_ir,
                "true_rain_cls":  true_rain_cls,
                "ssim":           ssim_val,
            })

            if ssim_val < self.ssim_blur_threshold:
                logger.info(
                    f"Epoch {epoch:03d} | ROLLOUT stopped at step {step+1} "
                    f"({(step+1)*15}min) — SSIM={ssim_val:.3f} < {self.ssim_blur_threshold}"
                )
                break

            # Always-predicted next frame (no scheduled sampling in viz)
            next_frame = self._build_next_frame(y_cloud_pred)
            x_input    = torch.cat([x_input[:, 3:], next_frame], dim=1)

        save_path = self.ckpt_dir / "viz" / f"rollout_epoch_{epoch:04d}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig = visualize_rollout(
            steps_data     = steps_data,
            station_pixels = self.station_pixels,
            norm_stats     = self.norm_stats,
            dem            = self.val_loader.dataset.dem_raw,
            ssim_threshold = self.ssim_blur_threshold,
            title_suffix   = f"| Epoch {epoch}",
            save_path      = str(save_path),
        )
        if self.use_wandb:
            import wandb
            wandb.log({"val/rollout_viz": wandb.Image(fig), "epoch": epoch})
        plt.close(fig)

        reached = steps_data[-1]["lead_min"]
        logger.info(
            f"Epoch {epoch:03d} | ROLLOUT viz  steps={len(steps_data)}  "
            f"reached={reached}min  final_SSIM={steps_data[-1]['ssim']:.3f}  "
            f"→ {save_path}"
        )

    def _log_metrics_csv(self, epoch: int, train: Dict, val: Dict, lr: float) -> None:
        """Append this epoch's train/val loss components and learning rate to ckpt_dir/metrics.csv."""
        path = self.ckpt_dir / "metrics.csv"
        write_header = not path.exists()
        with open(path, "a") as f:
            if write_header:
                f.write("epoch,train_total,train_cloud,train_thermo,train_rain,"
                        "val_total,val_cloud,val_thermo,val_rain,lr\n")
            f.write(
                f"{epoch},"
                f"{train['total']:.6f},{train['cloud']:.6f},"
                f"{train['thermo']:.6f},{train['rain']:.6f},"
                f"{val['total']:.6f},{val['cloud']:.6f},"
                f"{val['thermo']:.6f},{val['rain']:.6f},"
                f"{lr:.6e}\n"
            )

    def _run_eval_metrics(self, epoch: int) -> None:
        """Periodically run multi-horizon skill evaluation (CSI/SSIM/POD/FAR) and log it — the
        CSI here, not val loss, is what we select the final checkpoint on."""
        from src.eval.evaluate import evaluate_checkpoint
        station_mask = getattr(self.val_loader.dataset, "station_mask", None)
        if station_mask is None:
            return
        metrics = evaluate_checkpoint(
            model=self.model,
            val_loader=self.val_loader,
            device=self.device,
            amp_dtype=self.amp_dtype,
            station_mask=station_mask,
        )
        logger.info(
            f"Epoch {epoch:03d} | EVAL   "
            f"CSI={metrics['csi']:.4f}  HSS={metrics['hss']:.4f}  "
            f"SSIM={metrics['ssim_cloud']:.4f}  "
            f"RMSE_wind={metrics['rmse_wind']:.4f}  RMSE_temp={metrics['rmse_temp']:.4f}"
        )
        if self.use_wandb:
            import wandb
            wandb.log({
                "eval/csi":       metrics["csi"],
                "eval/hss":       metrics["hss"],
                "eval/ssim":      metrics["ssim_cloud"],
                "eval/rmse_wind": metrics["rmse_wind"],
                "eval/rmse_temp": metrics["rmse_temp"],
                "epoch":          epoch,
            })
        eval_path = self.ckpt_dir / "eval_metrics.csv"
        write_header = not eval_path.exists()
        with open(eval_path, "a") as f:
            if write_header:
                f.write("epoch,csi,hss,ssim_cloud,rmse_wind,rmse_temp\n")
            f.write(
                f"{epoch},{metrics['csi']:.6f},{metrics['hss']:.6f},"
                f"{metrics['ssim_cloud']:.6f},{metrics['rmse_wind']:.6f},"
                f"{metrics['rmse_temp']:.6f}\n"
            )

    def _run_multihorizon_eval(self, epoch: int) -> None:
        """
        Expensive per-horizon eval (rollout to T_ROLLOUT). Runs at most every
        multihorizon_every epochs. Tracks best_rollout.pt separately from best.pt.
        Score: 0.5 * SSIM@T+60 + 0.5 * CSI@1@T+60 (higher = better).
        """
        from src.eval.evaluate import (
            evaluate_checkpoint_multihorizon,
            format_multihorizon_table,
            DEFAULT_HORIZONS_STEPS,
        )
        station_mask = getattr(self.val_loader.dataset, "station_mask", None)
        if station_mask is None:
            return

        prefix = f"{self.ckpt_prefix}_" if self.ckpt_prefix else ""
        out_path = self.ckpt_dir / "multihorizon" / f"{prefix}epoch_{epoch:04d}.json"

        results = evaluate_checkpoint_multihorizon(
            model=self.model,
            val_loader=self.val_loader,
            device=self.device,
            amp_dtype=self.amp_dtype,
            station_mask=station_mask,
            horizons_steps=DEFAULT_HORIZONS_STEPS,
            dem=self.dem,
            out_path=str(out_path),
        )

        logger.info(f"Epoch {epoch:03d} | MULTIHORIZON\n" + format_multihorizon_table(results))

        # Score uses T+60 as the headline operating horizon
        r60 = results.get(60, {})
        ssim60 = r60.get("ssim_cloud", float("nan"))
        csi60  = r60.get("csi@1",     float("nan"))
        score  = 0.0
        valid  = 0
        if not math.isnan(ssim60):
            score += 0.5 * ssim60
            valid += 1
        if not math.isnan(csi60):
            score += 0.5 * csi60
            valid += 1
        if valid == 0:
            score = float("-inf")

        if self.use_wandb:
            import wandb
            log = {"epoch": epoch, "eval/rollout_score": score}
            for h_min, r in results.items():
                log[f"eval/ssim_T+{h_min}"]   = r.get("ssim_cloud", float("nan"))
                log[f"eval/csi@1_T+{h_min}"]  = r.get("csi@1",     float("nan"))
                log[f"eval/csi@3_T+{h_min}"]  = r.get("csi@3",     float("nan"))
                log[f"eval/rmse_w_T+{h_min}"] = r.get("rmse_wind", float("nan"))
            wandb.log(log)

        if score > self.best_rollout:
            self.best_rollout = score
            ckpt = self._make_ckpt(epoch, val_loss=float("nan"), rollout_score=score)
            torch.save(ckpt, self.ckpt_dir / f"{prefix}best_rollout.pt")
            logger.info(
                f"  ★ New best rollout score={score:.4f} "
                f"(SSIM@T+60={ssim60:.3f} CSI@T+60={csi60:.3f}) "
                f"→ {prefix}best_rollout.pt"
            )

    def _make_ckpt(self, epoch: int, val_loss: float, rollout_score: float = float("nan")) -> Dict:
        """Assemble the checkpoint dict (model/optimizer/scheduler state, epoch, scores, model config) for saving."""
        raw_model = getattr(self.model, "_orig_mod", self.model)
        return {
            "epoch":         epoch,
            "val_loss":      val_loss,
            "rollout_score": rollout_score,
            "model_state":   raw_model.state_dict(),
            "model_config": {
                "in_channels": raw_model.in_channels,
                "latent_dim":  raw_model.latent_dim,
                "n_rain_bins": raw_model.n_rain_bins,
                "no_cascade":  getattr(raw_model, "no_cascade", False),
            },
        }

    def train(self) -> None:
        """
        Main training loop: for each epoch run train + validation, step the scheduler, log
        metrics, and checkpoint. Saves the best model by validation loss and (when enabled)
        runs periodic multi-horizon CSI evaluation. Frees CUDA cache between phases to stay
        under the VRAM cap. Runs from self.start_epoch to self.max_epochs (resume-aware).
        """
        logger.info(
            f"Training on {self.device} | "
            f"epochs={self.max_epochs} | "
            f"effective_batch={self.grad_accum * self.train_loader.batch_size}"
        )
        import gc
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            train_avg = self._train_epoch(epoch)
            gc.collect(); torch.cuda.empty_cache()    # release inter-phase cache
            val_avg   = self._val_epoch(epoch)
            gc.collect(); torch.cuda.empty_cache()
            val_loss  = val_avg["total"]
            is_best   = val_loss < self.best_val
            lr        = self.scheduler.get_last_lr()[0]
            if is_best:
                self.best_val = val_loss
            self._log_metrics_csv(epoch, train_avg, val_avg, lr)
            if epoch % 5 == 0 or is_best:
                self._save(epoch, val_loss, is_best)
            if self.eval_every > 0 and (epoch % self.eval_every == 0 or is_best):
                self._run_eval_metrics(epoch)
                gc.collect(); torch.cuda.empty_cache()
            if self.multihorizon_every > 0 and (epoch % self.multihorizon_every == 0):
                self._run_multihorizon_eval(epoch)
                gc.collect(); torch.cuda.empty_cache()
            if epoch % self.viz_every == 0 or is_best:
                self._viz_val_sample(epoch)
                self._viz_rollout_sample(epoch)
                gc.collect(); torch.cuda.empty_cache()


# Inference
def load_model_for_inference(
    checkpoint_path: str,
    device: str = "cuda",
    fp16: bool = True,
) -> HorizonForecastModel:
    """
    Load a trained checkpoint for inference.

    Memory budget at FP16:
      Model weights:   ~30.1M params × 2 bytes ≈ 60 MB
      Input tensor B=1: (1, 12, 256, 256) @ FP16 = ~1.6 MB
      Activations B=1:  ~400-600 MB (no gradients)
      Total estimate:   ~700 MB — runs comfortably on an 8 GB GPU.

    fp16=True  → FP16 inference (fastest on GPU tensor cores)
    fp16=False → FP32 inference (for debugging numerical issues)
    """
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg   = ckpt.get("model_config", {})
    model = HorizonForecastModel(
        in_channels=cfg.get("in_channels", 12),
        latent_dim =cfg.get("latent_dim",  256),
        n_rain_bins=cfg.get("n_rain_bins", 64),
        no_cascade =cfg.get("no_cascade",  False),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    if fp16:
        model = model.half()
    model.eval()

    logger.info(
        f"Model loaded | params={model.n_params/1e6:.1f}M | "
        f"val_loss={ckpt.get('val_loss', 'N/A')} | "
        f"precision={'FP16' if fp16 else 'FP32'} | device={device}"
    )
    return model


@torch.no_grad()
def run_inference(
    model: HorizonForecastModel,
    x: torch.Tensor,  # (1, 12, 256, 256) already on correct device + dtype
) -> Dict[str, torch.Tensor]:
    """
    Single-sample inference pass. Returns all outputs as float32 CPU tensors.
    Use autocast to keep tensor dtype consistent with loaded model precision.
    """
    amp_dtype = next(model.parameters()).dtype
    with autocast("cuda", dtype=amp_dtype):
        drivers, y_cloud, y_rain_logits = model(x)

    return {
        "wind":       drivers[:, 0].float().cpu(),        # (1, 256, 256)
        "temp":       drivers[:, 1].float().cpu(),        # (1, 256, 256)
        "cloud":      y_cloud[:, 0].float().cpu(),        # (1, 256, 256)
        "rain_class": y_rain_logits.argmax(dim=1).cpu(),  # (1, 256, 256) int64
        "rain_probs": y_rain_logits.softmax(dim=1).float().cpu(),  # (1,64,256,256)
    }
