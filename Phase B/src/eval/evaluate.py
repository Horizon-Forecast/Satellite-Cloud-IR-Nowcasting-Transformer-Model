# src/eval/evaluate.py
# Horizon Forecast — Evaluation Metrics (§7.1, Table 1)
# Authors: Or Mordechay Hod, Gilad Boudman | Braude College, CODE: 26-1-R-1
#
# Implements CSI, HSS, SSIM, and a full-val-set evaluate_checkpoint() runner.
# Called automatically by Trainer every eval_every epochs during training.
# Can also be called standalone on a loaded checkpoint.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import autocast

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Binary weather skill scores (station-pixel rain classification)
# ══════════════════════════════════════════════════════════════════════════════

def compute_csi(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    rain_threshold_class: int = 1,
) -> float:
    """
    Critical Success Index = H / (H + M + FA).

    Binarizes: rain = class >= rain_threshold_class (class 0 = dry).
    y_true_cls, y_pred_cls: 1-D int arrays of class indices at station pixels.
    Returns NaN if denominator is 0 (no rain events in batch).
    """
    true_rain = y_true_cls >= rain_threshold_class
    pred_rain = y_pred_cls >= rain_threshold_class
    H  = int(( true_rain &  pred_rain).sum())
    M  = int(( true_rain & ~pred_rain).sum())
    FA = int((~true_rain &  pred_rain).sum())
    denom = H + M + FA
    return H / denom if denom > 0 else float("nan")


def compute_hss(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    rain_threshold_class: int = 1,
) -> float:
    """
    Heidke Skill Score = 2(ad - bc) / ((a+c)(c+d) + (a+b)(b+d)).

    > 0: beats random; 1: perfect; < 0: worse than random.
    Binarizes rain the same way as compute_csi.
    """
    true_rain = y_true_cls >= rain_threshold_class
    pred_rain = y_pred_cls >= rain_threshold_class
    a = int(( true_rain &  pred_rain).sum())   # hits
    b = int((~true_rain &  pred_rain).sum())   # false alarms
    c = int(( true_rain & ~pred_rain).sum())   # misses
    d = int((~true_rain & ~pred_rain).sum())   # correct negatives
    denom = (a + c) * (c + d) + (a + b) * (b + d)
    return 2 * (a * d - b * c) / denom if denom > 0 else float("nan")


def compute_pod(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    rain_threshold_class: int = 1,
) -> float:
    """
    Probability of Detection (recall for rain events) = H / (H + M).

    Fraction of actual rain events that were correctly predicted.
    NaN if no actual rain events in the sample.
    """
    true_rain = y_true_cls >= rain_threshold_class
    pred_rain = y_pred_cls >= rain_threshold_class
    H = int(( true_rain &  pred_rain).sum())
    M = int(( true_rain & ~pred_rain).sum())
    denom = H + M
    return H / denom if denom > 0 else float("nan")


def compute_far(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    rain_threshold_class: int = 1,
) -> float:
    """
    False Alarm Ratio = FA / (H + FA).

    Fraction of predicted rain events that were actually dry.
    NaN if model predicted no rain events.
    """
    true_rain = y_true_cls >= rain_threshold_class
    pred_rain = y_pred_cls >= rain_threshold_class
    H  = int(( true_rain &  pred_rain).sum())
    FA = int((~true_rain &  pred_rain).sum())
    denom = H + FA
    return FA / denom if denom > 0 else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Structural similarity for cloud forecast quality (§7.1)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ssim_cloud(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean SSIM between predicted and true cloud frames.

    y_true, y_pred: float arrays shape (H, W) or (N, H, W).
    For a batch, computes per-image SSIM and returns the mean.
    Requires scikit-image (already in Colab; pip install scikit-image locally).
    """
    from skimage.metrics import structural_similarity as ssim

    if y_true.ndim == 2:
        y_true = y_true[None]
        y_pred = y_pred[None]

    scores: List[float] = []
    for true_img, pred_img in zip(y_true, y_pred):
        data_range = float(true_img.max() - true_img.min())
        if data_range < 1e-6:
            data_range = 1.0
        scores.append(ssim(true_img, pred_img, data_range=data_range))
    return float(np.mean(scores))


# ══════════════════════════════════════════════════════════════════════════════
# Full validation-set evaluation (spec §7.1 Table 1 targets)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_checkpoint(
    model,
    val_loader,
    device: torch.device,
    amp_dtype: torch.dtype,
    station_mask: torch.Tensor,
    rain_threshold_class: int = 1,
) -> Dict[str, float]:
    """
    Run full val-set evaluation and return metric dict:
      csi         — Critical Success Index (rain binary, spec target > 0.5)
      hss         — Heidke Skill Score     (spec target > 0.0)
      ssim_cloud  — Mean SSIM over cloud frames (spec target > 0.7)
      cloud_mse   — Dense cloud MSE
      rmse_wind   — RMSE of wind prediction at station pixels
      rmse_temp   — RMSE of temperature prediction at station pixels

    Collects predictions across ALL val batches before computing metrics
    so rain-class statistics are accurate even at low rain frequency.
    """
    model.eval()
    mask_flat = station_mask.bool().reshape(-1)  # (H*W,)

    all_true_rain: List[np.ndarray] = []
    all_pred_rain: List[np.ndarray] = []
    cloud_mse_sum = 0.0
    ssim_sum      = 0.0
    wind_sq_sum   = 0.0
    temp_sq_sum   = 0.0
    n_station_px  = 0
    n_batches     = 0

    for batch in val_loader:
        x            = batch["x"].to(device, non_blocking=True)
        y_cloud      = batch["y_sat"][:, 0].to(device, non_blocking=True)      # (B, 2, H, W) IR+WV at t+15
        y_thermo     = batch["y_thermo"][:, 0].to(device, non_blocking=True)   # (B, 2, H, W) wind+temp at t+15
        y_rain       = batch["y_rain"][:, 0].to(device, non_blocking=True)     # (B, H, W) rain class at t+15
        batch_mask   = batch["station_mask"].to(device, non_blocking=True)  # (B, H, W)

        with autocast("cuda", dtype=amp_dtype):
            drivers, y_cloud_pred, y_rain_logits = model(x)

        # Cloud MSE (dense, full 256×256)
        cloud_mse_sum += torch.nn.functional.mse_loss(y_cloud_pred, y_cloud).item()

        # SSIM (CPU, float32)
        true_np = y_cloud[:, 0].float().cpu().numpy()   # (B, H, W)
        pred_np = y_cloud_pred[:, 0].float().cpu().numpy()
        ssim_sum += compute_ssim_cloud(true_np, pred_np)

        # Station-pixel rain classification
        B = y_rain_logits.shape[0]
        logits_flat  = y_rain_logits.permute(0, 2, 3, 1).reshape(B, -1, y_rain_logits.shape[1])
        target_flat  = y_rain.reshape(B, -1)
        mask_flat_b  = batch_mask.reshape(B, -1)

        for b in range(B):
            active = mask_flat_b[b]
            if active.any():
                pred_cls = logits_flat[b][active].argmax(dim=-1).cpu().numpy()
                true_cls = target_flat[b][active].cpu().numpy()
                all_pred_rain.append(pred_cls)
                all_true_rain.append(true_cls)

        # Thermo RMSE at station pixels (wind ch0, temp ch1)
        bm = batch_mask.bool()  # (B, H, W)
        for b in range(B):
            px = bm[b]  # (H, W) bool
            if px.any():
                pred_w = drivers[b, 0][px].float()
                true_w = y_thermo[b, 0][px].float()
                pred_t = drivers[b, 1][px].float()
                true_t = y_thermo[b, 1][px].float()
                wind_sq_sum += ((pred_w - true_w) ** 2).sum().item()
                temp_sq_sum += ((pred_t - true_t) ** 2).sum().item()
                n_station_px += px.sum().item()

        n_batches += 1

    n = max(n_batches, 1)

    if all_true_rain:
        true_arr = np.concatenate(all_true_rain)
        pred_arr = np.concatenate(all_pred_rain)
        csi = compute_csi(true_arr, pred_arr, rain_threshold_class)
        hss = compute_hss(true_arr, pred_arr, rain_threshold_class)
    else:
        csi = float("nan")
        hss = float("nan")

    rmse_wind = float(np.sqrt(wind_sq_sum / max(n_station_px, 1)))
    rmse_temp = float(np.sqrt(temp_sq_sum / max(n_station_px, 1)))

    return {
        "csi":        csi,
        "hss":        hss,
        "ssim_cloud": ssim_sum / n,
        "cloud_mse":  cloud_mse_sum / n,
        "rmse_wind":  rmse_wind,
        "rmse_temp":  rmse_temp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per-horizon multi-step evaluation (Phase 3.1 — research-grade results)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_HORIZONS_STEPS  = (1, 2, 4, 8, 12, 16)   # 15, 30, 60, 120, 180, 240 min
DEFAULT_RAIN_THRESHOLDS = (1, 3, 6, 24)          # trace, light, moderate, heavy


def _build_next_frame_batch(
    y_cloud_pred: torch.Tensor,   # (B, 2, H, W) IR+WV
    dem:          torch.Tensor,   # (1, H, W) or (H, W)
) -> torch.Tensor:
    """Construct next 3-channel input frame [IR, WV, DEM] for batch autoregressive rollout."""
    B = y_cloud_pred.size(0)
    if dem.dim() == 2:
        dem = dem.unsqueeze(0)            # (1, H, W)
    dem_b = dem.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 1, H, W)
    return torch.cat([y_cloud_pred, dem_b], dim=1)  # (B, 3, H, W)


@torch.no_grad()
def evaluate_checkpoint_multihorizon(
    model,
    val_loader,
    device:          torch.device,
    amp_dtype:       torch.dtype,
    station_mask:    torch.Tensor,
    horizons_steps:  Sequence[int] = DEFAULT_HORIZONS_STEPS,
    rain_thresholds: Sequence[int] = DEFAULT_RAIN_THRESHOLDS,
    dem:             Optional[torch.Tensor] = None,
    out_path:        Optional[str] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Multi-step per-horizon evaluation.

    For each requested horizon (steps × 15 min), runs autoregressive rollout
    on every val batch and aggregates the metrics across the full val set.
    Rain skill scores swept across `rain_thresholds` for FAR/POD/CSI/HSS.

    Returns nested dict keyed by horizon-minutes:
      {
        15:  {csi@1, pod@1, far@1, hss@1, csi@3, ..., ssim_cloud, cloud_mse,
              rmse_wind, rmse_temp, n_samples},
        30:  {...},
        ...
        240: {...},
      }

    If `out_path` is provided, the dict is also written to that JSON file.
    """
    model.eval()
    horizons_steps = sorted(set(int(s) for s in horizons_steps))
    max_step = max(horizons_steps)
    rain_thresholds = tuple(int(t) for t in rain_thresholds)

    # Auto-locate DEM if not passed: assume val_loader.dataset.dem exists.
    if dem is None:
        dem = getattr(val_loader.dataset, "dem", None)
    if dem is None:
        raise ValueError("DEM tensor required for multi-step rollout (pass dem=...)")
    dem = dem.to(device)

    # Per-horizon accumulators.
    acc = {
        s: {
            "true_rain":   [],   # list of 1-D np arrays
            "pred_rain":   [],
            "cloud_mse":   0.0,
            "ssim":        0.0,
            "wind_sq":     0.0,
            "temp_sq":     0.0,
            "n_station":   0,
            "n_batches":   0,
            "n_samples":   0,
        }
        for s in horizons_steps
    }

    for batch in val_loader:
        x      = batch["x"].to(device, non_blocking=True)         # (B, 12, H, W)
        y_sat  = batch["y_sat"].to(device, non_blocking=True)     # (B, T, 2, H, W)
        y_thermo = batch["y_thermo"].to(device, non_blocking=True)  # (B, T, 2, H, W)
        y_rain = batch["y_rain"].to(device, non_blocking=True)    # (B, T, H, W)
        bmask  = batch["station_mask"].to(device, non_blocking=True)  # (B, H, W)
        valid  = batch["valid_steps"].to(device).long()           # (B,)

        B = x.size(0)
        current_x = x

        for step in range(1, max_step + 1):
            with autocast("cuda", dtype=amp_dtype):
                drivers, y_cloud_pred, y_rain_logits = model(current_x)

            if step in horizons_steps:
                a = acc[step]

                # Only score samples that have ground truth at this step.
                step_idx = step - 1  # 0-based into y_sat
                sample_valid = valid > step_idx                                  # (B,) bool
                if sample_valid.any():
                    sel = sample_valid.nonzero(as_tuple=True)[0]
                    pred_cloud_sel = y_cloud_pred[sel]                            # (Bv, 2, H, W)
                    true_cloud_sel = y_sat[sel, step_idx]                         # (Bv, 2, H, W)
                    pred_rain_sel  = y_rain_logits[sel].argmax(dim=1)             # (Bv, H, W)
                    true_rain_sel  = y_rain[sel, step_idx]                        # (Bv, H, W)
                    pred_drv_sel   = drivers[sel]                                 # (Bv, 2, H, W)
                    true_drv_sel   = y_thermo[sel, step_idx]                      # (Bv, 2, H, W)
                    bmask_sel      = bmask[sel].bool()                            # (Bv, H, W)
                    Bv = pred_cloud_sel.size(0)

                    # Cloud MSE (dense)
                    a["cloud_mse"] += torch.nn.functional.mse_loss(
                        pred_cloud_sel, true_cloud_sel
                    ).item()

                    # SSIM on IR channel
                    true_np = true_cloud_sel[:, 0].float().cpu().numpy()
                    pred_np = pred_cloud_sel[:, 0].float().cpu().numpy()
                    a["ssim"] += compute_ssim_cloud(true_np, pred_np)

                    # Rain + thermo at station pixels per sample
                    for bi in range(Bv):
                        px = bmask_sel[bi]
                        if px.any():
                            a["pred_rain"].append(pred_rain_sel[bi][px].cpu().numpy())
                            a["true_rain"].append(true_rain_sel[bi][px].cpu().numpy())
                            pw = pred_drv_sel[bi, 0][px].float()
                            tw = true_drv_sel[bi, 0][px].float()
                            pt = pred_drv_sel[bi, 1][px].float()
                            tt = true_drv_sel[bi, 1][px].float()
                            a["wind_sq"]   += ((pw - tw) ** 2).sum().item()
                            a["temp_sq"]   += ((pt - tt) ** 2).sum().item()
                            a["n_station"] += int(px.sum().item())

                    a["n_batches"] += 1
                    a["n_samples"] += Bv

            # Autoregressive shift if more steps to come
            if step < max_step:
                next_frame = _build_next_frame_batch(y_cloud_pred, dem)
                current_x  = torch.cat([current_x[:, 3:], next_frame], dim=1)

    # Aggregate metrics per horizon
    results: Dict[int, Dict[str, float]] = {}
    for s in horizons_steps:
        a = acc[s]
        horizon_min = s * 15
        n_b = max(a["n_batches"], 1)

        out: Dict[str, float] = {
            "cloud_mse":  a["cloud_mse"] / n_b,
            "ssim_cloud": a["ssim"]      / n_b,
            "rmse_wind":  float(np.sqrt(a["wind_sq"] / max(a["n_station"], 1))),
            "rmse_temp":  float(np.sqrt(a["temp_sq"] / max(a["n_station"], 1))),
            "n_samples":  float(a["n_samples"]),
            "n_station_px": float(a["n_station"]),
        }

        if a["true_rain"]:
            true_arr = np.concatenate(a["true_rain"])
            pred_arr = np.concatenate(a["pred_rain"])
            for thr in rain_thresholds:
                out[f"csi@{thr}"] = compute_csi(true_arr, pred_arr, thr)
                out[f"pod@{thr}"] = compute_pod(true_arr, pred_arr, thr)
                out[f"far@{thr}"] = compute_far(true_arr, pred_arr, thr)
                out[f"hss@{thr}"] = compute_hss(true_arr, pred_arr, thr)
        else:
            for thr in rain_thresholds:
                for m in ("csi", "pod", "far", "hss"):
                    out[f"{m}@{thr}"] = float("nan")

        results[horizon_min] = out

    if out_path is not None:
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        logger.info(f"Multi-horizon results → {op}")

    return results


def format_multihorizon_table(results: Dict[int, Dict[str, float]]) -> str:
    """Return a human-readable table summary of per-horizon results."""
    if not results:
        return "(no results)"

    horizons = sorted(results.keys())
    first    = results[horizons[0]]
    thr_keys = [k for k in first.keys() if k.startswith("csi@")]
    thrs     = sorted(int(k.split("@")[1]) for k in thr_keys)

    lines = []
    head = ["horizon"] + ["SSIM", "MSE", "RMSEw", "RMSEt"]
    for thr in thrs:
        head += [f"CSI@{thr}", f"POD@{thr}", f"FAR@{thr}"]
    lines.append("  ".join(f"{h:>10s}" for h in head))
    lines.append("  ".join(["-" * 10] * len(head)))

    for h_min in horizons:
        r = results[h_min]
        row = [
            f"T+{h_min:3d}min",
            f"{r.get('ssim_cloud', float('nan')):.3f}",
            f"{r.get('cloud_mse', float('nan')):.3f}",
            f"{r.get('rmse_wind', float('nan')):.3f}",
            f"{r.get('rmse_temp', float('nan')):.3f}",
        ]
        for thr in thrs:
            row += [
                f"{r.get(f'csi@{thr}', float('nan')):.3f}",
                f"{r.get(f'pod@{thr}', float('nan')):.3f}",
                f"{r.get(f'far@{thr}', float('nan')):.3f}",
            ]
        lines.append("  ".join(f"{c:>10s}" for c in row))

    return "\n".join(lines)
