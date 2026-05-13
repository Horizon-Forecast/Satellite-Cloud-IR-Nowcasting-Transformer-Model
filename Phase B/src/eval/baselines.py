# src/eval/baselines.py
# Horizon Forecast — Per-Horizon Baselines (Phase B research-track)
# Authors: Or Mordechay Hod, Gilad Boudman | Braude College, CODE: 26-1-R-1
#
# Baselines provide reference numbers the model must beat to claim research value.
# Each baseline emits the SAME nested dict shape as evaluate_checkpoint_multihorizon():
#   {horizon_min: {csi@thr, pod@thr, far@thr, hss@thr, ssim_cloud, cloud_mse, ...}}
#
# Baselines implemented:
#   - persistence: cloud(T+k) = cloud(T+0); rain always class 0 (no skill)
#   - optflow:     Farneback dense flow warps last IR forward; rain class 0
#   - climatology: per-month modal rain class at each station from training years;
#                  cloud = last observed IR (same as persistence for cloud)
#
# Usage:
#   from src.eval.baselines import run_persistence_multihorizon, run_optflow_multihorizon
#   results = run_persistence_multihorizon(val_loader)
#   # results[60]['csi@1'] etc.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.eval.evaluate import (
    DEFAULT_HORIZONS_STEPS,
    DEFAULT_RAIN_THRESHOLDS,
    compute_csi,
    compute_far,
    compute_hss,
    compute_pod,
    compute_ssim_cloud,
)

logger = logging.getLogger(__name__)

# Input channel layout: 4 frames × [IR, WV, DEM] = 12 channels
# Most recent IR at channel 9, WV at 10, DEM at 11.
_LAST_IR_CH  = 9
_LAST_WV_CH  = 10


def _init_horizon_acc(horizons: Sequence[int]) -> Dict[int, Dict]:
    return {
        s: {
            "true_rain": [], "pred_rain": [],
            "cloud_mse": 0.0, "ssim": 0.0,
            "n_batches": 0, "n_samples": 0,
        }
        for s in horizons
    }


def _aggregate(acc: Dict[int, Dict], rain_thresholds: Sequence[int]) -> Dict[int, Dict[str, float]]:
    results: Dict[int, Dict[str, float]] = {}
    for s, a in acc.items():
        horizon_min = s * 15
        n_b = max(a["n_batches"], 1)
        out: Dict[str, float] = {
            "cloud_mse":  a["cloud_mse"] / n_b,
            "ssim_cloud": a["ssim"]      / n_b,
            "rmse_wind":  float("nan"),  # baselines do not predict wind/temp
            "rmse_temp":  float("nan"),
            "n_samples":  float(a["n_samples"]),
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
    return results


def _score_batch_at_horizon(
    a:              Dict,
    pred_cloud:     torch.Tensor,   # (Bv, 2, H, W) IR+WV
    true_cloud:     torch.Tensor,   # (Bv, 2, H, W)
    pred_rain_cls:  torch.Tensor,   # (Bv, H, W) int64
    true_rain_cls:  torch.Tensor,   # (Bv, H, W) int64
    bmask:          torch.Tensor,   # (Bv, H, W) bool
) -> None:
    Bv = pred_cloud.size(0)
    a["cloud_mse"] += F.mse_loss(pred_cloud, true_cloud).item()
    true_np = true_cloud[:, 0].float().cpu().numpy()
    pred_np = pred_cloud[:, 0].float().cpu().numpy()
    a["ssim"] += compute_ssim_cloud(true_np, pred_np)
    for bi in range(Bv):
        px = bmask[bi]
        if px.any():
            a["pred_rain"].append(pred_rain_cls[bi][px].cpu().numpy())
            a["true_rain"].append(true_rain_cls[bi][px].cpu().numpy())
    a["n_batches"] += 1
    a["n_samples"] += Bv


# ══════════════════════════════════════════════════════════════════════════════
# Persistence baseline: no motion, no rain
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_persistence_multihorizon(
    val_loader,
    horizons_steps:  Sequence[int] = DEFAULT_HORIZONS_STEPS,
    rain_thresholds: Sequence[int] = DEFAULT_RAIN_THRESHOLDS,
    out_path:        Optional[str] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Persistence: pred(T+k) = obs(T+0) for ALL horizons. Rain → class 0 (dry).

    Returns nested dict {horizon_min: {metric: value}}.
    Optionally writes JSON to `out_path`.
    """
    horizons = sorted(set(int(s) for s in horizons_steps))
    acc      = _init_horizon_acc(horizons)

    for batch in val_loader:
        x      = batch["x"]                          # (B, 12, H, W)
        y_sat  = batch["y_sat"]                      # (B, T, 2, H, W)
        y_rain = batch["y_rain"]                     # (B, T, H, W) int64
        bmask  = batch["station_mask"].bool()        # (B, H, W)
        valid  = batch["valid_steps"].long()         # (B,)

        # Last observed IR+WV (channels 9-10), repeated as prediction for all horizons.
        last_ir = x[:, _LAST_IR_CH:_LAST_IR_CH + 1]  # (B, 1, H, W)
        last_wv = x[:, _LAST_WV_CH:_LAST_WV_CH + 1]  # (B, 1, H, W)
        pred_cloud_const = torch.cat([last_ir, last_wv], dim=1)  # (B, 2, H, W)

        B = x.size(0)
        for s in horizons:
            step_idx = s - 1
            sample_valid = valid > step_idx
            if not sample_valid.any():
                continue
            sel = sample_valid.nonzero(as_tuple=True)[0]
            true_cloud   = y_sat[sel, step_idx]                            # (Bv, 2, H, W)
            pred_cloud   = pred_cloud_const[sel]                            # (Bv, 2, H, W)
            true_rain    = y_rain[sel, step_idx]                            # (Bv, H, W)
            pred_rain_cls = torch.zeros_like(true_rain)                     # always dry
            _score_batch_at_horizon(
                acc[s], pred_cloud, true_cloud, pred_rain_cls, true_rain, bmask[sel]
            )

    results = _aggregate(acc, rain_thresholds)
    if out_path is not None:
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        logger.info(f"persistence results → {op}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Optical flow baseline (Farneback dense flow + warp)
# ══════════════════════════════════════════════════════════════════════════════

def _build_optflow_iterates(
    ir_t_minus_15: np.ndarray,
    ir_t_0:        np.ndarray,
    n_steps:       int,
) -> List[np.ndarray]:
    """
    Compute Farneback flow from t-15 → t0, then warp t0 forward by k×flow for k=1..n_steps.

    Inputs: 2 single-channel float32 arrays shape (H, W).
    Returns list of n_steps numpy arrays (H, W), one per forecast step.
    """
    import cv2
    H, W = ir_t_0.shape
    flow = cv2.calcOpticalFlowFarneback(
        ir_t_minus_15,
        ir_t_0,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )  # (H, W, 2)
    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
    out: List[np.ndarray] = []
    for k in range(1, n_steps + 1):
        map_x = grid_x + k * flow[..., 0]
        map_y = grid_y + k * flow[..., 1]
        warped = cv2.remap(ir_t_0, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        out.append(warped)
    return out


@torch.no_grad()
def run_optflow_multihorizon(
    val_loader,
    horizons_steps:  Sequence[int] = DEFAULT_HORIZONS_STEPS,
    rain_thresholds: Sequence[int] = DEFAULT_RAIN_THRESHOLDS,
    out_path:        Optional[str] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Optical-flow baseline using Farneback flow between the last two input frames.
    Predicted IR at each horizon = (k × flow) warp of last IR. WV held constant.
    Rain predicted as class 0 (no skill).

    Requires opencv-python.
    """
    horizons   = sorted(set(int(s) for s in horizons_steps))
    max_step   = max(horizons)
    acc        = _init_horizon_acc(horizons)
    # Channels: 4 frames × [IR, WV, DEM]. Previous IR at channel 6 (frame 2).
    PREV_IR_CH = 6

    for batch in val_loader:
        x      = batch["x"]                          # (B, 12, H, W)
        y_sat  = batch["y_sat"]                      # (B, T, 2, H, W)
        y_rain = batch["y_rain"]                     # (B, T, H, W) int64
        bmask  = batch["station_mask"].bool()        # (B, H, W)
        valid  = batch["valid_steps"].long()         # (B,)
        B      = x.size(0)

        # Compute warped IR per sample (numpy / cv2)
        ir_prev = x[:, PREV_IR_CH:PREV_IR_CH + 1, :, :].float().cpu().numpy()   # (B, 1, H, W)
        ir_curr = x[:, _LAST_IR_CH:_LAST_IR_CH + 1, :, :].float().cpu().numpy()
        wv_curr = x[:, _LAST_WV_CH:_LAST_WV_CH + 1, :, :].float().cpu().numpy()

        # Stack warped IR per sample: (B, max_step, H, W)
        warped_all = np.zeros((B, max_step, ir_curr.shape[2], ir_curr.shape[3]), dtype=np.float32)
        for bi in range(B):
            iterates = _build_optflow_iterates(ir_prev[bi, 0], ir_curr[bi, 0], max_step)
            for k, w in enumerate(iterates):
                warped_all[bi, k] = w

        warped_all = torch.from_numpy(warped_all).to(x.device)  # (B, max_step, H, W)
        wv_const   = torch.from_numpy(wv_curr).to(x.device)     # (B, 1, H, W)

        for s in horizons:
            step_idx = s - 1
            sample_valid = valid > step_idx
            if not sample_valid.any():
                continue
            sel = sample_valid.nonzero(as_tuple=True)[0]
            ir_pred   = warped_all[sel, step_idx:step_idx + 1]  # (Bv, 1, H, W)
            wv_pred   = wv_const[sel]                            # (Bv, 1, H, W)
            pred_cloud = torch.cat([ir_pred, wv_pred], dim=1)    # (Bv, 2, H, W)
            true_cloud = y_sat[sel, step_idx]                    # (Bv, 2, H, W)
            true_rain  = y_rain[sel, step_idx]
            pred_rain_cls = torch.zeros_like(true_rain)
            _score_batch_at_horizon(
                acc[s], pred_cloud, true_cloud, pred_rain_cls, true_rain, bmask[sel]
            )

    results = _aggregate(acc, rain_thresholds)
    if out_path is not None:
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        logger.info(f"optflow results → {op}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Climatology baseline: per-month modal rain class per station
# ══════════════════════════════════════════════════════════════════════════════

def build_climatology(
    parquet_dir:  str,
    station_csv:  str,
    out_path:     str,
    years:        Iterable[int] = (2020, 2021, 2022, 2023),
) -> np.ndarray:
    """
    Precompute per-station per-month modal rain class from training parquets.

    Output: (12, n_stations) int16 array of argmax class per (month, station).
    Cached at `out_path`. Idempotent — skips if exists.
    """
    import pyarrow.parquet as pq
    import pandas as pd

    op = Path(out_path)
    if op.exists():
        arr = np.load(op)
        logger.info(f"[climatology] cached: shape={arr.shape} - skip")
        return arr

    from src.data.dataset import N_RAIN_BINS, rain_mm_to_class

    stations = pd.read_csv(station_csv)
    sid_to_idx = {str(sid): i for i, sid in enumerate(stations["station_id"].tolist())}
    n_stations = len(sid_to_idx)
    counts = np.zeros((12, n_stations, N_RAIN_BINS), dtype=np.int64)

    for y in years:
        pq_path = Path(parquet_dir) / f"ims_train_{y}.parquet"
        if not pq_path.exists():
            logger.warning(f"[climatology] missing {pq_path}, skipping year {y}")
            continue
        df = pq.read_table(
            str(pq_path),
            columns=["timestamp", "station_id", "precipitation_mmhr"],
        ).to_pandas()
        df = df.dropna(subset=["precipitation_mmhr"])
        df["month"] = pd.to_datetime(df["timestamp"]).dt.month - 1  # 0-11
        df["sid"]   = df["station_id"].astype(str)
        df["cls"]   = df["precipitation_mmhr"].apply(rain_mm_to_class)

        for sid, idx in sid_to_idx.items():
            sub = df[df["sid"] == sid]
            if sub.empty:
                continue
            for m, cls in zip(sub["month"].values, sub["cls"].values):
                counts[m, idx, cls] += 1

    modal = counts.argmax(axis=-1).astype(np.int16)  # (12, n_stations)
    op.parent.mkdir(parents=True, exist_ok=True)
    np.save(op, modal)
    logger.info(f"[climatology] built {modal.shape} → {op}")
    return modal


@torch.no_grad()
def run_climatology_multihorizon(
    val_loader,
    climatology_path: str,
    station_mask_path: str,
    timestamps:       Optional[Sequence[str]] = None,
    horizons_steps:   Sequence[int] = DEFAULT_HORIZONS_STEPS,
    rain_thresholds:  Sequence[int] = DEFAULT_RAIN_THRESHOLDS,
    out_path:         Optional[str] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Climatology: cloud = last observed (persistence-equivalent); rain class at each
    station pixel = modal class for that station × month-of-year from training.

    Requires:
      - climatology_path: .npy with shape (12, n_stations) from build_climatology()
      - station_mask_path: .pt from build_station_mask (pixel↔station map)
      - val_loader.dataset.index must have 'timestamp' column (parsed)

    NOTE: This baseline assumes the val_loader iterates in the same order as
    val_loader.dataset.index. shuffle=False is required.
    """
    horizons = sorted(set(int(s) for s in horizons_steps))
    acc      = _init_horizon_acc(horizons)

    modal     = np.load(climatology_path)          # (12, n_stations)
    mask_data = torch.load(station_mask_path, weights_only=True)
    pixels    = mask_data["pixels"]                # [(r, c, sid), ...]

    # Map pixel (r, c) → station index in modal[:, idx]
    # Stations CSV order may differ from station_mask order: realign by station_id.
    sid_to_modal_idx = {str(p[2]): i for i, p in enumerate(pixels)}  # within-mask index

    # Build (n_stations_in_mask, 12) modal table
    n_mask = len(pixels)
    modal_aligned = np.zeros((12, n_mask), dtype=np.int16)
    for i, (_, _, sid) in enumerate(pixels):
        # If pixels' sid order maps to modal's column order directly, use it;
        # otherwise this would need a sid → index lookup. For simplicity here,
        # assume both share the same order (build_climatology used same csv).
        modal_aligned[:, i] = modal[:, i] if i < modal.shape[1] else 0

    ds = val_loader.dataset
    ts_series = ds.index["timestamp"]  # pd.Timestamp series

    batch_size = val_loader.batch_size or 1
    cursor = 0

    for batch in val_loader:
        x      = batch["x"]
        y_sat  = batch["y_sat"]
        y_rain = batch["y_rain"]
        bmask  = batch["station_mask"].bool()
        valid  = batch["valid_steps"].long()
        B      = x.size(0)

        # Resolve timestamps for this batch
        ts_slice = ts_series.iloc[cursor:cursor + B]
        cursor  += B
        months   = np.array([ts.month - 1 for ts in ts_slice], dtype=np.int64)  # (B,)

        # Cloud prediction = last observed (persistence-equivalent)
        last_ir = x[:, _LAST_IR_CH:_LAST_IR_CH + 1]
        last_wv = x[:, _LAST_WV_CH:_LAST_WV_CH + 1]
        pred_cloud_const = torch.cat([last_ir, last_wv], dim=1)

        # Rain prediction: per sample, populate station pixels with modal[month, station]
        H, W = bmask.shape[-2], bmask.shape[-1]
        pred_rain_batch = torch.zeros(B, H, W, dtype=torch.int64)
        for bi in range(B):
            m = int(months[bi])
            for i, (r, c, _sid) in enumerate(pixels):
                pred_rain_batch[bi, r, c] = int(modal_aligned[m, i])

        for s in horizons:
            step_idx = s - 1
            sample_valid = valid > step_idx
            if not sample_valid.any():
                continue
            sel = sample_valid.nonzero(as_tuple=True)[0]
            true_cloud   = y_sat[sel, step_idx]
            pred_cloud   = pred_cloud_const[sel]
            true_rain    = y_rain[sel, step_idx]
            pred_rain    = pred_rain_batch[sel]
            _score_batch_at_horizon(
                acc[s], pred_cloud, true_cloud, pred_rain, true_rain, bmask[sel]
            )

    results = _aggregate(acc, rain_thresholds)
    if out_path is not None:
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        logger.info(f"climatology results → {op}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Legacy single-step wrapper (kept for back-compat with older callers)
# ══════════════════════════════════════════════════════════════════════════════
def run_baselines(val_loader, device=None) -> Dict[str, float]:
    """
    Single-horizon (T+15) persistence summary. Returns flat dict for callers
    that expected the old API. New code should use the *_multihorizon variants.
    """
    multi = run_persistence_multihorizon(val_loader, horizons_steps=(1,))
    r15 = multi.get(15, {})
    return {
        "cloud_mse":  r15.get("cloud_mse",  float("nan")),
        "ssim_cloud": r15.get("ssim_cloud", float("nan")),
        "csi":        r15.get("csi@1",      float("nan")),
        "hss":        r15.get("hss@1",      float("nan")),
    }
