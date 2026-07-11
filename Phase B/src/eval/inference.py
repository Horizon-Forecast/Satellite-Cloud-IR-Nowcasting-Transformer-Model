"""
Multi-Step Auto-Regressive Inference (§6.1 feedback loop).

Implements the auto-regressive forecast loop: each step's predicted cloud IR frame is fed
back as the next step's most-recent input frame, producing T+15, T+30, T+45, T+60 min
(or more) forecasts from a single 4-frame input.

Channel layout in x (12 channels = 4 frames × [IR, WV, DEM]):
  ch 0-2  : t-45  [IR, WV, DEM]
  ch 3-5  : t-30  [IR, WV, DEM]
  ch 6-8  : t-15  [IR, WV, DEM]
  ch 9-11 : t-0   [IR, WV, DEM]  ← most recent

Auto-regressive shift (per step):
  new_ir  = predicted cloud (y_cloud from model output)
  new_wv  = last observed WV (channel 10), held constant (no WV model)
  new_dem = DEM (channel 11), always static
  next_x  = concat(current_x[:, 3:], [new_ir, new_wv, new_dem])

Usage:
  from src.eval.inference import run_multi_step_inference, run_inference_from_checkpoint
  preds = run_multi_step_inference(model, x, n_steps=4)
  # preds[i]["cloud"] = predicted cloud at T + (i+1)*15 min
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch import autocast

logger = logging.getLogger(__name__)

# Channel indices (constant for this project's input layout)
_IR_CH  = 0   # IR within a frame
_WV_CH  = 1   # WV within a frame
_DEM_CH = 2   # DEM within a frame
_FRAME_SIZE = 3
_N_FRAMES   = 4


@torch.no_grad()
def run_multi_step_inference(
    model,
    x: torch.Tensor,
    n_steps: int = 4,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> List[Dict[str, torch.Tensor]]:
    """
    Auto-regressive multi-step forecast.

    Args:
        model    : HorizonForecastModel (already on device, eval mode)
        x        : (1, 12, 256, 256) input tensor on device
        n_steps  : number of 15-min steps to forecast ahead (default 4 = 60 min)
        amp_dtype: bf16 on Ampere+/H100, fp16 on older GPUs

    Returns:
        List of n_steps dicts, each containing CPU float32 tensors:
          step       — int, 1-based step index
          lead_min   — int, lead time in minutes (step × 15)
          wind       — (1, 256, 256) predicted wind speed
          temp       — (1, 256, 256) predicted temperature
          cloud      — (1, 256, 256) predicted cloud IR
          rain_class — (1, 256, 256) predicted rain bin (argmax)
          rain_probs — (1, 64, 256, 256) full rain class probabilities
    """
    model.eval()
    results: List[Dict] = []
    current_x = x.clone()

    last_wv_ch  = (_N_FRAMES - 1) * _FRAME_SIZE + _WV_CH   # channel 10
    last_dem_ch = (_N_FRAMES - 1) * _FRAME_SIZE + _DEM_CH  # channel 11

    for step in range(1, n_steps + 1):
        with autocast("cuda", dtype=amp_dtype):
            drivers, y_cloud, y_rain_logits = model(current_x)

        results.append({
            "step":       step,
            "lead_min":   step * 15,
            "wind":       drivers[:, 0:1].float().cpu(),           # (1, 1, H, W)
            "temp":       drivers[:, 1:2].float().cpu(),           # (1, 1, H, W)
            "cloud":      y_cloud[:, 0:1].float().cpu(),           # (1, 1, H, W)
            "rain_class": y_rain_logits.argmax(dim=1, keepdim=True).cpu(),  # (1,1,H,W)
            "rain_probs": y_rain_logits.softmax(dim=1).float().cpu(),       # (1,64,H,W)
        })

        # Build next input: shift history by one frame, append new predicted frame.
        # Model predicts BOTH IR+WV (y_cloud is 2-ch), so feed both back. DEM static.
        new_frame = torch.cat([
            y_cloud[:, 0:2],                           # (1, 2, H, W) predicted IR + WV
            current_x[:, last_dem_ch:last_dem_ch + 1], # (1, 1, H, W) DEM (static)
        ], dim=1)  # (1, 3, H, W)

        # Drop oldest frame (channels 0-2), append new frame at end
        current_x = torch.cat([current_x[:, _FRAME_SIZE:], new_frame], dim=1)

    return results


def run_inference_from_checkpoint(
    checkpoint_path: str,
    x: torch.Tensor,
    n_steps: int = 4,
    device: str = "cuda",
    fp16: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    """
    Load a checkpoint and run multi-step inference.

    Args:
        checkpoint_path : path to .pt checkpoint saved by Trainer._save()
        x               : (1, 12, 256, 256) input tensor (CPU or any device)
        n_steps         : forecast steps (each = 15 min)
        device          : "cuda" or "cpu"
        fp16            : True = FP16 weights (faster on consumer GPU), False = FP32

    Returns:
        Same list of dicts as run_multi_step_inference().
    """
    from src.train.train import load_model_for_inference

    model = load_model_for_inference(checkpoint_path, device=device, fp16=fp16)
    amp_dtype = torch.float16 if fp16 else torch.float32
    x = x.to(device)
    return run_multi_step_inference(model, x, n_steps=n_steps, amp_dtype=amp_dtype)
