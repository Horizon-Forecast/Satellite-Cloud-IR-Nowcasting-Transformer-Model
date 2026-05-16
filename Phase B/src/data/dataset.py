# src/data/dataset.py
# Horizon Forecast — Data Fusion Pipeline (Phase B)
# Authors: Or Mordechay Hod, Gilad Boudman  |  Braude College, CODE: 26-1-R-1
#
# Fuses 3 data sources into a unified (B, C_stacked=12, H=256, W=256) 4D tensor:
#   - EUMETSAT SEVIRI  : IR 10.8µm + WV 6.2µm  (dense, dynamic)
#   - NASA SRTM DEM    : Digital Elevation Model (dense, static)
#   - IMS Ground Stations: wind, temperature, precipitation (sparse, dynamic)

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ── Spatial grid constants (§3.1.1 of project doc) ─────────────────────────────
GRID_BOUNDS = dict(lat_min=29.0, lat_max=34.0, lon_min=34.0, lon_max=36.0)
GRID_H, GRID_W = 256, 256
T_IN      = 4    # 4 historical frames × 15 min = 60 min history
C_SAT     = 2    # IR + WV channels
C_STATIC  = 1    # DEM replicated per timestep
C_STACKED = T_IN * (C_SAT + C_STATIC)  # 12 — strict 4D constraint (no 5D video tensors)
N_RAIN_BINS = 64
T_ROLLOUT   = 16  # max autoregressive rollout steps (16 × 15min = 4h horizon)

# Exponential rain bin edges (mm/hr): dense resolution at low rain, captures extremes
_RAIN_EDGES = np.concatenate([
    [0.0],
    np.exp(np.linspace(np.log(0.1), np.log(50.0), N_RAIN_BINS - 1)),
    [np.inf],
])  # shape (65,) — 64 intervals
RAIN_BIN_MID = 0.5 * (_RAIN_EDGES[:-1] + np.minimum(_RAIN_EDGES[1:], 60.0))


# ── Geospatial helpers ──────────────────────────────────────────────────────────
def latlon_to_pixel(lat: float, lon: float) -> Tuple[int, int]:
    """
    Convert GPS coordinates to pixel (row, col) on the 256x256 grid.
    North = row 0 (top of image). Domain: [29N-34N] x [34E-36E].
    """
    row = int(
        (GRID_BOUNDS["lat_max"] - lat)
        / (GRID_BOUNDS["lat_max"] - GRID_BOUNDS["lat_min"])
        * GRID_H
    )
    col = int(
        (lon - GRID_BOUNDS["lon_min"])
        / (GRID_BOUNDS["lon_max"] - GRID_BOUNDS["lon_min"])
        * GRID_W
    )
    return max(0, min(GRID_H - 1, row)), max(0, min(GRID_W - 1, col))


def build_station_mask(
    station_csv: str,
    save_path: str = "data/processed/station_mask.pt",
) -> torch.Tensor:
    """
    Pre-compute sparse (256, 256) boolean mask: True at active IMS station pixels.
    Run this ONCE before training. Output saved to disk and loaded at Dataset init.

    station_csv must have columns: station_id, lat, lon
    Saved dict keys: 'mask' (H,W bool tensor), 'pixels' [(row,col,station_id),...]
    """
    df = pd.read_csv(station_csv)
    mask = torch.zeros(GRID_H, GRID_W, dtype=torch.bool)
    pixels: List[Tuple[int, int, str]] = []
    seen: set = set()

    for _, row in df.iterrows():
        r, c = latlon_to_pixel(float(row["lat"]), float(row["lon"]))
        if (r, c) in seen:
            continue  # two stations in same pixel — keep first, skip rest
        seen.add((r, c))
        mask[r, c] = True
        pixels.append((r, c, str(row["station_id"])))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mask": mask, "pixels": pixels}, save_path)
    n_total = len(df)
    n_kept  = len(pixels)
    logger.info(
        f"Station mask: {n_kept} unique pixels from {n_total} stations "
        f"({n_total - n_kept} duplicates merged) → {save_path}"
    )
    return mask


def rain_mm_to_class(rain_mm: float) -> int:
    """Map continuous precipitation (mm/hr) to discrete bin index [0, 63]."""
    return int(np.searchsorted(_RAIN_EDGES[1:], rain_mm, side="right"))


def compute_rain_class_weights(
    ims_df: pd.DataFrame,
    rain_col: str = "precipitation_mmhr",
) -> torch.Tensor:
    """
    Compute class weights for Class-Weighted Cross-Entropy Loss (§6.5, §6.6 eq.5).
    Addresses the extreme 98.84% dry-class dominance in Eastern Mediterranean data.

    Strategy: inverse-sqrt frequency + 10x boost for rain classes (class > 0).
    Returns (64,) float32 tensor — normalized so mean weight = 1.
    """
    counts = np.ones(N_RAIN_BINS, dtype=np.float64)  # Laplace smoothing
    for val in ims_df[rain_col].dropna():
        c = rain_mm_to_class(float(val))
        counts[c] += 1

    weights = 1.0 / np.sqrt(counts)
    weights[1:] *= 10.0       # penalize missing storm events heavily (False Negatives)
    weights /= weights.mean() # normalize: mean weight = 1
    return torch.tensor(weights, dtype=torch.float32)


# ── Main Dataset ────────────────────────────────────────────────────────────────
def _rebase_index(index: pd.DataFrame, project_root: str) -> pd.DataFrame:
    """
    Rewrite absolute path columns so they point to project_root instead of
    wherever data prep ran. Handles Windows↔Linux cross-platform (Colab).
    Finds '/data/' as the stable anchor between the machine-specific prefix
    and the portable relative tail.
    """
    path_cols = [c for c in index.columns if "path" in c]
    if not path_cols or index.empty:
        return index
    sample = str(index[path_cols[0]].iloc[0])
    anchor = "/data/"
    pos = sample.replace("\\", "/").find(anchor)
    if pos == -1:
        return index
    old_prefix = sample[:pos]
    new_prefix = str(project_root).rstrip("/\\")

    def _fix(p):
        if not isinstance(p, str):
            return p  # NaN / None — leave as-is, downstream loaders skip them
        return p.replace(old_prefix, new_prefix).replace("\\", "/")

    for col in path_cols:
        index[col] = index[col].apply(_fix)
    return index


class HorizonDataset(Dataset):
    """
    Multi-modal fusion dataset for Horizon Forecast precipitation nowcasting.

    Input tensor  x: (C_stacked=12, H=256, W=256)
      - 4 timesteps × [IR 10.8µm | WV 6.2µm | DEM] = 12 channels
      - Encodes 60 minutes of atmospheric history as image channels (4D, not 5D video)

    Target tensors (multi-step rollout, T_ROLLOUT=16 steps = 4h):
      y_sat       : (T_ROLLOUT, 2, H, W) float32  IR+WV future frames (Option A — both channels
                    predicted so model can feed its own output back as input)
      y_thermo    : (T_ROLLOUT, 2, H, W) float32  wind+temp at station pixels per step
      station_mask: (H, W) bool    True at active IMS station pixels (same for all steps)
      y_rain      : (T_ROLLOUT, H, W) int64  rain class [0-63] per step
      valid_steps : scalar int32   number of consecutive future steps with data (1..T_ROLLOUT)

    index_csv required columns (new rollout format):
        timestamp, sat_path_t0..t3           (input frames)
        sat_target_path_t1..t{T_ROLLOUT}     (.npy shape (2,H,W) per future step)
        ims_target_path_t1..t{T_ROLLOUT}     (IMS .csv per future step)
        valid_steps                           (int: consecutive steps with both sat+IMS)

    Legacy single-step format (sat_target_path / ims_target_path) is still accepted;
    valid_steps defaults to 1 in that case.

    Data split (chronological — no data leakage):
        Train: 2020-2023  |  Val: 2024 Jan-Jun  |  Test: 2024 Jul-2025 Dec
    """

    def __init__(
        self,
        index_csv: str,
        dem_path: str,
        station_mask_path: str,
        norm_stats: Optional[Dict] = None,
        augment: bool = False,
        project_root: Optional[str] = None,
    ):
        self.index = pd.read_csv(index_csv, parse_dates=["timestamp"])
        if project_root is not None:
            self.index = _rebase_index(self.index, project_root)
        self.augment = augment
        self.norm    = norm_stats or {}

        # Load static assets once at init (not per __getitem__ — avoids I/O overhead)
        dem_arr      = np.load(dem_path).astype(np.float32)
        # dem_raw kept in meters for visualization (contour thresholds work
        # in real elevation, not normalized z-scores). dem is the normalized
        # version fed into the model as one of the 12 input channels.
        self.dem_raw = torch.from_numpy(dem_arr).unsqueeze(0)  # (1, 256, 256) meters
        self.dem     = self.dem_raw.clone()
        if "dem" in self.norm:
            mean, std = self.norm["dem"]
            self.dem = (self.dem - mean) / (std + 1e-8)

        mask_data = torch.load(station_mask_path, weights_only=True)
        self.station_mask   = mask_data["mask"]     # (256, 256) bool
        self.station_pixels = mask_data["pixels"]   # [(r, c, sid), ...]
        self._sid_to_pixel  = {sid: (r, c) for r, c, sid in self.station_pixels}

    def __len__(self) -> int:
        return len(self.index)

    def _norm(self, t: torch.Tensor, key: str) -> torch.Tensor:
        if key in self.norm:
            mean, std = self.norm[key]
            return (t - mean) / (std + 1e-8)
        return t

    def _load_sat(self, path: str) -> torch.Tensor:
        """Load one SEVIRI frame → (2, H, W): channel 0=IR, channel 1=WV."""
        arr = np.load(path).astype(np.float32)  # (2, 256, 256)
        ir  = self._norm(torch.from_numpy(arr[0:1]), "ir")
        wv  = self._norm(torch.from_numpy(arr[1:2]), "wv")
        return torch.cat([ir, wv], dim=0)       # (2, 256, 256)

    def _load_ims(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse IMS CSV into sparse grid targets.

        y_thermo: (2, 256, 256) — wind+temp only at station pixels, 0 elsewhere.
                  Zeros outside stations are masked away by MaskedMSELoss — no
                  gradients flow from empty grid regions (§6.3, §6.6 eq.4).
        y_rain:   (256, 256) int64 — rain class [0-63] at station pixels, 0 elsewhere.
                  CE loss also computed only at station pixels.
        """
        df = pd.read_csv(path)
        y_thermo = torch.zeros(2, GRID_H, GRID_W, dtype=torch.float32)
        y_rain   = torch.zeros(GRID_H, GRID_W, dtype=torch.int64)

        for _, row in df.iterrows():
            sid = str(row["station_id"])
            if sid not in self._sid_to_pixel:
                continue
            r, c = self._sid_to_pixel[sid]

            wind = float(row.get("wind_speed_ms",     0.0))
            temp = float(row.get("temperature_c",     0.0))
            rain = float(row.get("precipitation_mmhr", 0.0))

            if "wind" in self.norm:
                mean, std = self.norm["wind"]
                wind = (wind - mean) / (std + 1e-8)
            if "temp" in self.norm:
                mean, std = self.norm["temp"]
                temp = (temp - mean) / (std + 1e-8)

            y_thermo[0, r, c] = wind
            y_thermo[1, r, c] = temp
            y_rain[r, c]      = rain_mm_to_class(rain)

        return y_thermo, y_rain

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.index.iloc[idx]

        # Stack T_IN frames along channel dim: [IR,WV,DEM] x 4 → (12, 256, 256)
        frames = []
        for t in range(T_IN):
            sat = self._load_sat(row[f"sat_path_t{t}"])    # (2, H, W)
            frames.append(torch.cat([sat, self.dem], dim=0))  # (3, H, W)
        x = torch.cat(frames, dim=0)  # (12, 256, 256)

        # Multi-step targets: allocate full T_ROLLOUT tensors (zeros = padding)
        y_sat    = torch.zeros(T_ROLLOUT, 2, GRID_H, GRID_W, dtype=torch.float32)
        y_thermo = torch.zeros(T_ROLLOUT, 2, GRID_H, GRID_W, dtype=torch.float32)
        y_rain   = torch.zeros(T_ROLLOUT, GRID_H, GRID_W, dtype=torch.int64)

        # Detect index format: new rollout (sat_target_path_t1) vs legacy (sat_target_path)
        if "sat_target_path_t1" in row.index:
            valid_steps = int(row.get("valid_steps", 1))
            for step in range(valid_steps):
                col_sat = f"sat_target_path_t{step + 1}"
                col_ims = f"ims_target_path_t{step + 1}"
                if col_sat in row.index and pd.notna(row[col_sat]):
                    y_sat[step] = self._load_sat(str(row[col_sat]))
                if col_ims in row.index and pd.notna(row[col_ims]):
                    th, r = self._load_ims(str(row[col_ims]))
                    y_thermo[step] = th
                    y_rain[step]   = r
        else:
            # Legacy single-step format — compatible with pre-rollout index CSVs
            valid_steps = 1
            y_sat[0] = self._load_sat(str(row["sat_target_path"]))
            th, r = self._load_ims(str(row["ims_target_path"]))
            y_thermo[0] = th
            y_rain[0]   = r

        mask = self.station_mask

        # Augmentation: horizontal flip only (preserves N-S lat gradient)
        if self.augment and torch.rand(1).item() > 0.5:
            x        = torch.flip(x,        dims=[-1])
            y_sat    = torch.flip(y_sat,    dims=[-1])
            y_thermo = torch.flip(y_thermo, dims=[-1])
            y_rain   = torch.flip(y_rain,   dims=[-1])
            mask     = torch.flip(mask,     dims=[-1])

        return {
            "x":            x,           # (12, 256, 256) float32
            "y_sat":        y_sat,       # (T_ROLLOUT, 2, 256, 256) float32  IR+WV per step
            "y_thermo":     y_thermo,    # (T_ROLLOUT, 2, 256, 256) float32  wind+temp per step
            "station_mask": mask,        # (256, 256) bool
            "y_rain":       y_rain,      # (T_ROLLOUT, 256, 256) int64
            "valid_steps":  torch.tensor(valid_steps, dtype=torch.int32),
        }


# ── DataLoader factory ──────────────────────────────────────────────────────────
def get_dataloaders(
    train_csv:    str,
    val_csv:      str,
    dem_path:     str,
    mask_path:    str,
    norm_stats:   Dict,
    batch_size:   int = 16,
    num_workers:  int = 8,
    project_root: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    H100-optimized DataLoader configuration:
      pin_memory=True          → zero-copy CPU→GPU transfer
      persistent_workers=True  → worker processes survive between epochs
      prefetch_factor=4        → 4 batches pre-loaded while GPU trains
      drop_last=True           → stable GroupNorm statistics (no single-sample batches)
    """
    train_ds = HorizonDataset(train_csv, dem_path, mask_path, norm_stats, augment=True,  project_root=project_root)
    val_ds   = HorizonDataset(val_csv,   dem_path, mask_path, norm_stats, augment=False, project_root=project_root)

    # Per-batch tensors are ~550 MB (mostly y_rain int64 for 16-step rollout × 256×256).
    # prefetch_factor capped at 2 to bound pinned RAM (else 26 GB pin demand at 12w/pf=4).
    kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True, **kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, **kwargs)
    return train_loader, val_loader
