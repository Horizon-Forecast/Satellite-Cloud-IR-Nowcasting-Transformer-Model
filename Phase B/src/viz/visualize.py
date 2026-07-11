"""
Multi-Panel Forecast Visualization.

Generates a 4-row dashboard:
  Row 0: Input History    (4 IR satellite frames, T-60 to T-15 min)
  Row 1: Stage 1 Drivers  (predicted Wind Speed + Temperature)
  Row 2: Stage 2 Output   (predicted Cloud Structure + Rain Intensity)
  Row 3: Ground Truth     (optional — satellite cloud + IMS rain)
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from src.data.dataset import RAIN_BIN_MID, N_RAIN_BINS

# Colormaps
CMAP_CLOUD = "gray_r"
CMAP_TEMP  = "RdBu_r"
CMAP_WIND  = "YlOrRd"

# Custom rain colormap: white=dry, blue=trace, green=moderate, red=extreme, purple=violent
CMAP_RAIN = LinearSegmentedColormap.from_list(
    "rain_horizon",
    [
        (1.0, 1.0, 1.0, 0.0),  # Class 0: transparent (dry — dominant class)
        "#a6cee3",              # Class 1-5: trace rain
        "#1f78b4",              # Class 6-15: light rain
        "#33a02c",              # Class 16-25: moderate rain
        "#ffff00",              # Class 26-40: heavy rain
        "#ff7f00",              # Class 41-55: very heavy
        "#e31a1c",              # Class 56-62: extreme
        "#6600cc",              # Class 63: violent (>50 mm/hr)
    ],
    N=N_RAIN_BINS,
)

# Station-rain colormap: blue (light) -> red (heavy). Used for IMS station rain dots.
CMAP_STATION_RAIN = LinearSegmentedColormap.from_list(
    "station_rain",
    ["#2166ac", "#4393c3", "#92c5de", "#f4a582", "#d6604d", "#b2182b"],
    N=256,
)

# Dark theme palette for meteorological contrast
_BG    = "#0d1117"   # figure background
_PANEL = "#161b22"   # axes background
_TEXT  = "#e6edf3"   # titles and labels
_GRID  = "#30363d"   # spine and border color


def _prep(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    """Squeeze batch dim and convert to float32 numpy."""
    return t.squeeze().cpu().float().numpy() if t is not None else None


def _style_ax(ax: plt.Axes, title: str) -> None:
    """Apply dark theme styling to an axes panel."""
    ax.set_title(title, color=_TEXT, fontsize=10, pad=5, fontweight="bold")
    ax.set_facecolor(_PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.set_xticks([])
    ax.set_yticks([])


def _add_colorbar(
    fig: plt.Figure, im, ax: plt.Axes, label: str
) -> None:
    """Add styled colorbar to an axes panel."""
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
    cb.ax.yaxis.set_tick_params(color=_TEXT, labelcolor=_TEXT, labelsize=7)
    cb.set_label(label, color=_TEXT, fontsize=8)
    cb.outline.set_edgecolor(_GRID)


def _overlay_dem(ax: plt.Axes, dem: Optional[np.ndarray], alpha: float = 0.15) -> None:
    """Subtle grayscale DEM hillshade for elevation context.
    Switched from 'terrain' colormap (saturated rainbow) to 'gray' so it does
    NOT compete with the data colormaps on top. Low alpha keeps it as a
    background hint, not a foreground feature."""
    if dem is None:
        return
    ax.imshow(dem, cmap="gray", origin="upper", aspect=2.5,
              alpha=alpha, zorder=2, interpolation="bilinear")


def _draw_coastline(
    ax: plt.Axes,
    dem_raw: Optional[np.ndarray],
    threshold: float = 0.0,
    color:     str   = "#ffeb3b",
    linewidth: float = 1.5,
    alpha:     float = 0.9,
) -> None:
    """Mediterranean coast + Dead Sea rift boundary at 0m DEM elevation.

    NOTE: this is NOT a political border — it's an elevation contour. The line
    appears at MULTIPLE places where DEM crosses 0m:
      1. Mediterranean coast (sharp -450m sea / +50m land transition)
      2. Western + eastern walls of Jordan Rift Valley (land descending below 0m)
      3. Gulf of Aqaba near Eilat
    All are geographically real. The "extra" loops in middle-right are the rift
    boundary, not coastline misalignment.

    Previously drew a dashed -200m contour for Dead Sea — removed because the
    sharp sea/land gradient at the Mediterranean produces a spurious -200m line
    just offshore that looks like a 'westward bulge into the sea'."""
    if dem_raw is None:
        return
    # matplotlib `contour(origin='upper')` does NOT align with `imshow(origin='upper')`
    # when aspect != 1.0 — empirically verified north-south flip. Vertical flip the
    # DEM array before contour to match imshow's actual rendered orientation.
    # Tested against IR satellite cloud/sea boundary at multiple lat lines.
    dem_flipped = dem_raw[::-1, :]
    ax.contour(
        dem_flipped, levels=[threshold], colors=color,
        linewidths=linewidth, alpha=alpha, zorder=5,
        origin="upper", antialiased=True,
    )


def _scatter_stations(
    ax: plt.Axes,
    station_pixels: Optional[List[Tuple[int, int]]],
    size: int = 22,
) -> None:
    """Overlay IMS station locations as cyan triangles (visible on dark + light bg)."""
    if station_pixels and len(station_pixels) > 0:
        rows = [p[0] for p in station_pixels]
        cols = [p[1] for p in station_pixels]
        ax.scatter(
            cols, rows, s=size, c="#00ffd0", marker="^",
            zorder=6, alpha=0.95, linewidths=0.6, edgecolors="black",
        )


def _scatter_station_rain(
    ax: plt.Axes,
    station_pixels: Optional[List[Tuple[int, int]]],
    rain_field: np.ndarray,
    vmax: float,
):
    """Plot IMS stations as points colored by rain amount (blue=light -> red=heavy).

    Each station = small red center dot (location marker) + a larger halo whose
    color encodes the rain value sampled at that station pixel. Dry stations show
    as small faint blue points. Wetter stations grow redder. Returns the halo
    PathCollection for use as a colorbar mappable (None if no stations)."""
    if not station_pixels or len(station_pixels) == 0:
        return None
    rows = np.array([p[0] for p in station_pixels])
    cols = np.array([p[1] for p in station_pixels])
    h, w = rain_field.shape[-2:]
    rows = np.clip(rows, 0, h - 1)
    cols = np.clip(cols, 0, w - 1)
    vals = rain_field[rows, cols].astype(float)

    vmax = max(float(vmax), 0.5)
    # Halo size grows mildly with rain so heavy stations read bigger too.
    sizes = 45.0 + 95.0 * np.clip(vals / vmax, 0.0, 1.0)
    halo = ax.scatter(
        cols, rows, c=vals, cmap=CMAP_STATION_RAIN, vmin=0.0, vmax=vmax,
        s=sizes, alpha=0.92, linewidths=0.5, edgecolors="black", zorder=6,
    )
    # Small red center dot marking the station location.
    ax.scatter(cols, rows, c="#ff1a1a", s=5, marker="o", zorder=7, linewidths=0)
    return halo


def visualize_forecast(
    x_stacked:      torch.Tensor,            # (12, H, W) or (1, 12, H, W)
    pred_wind:      torch.Tensor,            # (H, W) or (1, H, W) — denormalized m/s
    pred_temp:      torch.Tensor,            # (H, W) or (1, H, W) — denormalized °C
    pred_cloud:     torch.Tensor,            # (H, W) or (1, H, W) — normalized BT
    pred_rain_cls:  torch.Tensor,            # (H, W) or (1, H, W) — integer [0-63]
    true_cloud:     Optional[torch.Tensor] = None,
    true_rain_cls:  Optional[torch.Tensor] = None,
    station_pixels: Optional[List[Tuple[int, int]]] = None,
    norm_stats:     Optional[Dict]         = None,
    dem:            Optional[torch.Tensor] = None,   # (1,H,W) or (H,W) — raw DEM for overlay
    title_suffix:   str                    = "",
    save_path:      Optional[str]          = None,
) -> plt.Figure:
    """
    Generate 4-row Horizon Forecast dashboard.

    Panel layout:
      Row 0: Input History — 4 IR frames (T-60 to T-15 min)
      Row 1: Stage 1 — Wind Speed (m/s) | Temperature (°C)
      Row 2: Stage 2 — Cloud T+15min    | Rain T+15min
      Row 3: Ground Truth (optional) — Satellite | IMS Rain

    Station locations shown as red triangle markers where applicable.
    Rain maps use log-scale colormap to reveal both light and extreme events.

    Args:
        x_stacked:     Stacked 12-channel input tensor (60 min history)
        pred_wind:     Stage 1 wind speed prediction
        pred_temp:     Stage 1 temperature prediction
        pred_cloud:    Stage 2 cloud structure prediction
        pred_rain_cls: Stage 2 rain class predictions (argmax of logits)
        true_cloud:    Ground truth satellite frame (optional, for validation)
        true_rain_cls: Ground truth IMS rain classes (optional)
        station_pixels: List of (row, col) station pixel coordinates
        norm_stats:    Normalization stats dict for denormalization (optional)
        title_suffix:  Timestamp string appended to figure title
        save_path:     If provided, save figure to this path at 150 DPI
    """
    # Prepare numpy arrays
    x_np    = _prep(x_stacked)                                               # (12,H,W)
    w_np    = _prep(pred_wind)
    t_np    = _prep(pred_temp)
    c_np    = _prep(pred_cloud)
    r_np    = RAIN_BIN_MID[_prep(pred_rain_cls).astype(int)]                # mm/hr
    gt_c    = _prep(true_cloud)
    gt_r    = (RAIN_BIN_MID[_prep(true_rain_cls).astype(int)]
               if true_rain_cls is not None else None)

    # DEM overlay: keep raw for coastline contour, normalised for hillshade.
    dem_np:  Optional[np.ndarray] = None
    dem_raw: Optional[np.ndarray] = None
    if dem is not None:
        d = _prep(dem).astype(np.float32)
        dem_raw = d
        d_min, d_max = d.min(), d.max()
        dem_np = (d - d_min) / (d_max - d_min + 1e-8)

    # Denormalize wind and temperature if stats provided
    if norm_stats:
        w_mean, w_std = norm_stats.get("wind", (0.0, 1.0))
        t_mean, t_std = norm_stats.get("temp", (0.0, 1.0))
        w_np = w_np * w_std + w_mean
        t_np = t_np * t_std + t_mean

    has_gt = gt_c is not None
    n_rows = 3 + int(has_gt)

    # Per-row height tuned for aspect=2.5 panels (real Israel geographic aspect):
    # tall-narrow panels need ~13 height per row to render without squishing.
    fig = plt.figure(figsize=(22, 12.5 * n_rows), facecolor=_BG)

    row_h = 1.0 / n_rows   # fractional height per row
    pad   = 0.012

    def _make_gs(row_idx: int, n_cols: int) -> gridspec.GridSpec:
        """Create a GridSpec for one row, positioned correctly in the figure."""
        top    = 0.96 - row_idx * row_h
        bottom = top - row_h + pad
        return gridspec.GridSpec(
            1, n_cols, figure=fig,
            left=0.04, right=0.97,
            top=top, bottom=bottom,
            wspace=0.06,
        )

    # Row 0: Input History (4 IR frames)
    T_IN      = 4
    gs0       = _make_gs(0, T_IN)
    ir_frames = [x_np[t * 3] for t in range(T_IN)]  # channel 0 = IR per 3-ch block
    vmin_ir   = min(f.min() for f in ir_frames)
    vmax_ir   = max(f.max() for f in ir_frames)
    t_labels  = ["T−60 min", "T−45 min", "T−30 min", "T−15 min"]

    for i, (frame, lbl) in enumerate(zip(ir_frames, t_labels)):
        ax = fig.add_subplot(gs0[0, i])
        im = ax.imshow(frame, cmap=CMAP_CLOUD, vmin=vmin_ir, vmax=vmax_ir,
                       origin="upper", aspect=2.5)
        _draw_coastline(ax, dem_raw)
        _scatter_stations(ax, station_pixels, size=10)
        _style_ax(ax, f"IR Satellite  {lbl}")
        if i == T_IN - 1:
            _add_colorbar(fig, im, ax, "BT (norm.)")

    # Row 1: Stage 1 — Thermodynamic Drivers
    gs1  = _make_gs(1, 2)

    ax_w = fig.add_subplot(gs1[0, 0])
    im_w = ax_w.imshow(w_np, cmap=CMAP_WIND, origin="upper", aspect=2.5)
    _overlay_dem(ax_w, dem_np, alpha=0.20)
    _draw_coastline(ax_w, dem_raw)
    _style_ax(ax_w, "Stage 1  →  Wind Speed (m/s)")
    _scatter_stations(ax_w, station_pixels)
    _add_colorbar(fig, im_w, ax_w, "m/s")

    ax_t = fig.add_subplot(gs1[0, 1])
    im_t = ax_t.imshow(t_np, cmap=CMAP_TEMP, origin="upper", aspect=2.5)
    _overlay_dem(ax_t, dem_np, alpha=0.20)
    _draw_coastline(ax_t, dem_raw)
    _style_ax(ax_t, "Stage 1  →  Surface Temperature (°C)")
    _scatter_stations(ax_t, station_pixels)
    _add_colorbar(fig, im_t, ax_t, "°C")

    # Row 2: Stage 2 — Storm Manifestation
    gs2       = _make_gs(2, 2)
    rain_vmax = max(float(r_np.max()), 1.0)

    ax_c = fig.add_subplot(gs2[0, 0])
    im_c = ax_c.imshow(c_np, cmap=CMAP_CLOUD, vmin=vmin_ir, vmax=vmax_ir,
                       origin="upper", aspect=2.5)
    _draw_coastline(ax_c, dem_raw)
    _scatter_stations(ax_c, station_pixels, size=14)
    _style_ax(ax_c, "Stage 2  →  Cloud Structure  T+15 min")
    _add_colorbar(fig, im_c, ax_c, "BT (norm.)")

    ax_r = fig.add_subplot(gs2[0, 1])
    im_r = ax_r.imshow(r_np, cmap=CMAP_RAIN, vmin=0, vmax=rain_vmax,
                       origin="upper", aspect=2.5)
    _draw_coastline(ax_r, dem_raw)
    _scatter_stations(ax_r, station_pixels)
    _style_ax(ax_r, "Stage 2  →  Rain Intensity  T+15 min")
    _add_colorbar(fig, im_r, ax_r, "mm/hr")

    # Row 3: Ground Truth (optional)
    if has_gt:
        gs3 = _make_gs(3, 3)  # 3 panels: GT cloud | GT cloud + pred rain overlay | GT rain

        ax_gc = fig.add_subplot(gs3[0, 0])
        im_gc = ax_gc.imshow(gt_c, cmap=CMAP_CLOUD, vmin=vmin_ir, vmax=vmax_ir,
                              origin="upper", aspect=2.5)
        _draw_coastline(ax_gc, dem_raw)
        _scatter_stations(ax_gc, station_pixels, size=14)
        _style_ax(ax_gc, "Ground Truth  →  Satellite Cloud")
        _add_colorbar(fig, im_gc, ax_gc, "BT (norm.)")

        # GT cloud + predicted rain overlay — for human visual comparison
        ax_ov = fig.add_subplot(gs3[0, 1])
        ax_ov.imshow(gt_c, cmap=CMAP_CLOUD, vmin=vmin_ir, vmax=vmax_ir,
                     origin="upper", aspect=2.5)
        _draw_coastline(ax_ov, dem_raw)
        rain_mask = r_np > 1.0  # only show pixels with predicted rain > 1 mm/hr
        if rain_mask.any():
            norm_r = np.clip(r_np / max(rain_vmax, 1.0), 0, 1)
            rgba   = CMAP_RAIN(norm_r)                              # (H, W, 4)
            rgba[..., 3] = np.where(rain_mask, 0.65, 0.0)          # semi-transparent where rain
            ax_ov.imshow(rgba, origin="upper", aspect=2.5, zorder=4)
        _scatter_stations(ax_ov, station_pixels, size=14)
        _style_ax(ax_ov, "GT Cloud  +  Predicted Rain Overlay")

        if gt_r is not None:
            ax_gr = fig.add_subplot(gs3[0, 2])
            ax_gr.imshow(gt_c, cmap=CMAP_CLOUD, vmin=vmin_ir, vmax=vmax_ir,
                         origin="upper", aspect=2.5, alpha=0.55)
            _draw_coastline(ax_gr, dem_raw)
            halo = _scatter_station_rain(ax_gr, station_pixels, gt_r, rain_vmax)
            _style_ax(ax_gr, "Ground Truth  →  IMS Rain (Station Points)")
            if halo is not None:
                _add_colorbar(fig, halo, ax_gr, "mm/hr")

    # Title
    fig.suptitle(
        f"Horizon Forecast  ·  Driver-First Cascade Nowcast  {title_suffix}",
        color=_TEXT, fontsize=13, y=0.995, fontweight="bold",
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"Saved visualization -> {save_path}")

    return fig


def visualize_rollout(
    steps_data:     list,                            # list of dicts from _viz_rollout_sample
    station_pixels: Optional[List[Tuple[int, int]]] = None,
    norm_stats:     Optional[Dict]                  = None,
    dem:            Optional[torch.Tensor]          = None,
    ssim_threshold: float                           = 0.25,
    title_suffix:   str                             = "",
    save_path:      Optional[str]                   = None,
) -> plt.Figure:
    """
    Multi-row rollout comparison figure. One row per predicted timestep.

    Columns per row:
      [0] Predicted IR   [1] Real IR   [2] Predicted Rain   [3] Real Rain   [4] SSIM bar

    Rows are green-bordered while SSIM is healthy, red-bordered when below threshold.

    steps_data: list of dicts with keys:
        step, lead_min, pred_ir, pred_rain_cls, true_ir, true_rain_cls, ssim
    """
    if not steps_data:
        fig, ax = plt.subplots(facecolor=_BG)
        ax.set_facecolor(_PANEL)
        ax.text(0.5, 0.5, "No rollout steps", color=_TEXT, ha="center", va="center",
                transform=ax.transAxes)
        return fig

    # DEM overlay + coastline
    dem_np:  Optional[np.ndarray] = None
    dem_raw: Optional[np.ndarray] = None
    if dem is not None:
        d = _prep(dem).astype(np.float32)
        dem_raw = d
        d_min, d_max = d.min(), d.max()
        dem_np = (d - d_min) / (d_max - d_min + 1e-8)

    n_rows = len(steps_data)
    n_cols = 5  # pred_IR | real_IR | pred_rain | real_rain | SSIM bar

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4.5, n_rows * 8.0),  # tall panels for aspect=2.5
        facecolor=_BG,
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.35], "wspace": 0.05, "hspace": 0.25},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing

    # IR range is computed PER ROW (per step) from that row's real IR, so each row
    # uses its full contrast range instead of one global scale. A global scale made
    # bright/low-contrast steps (e.g. T+15) look washed-out gray while colder steps
    # (T+30) looked rich. Per-row keeps every row visually consistent in style.
    # Pred and real in the same row share the row's scale so they stay comparable.

    all_rain_mm = [RAIN_BIN_MID[sd["pred_rain_cls"].numpy().astype(int)] for sd in steps_data]
    rain_vmax = max(float(a.max()) for a in all_rain_mm) if all_rain_mm else 1.0
    rain_vmax = max(rain_vmax, 1.0)

    for i, sd in enumerate(steps_data):
        ssim_val  = sd["ssim"]
        lead_min  = sd["lead_min"]
        healthy   = ssim_val >= ssim_threshold
        border_c  = "#33cc66" if healthy else "#cc3333"

        pred_ir   = sd["pred_ir"].numpy()
        true_ir   = sd["true_ir"].numpy()
        pred_rain = RAIN_BIN_MID[sd["pred_rain_cls"].numpy().astype(int)]
        true_rain = RAIN_BIN_MID[sd["true_rain_cls"].numpy().astype(int)]

        # Per-row IR scale from this step's real IR (full contrast, no global washout).
        vmin_ir = float(true_ir.min())
        vmax_ir = float(true_ir.max())
        if vmax_ir - vmin_ir < 1e-6:
            vmax_ir = vmin_ir + 1e-6

        # [0] Predicted IR
        ax = axes[i, 0]
        im = ax.imshow(pred_ir, cmap=CMAP_CLOUD, vmin=vmin_ir, vmax=vmax_ir,
                       origin="upper", aspect=2.5)
        _draw_coastline(ax, dem_raw)
        _scatter_stations(ax, station_pixels, size=10)
        _style_ax(ax, f"T+{lead_min}min  Pred IR")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_c)
            spine.set_linewidth(2.0)

        # [1] Real IR
        ax = axes[i, 1]
        ax.imshow(true_ir, cmap=CMAP_CLOUD, vmin=vmin_ir, vmax=vmax_ir,
                  origin="upper", aspect=2.5)
        _draw_coastline(ax, dem_raw)
        _scatter_stations(ax, station_pixels, size=10)
        _style_ax(ax, f"T+{lead_min}min  Real IR")

        # [2] Predicted Rain (dense model field)
        ax = axes[i, 2]
        ax.imshow(pred_rain, cmap=CMAP_RAIN, vmin=0, vmax=rain_vmax,
                  origin="upper", aspect=2.5)
        _draw_coastline(ax, dem_raw)
        _scatter_stations(ax, station_pixels)
        _style_ax(ax, f"T+{lead_min}min  Pred Rain")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_c)
            spine.set_linewidth(2.0)

        # [3] Real Rain (IMS stations as blue->red dots over faint IR backdrop)
        ax = axes[i, 3]
        ax.imshow(true_ir, cmap=CMAP_CLOUD, vmin=vmin_ir, vmax=vmax_ir,
                  origin="upper", aspect=2.5, alpha=0.5)
        _draw_coastline(ax, dem_raw)
        _scatter_station_rain(ax, station_pixels, true_rain, rain_vmax)
        _style_ax(ax, f"T+{lead_min}min  Real Rain")

        # [4] SSIM bar
        ax = axes[i, 4]
        ax.set_facecolor(_PANEL)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        bar_color = "#33cc66" if healthy else "#cc3333"
        ax.barh(0.5, ssim_val, height=0.55, color=bar_color, align="center")
        ax.axvline(ssim_threshold, color="#ffaa00", linewidth=1.5, linestyle="--")
        ax.text(ssim_val / 2, 0.5, f"{ssim_val:.2f}", color="white",
                ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(0.5, 0.08, "SSIM", color=_TEXT, ha="center", va="bottom",
                fontsize=8, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.set_facecolor(_PANEL)

    last_sd   = steps_data[-1]
    stopped   = "✓ still valid" if last_sd["ssim"] >= ssim_threshold else f"✗ stopped (SSIM<{ssim_threshold})"
    fig.suptitle(
        f"Horizon Forecast  ·  Autoregressive Rollout  {title_suffix}\n"
        f"Reached T+{last_sd['lead_min']}min ({len(steps_data)} steps)  |  {stopped}",
        color=_TEXT, fontsize=12, y=1.01, fontweight="bold",
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=_BG)
        print(f"Saved rollout visualization -> {save_path}")

    return fig
