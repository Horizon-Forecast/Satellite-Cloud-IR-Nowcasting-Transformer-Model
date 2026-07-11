#!/usr/bin/env python
"""
Long-horizon prediction images — Driver-First Cascade vs End-to-End Ablation, autoregressive
rollout to T+120 (2h). For each curated test case, renders 3 variants: combined (both + GT),
cascade-only, ablation-only. Panels: input IR history, Stage-1 drivers (predicted wind/temp),
cloud rollout vs GT, rain rollout vs GT.

Usage:
  venv\\Scripts\\python.exe scripts\\render_longhorizon.py --cases 1234,5678 --out scripts/figures/longhorizon
  (omit --cases to auto-pick N varied samples)
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from src.data.dataset import HorizonDataset, RAIN_BIN_MID
from src.train.train import load_model_for_inference
from src.eval.inference import run_multi_step_inference
from src.viz.visualize import (CMAP_CLOUD, CMAP_WIND, CMAP_TEMP, _BG, _PANEL, _TEXT,
                               _style_ax, _overlay_dem, _draw_coastline,
                               _scatter_stations, _scatter_station_rain)

# Poster-matching theme: navy background (matches the poster's navy header/palette)
# instead of the near-black default, so the figures harmonize with the poster.
_BG = "#0E2A47"; _PANEL = "#16314f"

# Radar-style precip colormap (NWS-like): light->blue->green->yellow->orange->red
RADAR = mcolors.LinearSegmentedColormap.from_list("radar", [
    "#9ec9ff", "#3a7bd5", "#2ecc71", "#a3e635", "#f1c40f", "#e67e22", "#e74c3c", "#8e1b1b"])


def _legend_strip_img(entries, width_px):
    """Render the colour key (value <-> colour) as its own self-contained strip image and
    return it as a PIL image. Composited onto the main PNG afterwards -> no layout fighting
    with the aspect-locked panels (which overflow their cells in every direction)."""
    from PIL import Image
    figL = plt.figure(figsize=(width_px / 110.0, 1.05), dpi=110, facecolor=_BG)
    n = len(entries); slot = 1.0 / n
    figL.text(0.012, 0.62, "COLOUR KEY  →", fontsize=9, color=CYAN, fontweight="bold",
              ha="left", va="center")
    for i, (cmap, vmin, vmax, label, qual) in enumerate(entries):
        cx = 0.14 + i * (0.83 / n)
        cax = figL.add_axes([cx, 0.40, 0.83 / n * 0.62, 0.15])
        sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap)
        cb = figL.colorbar(sm, cax=cax, orientation="horizontal")
        cb.outline.set_edgecolor(_TEXT); cb.outline.set_linewidth(0.4)
        if qual:
            cb.set_ticks([vmin, vmax]); cax.set_xticklabels(["light", "heavy"])
        else:
            ticks = np.linspace(vmin, vmax, 3); cb.set_ticks(ticks)
            cax.set_xticklabels([f"{t:.0f}" for t in ticks])
        cax.tick_params(labelsize=7, colors=_TEXT, length=2, width=0.4, pad=1.5)
        figL.text(cx + 0.83 / n * 0.31, 0.66, label, fontsize=8, color=CYAN,
                  ha="center", va="bottom")
    figL.canvas.draw()
    img = Image.frombytes("RGBA", figL.canvas.get_width_height(),
                          bytes(figL.canvas.buffer_rgba())).convert("RGB")
    plt.close(figL)
    return img

STEPS = [1, 2, 3, 4, 6, 8]               # T+15,30,45,60,90,120
LEADS = [s * 15 for s in STEPS]
# The delivered models ship in weights/ as FP16 checkpoints, so the demo/analysis
# tools load them directly — no separate training checkpoints required.
CASCADE_CKPT  = str(ROOT / "weights" / "driver_first.pt")
ABLATION_CKPT = str(ROOT / "weights" / "end_to_end.pt")
# Descriptive model names (proposed method first)
MODEL_CKPT = {"DriverFirst": CASCADE_CKPT, "EndToEnd": ABLATION_CKPT}
MODEL_ORDER = ["DriverFirst", "EndToEnd"]
LABEL = {"DriverFirst": "Driver-First Cascade", "EndToEnd": "End-to-End (Ablation)"}
SHORT = {"DriverFirst": "Driver-First", "EndToEnd": "End-to-End"}
CYAN = "#2EC4D8"
ASPECT = 2.5          # Israel tall-narrow proportions (matches training viz)
DEM_RAW = None        # DEM meters (coastline)
DEM_NP = None         # DEM normalized 0-1 (hillshade)
STATIONS = None       # IMS station pixels


def denorm_ir(t, norm):
    m, s = norm["ir"]; return t.squeeze().numpy() * s + m


def rain_mm_field(rain_cls_hw):
    """class id -> mm midpoint, full HxW. Class 0 is the DRY bin ([0,0.1) mm) whose midpoint
    is ~0.05 mm — force it to 0 so dry stations render NO rain (else a clear day paints a faint
    blob at every station)."""
    mids = np.array(RAIN_BIN_MID, dtype=np.float32)
    cls = rain_cls_hw.squeeze().numpy().astype(int).clip(0, len(mids) - 1)
    out = mids[cls]
    out[cls == 0] = 0.0
    return out


def ssim_cloud(pred_np, gt_np):
    from skimage.metrics import structural_similarity as ss
    dr = float(max(gt_np.max() - gt_np.min(), 1e-3))
    return ss(gt_np.astype(np.float32), pred_np.astype(np.float32), data_range=dr)


def csi_at_stations(pred_cls, gt_cls, stations, thr=1.0):
    mids = np.array(RAIN_BIN_MID, np.float32)
    pc = pred_cls.squeeze().numpy().astype(int); gc = gt_cls.squeeze().numpy().astype(int)
    p = np.array([mids[min(pc[r, c], len(mids) - 1)] >= thr for r, c in stations])
    g = np.array([mids[min(gc[r, c], len(mids) - 1)] >= thr for r, c in stations])
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0


def _metric_label(ax, txt):
    ax.text(0.5, 0.015, txt, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.5, color="#ffeb3b", fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="#0d1117", ec="none", alpha=0.65))


SHOW_STATIONS = True      # cloud/driver panels, set False to hide entirely
STATION_SIZE = 3          # small triangles (was 8)
GEO = False               # --geo: draw real country borders + cities instead of DEM contour
GEO_BORDERS = []          # list of (cols, rows) polylines in pixel space
GEO_CITIES = {"Tel Aviv": (34.78, 32.08), "Jerusalem": (35.21, 31.78),
              "Haifa": (34.99, 32.79), "Be'er Sheva": (34.79, 31.25), "Eilat": (34.95, 29.55)}
_GH, _GW = 256, 256


ISR_LON = (34.45, 35.95)   # crop window — focus on Israel (drop open sea / Sinai / E Jordan)
ISR_LAT = (29.35, 33.40)


def _lonlat_px(lon, lat):
    col = (lon - 34.0) / 2.0 * (_GW - 1)
    row = (34.0 - lat) / 5.0 * (_GH - 1)
    return col, row


def _crop(ax):
    """Zoom axes to the Israel bounding box (origin='upper': small row = north)."""
    x0, _ = _lonlat_px(ISR_LON[0], 0.0)
    x1, _ = _lonlat_px(ISR_LON[1], 0.0)
    _, ytop = _lonlat_px(0.0, ISR_LAT[1])     # north -> small row
    _, ybot = _lonlat_px(0.0, ISR_LAT[0])     # south -> large row
    ax.set_xlim(x0, x1)
    ax.set_ylim(ybot, ytop)


def _load_borders():
    import glob
    out = []
    for f in glob.glob(str(ROOT / "data" / "geo" / "*.geojson")):
        name = Path(f).stem
        try:
            g = json.load(open(f))["geometry"]
        except Exception:
            continue
        polys = g["coordinates"] if g["type"] == "Polygon" else [r for p in g["coordinates"] for r in p]
        rings = polys if g["type"] == "Polygon" else g["coordinates"]
        loops = (g["coordinates"] if g["type"] == "Polygon"
                 else [ring for poly in g["coordinates"] for ring in poly])
        for ring in loops:
            arr = np.array(ring)
            cols, rows = _lonlat_px(arr[:, 0], arr[:, 1])
            out.append((name, cols, rows))
    return out


def _draw_geo(ax):
    for name, cols, rows in GEO_BORDERS:
        is_isr = name == "israel"
        ax.plot(cols, rows, color=("#ffeb3b" if is_isr else "#6b7a8d"),
                lw=(1.4 if is_isr else 0.7), alpha=(0.95 if is_isr else 0.6), zorder=5)
    for nm, (lon, lat) in GEO_CITIES.items():
        c, r = _lonlat_px(lon, lat)
        ax.scatter([c], [r], s=10, c="#ffffff", edgecolors="#000", linewidths=0.4, zorder=7)
        ax.text(c + 2, r, nm, fontsize=4.5, color="#ffffff", va="center", zorder=7)


def _stations(ax):
    if SHOW_STATIONS:
        _scatter_stations(ax, STATIONS, size=STATION_SIZE)


def _coast(ax):
    if GEO:
        _draw_coastline(ax, DEM_RAW, color="#ff9d3a", linewidth=0.7, alpha=0.55)  # detailed coast/rift
        _draw_geo(ax)                                                              # national border + cities
    else:
        _draw_coastline(ax, DEM_RAW)


def panel_ir(ax, ir_np, vmin, vmax, title, stations=None):
    """Training-style clouds: grayscale IR + coastline + (small) station markers."""
    ax.imshow(ir_np, cmap=CMAP_CLOUD, vmin=vmin, vmax=vmax, origin="upper", aspect=ASPECT)
    _coast(ax)
    _stations(ax)
    _crop(ax)
    _style_ax(ax, title)


def panel_field(ax, arr, cmap, title, vmin=None, vmax=None):
    """Driver field (wind/temp) + coastline, optional shared vmin/vmax for comparability."""
    ax.imshow(arr, cmap=cmap, origin="upper", aspect=ASPECT, vmin=vmin, vmax=vmax)
    _coast(ax)
    _crop(ax)
    _style_ax(ax, title)


def panel_rain(ax, ir_backdrop, vmin, vmax, rain_mm, station_pixels, title):
    """Weather-radar rain over visible clouds: smooth precip blobs (not big circles)."""
    ax.imshow(ir_backdrop, cmap=CMAP_CLOUD, vmin=vmin, vmax=vmax, origin="upper",
              aspect=ASPECT, alpha=0.85)                       # clouds stay visible
    H, W = rain_mm.shape[-2:]
    sparse = np.zeros((H, W), np.float32)
    for r, c in station_pixels:
        sparse[r, c] = max(sparse[r, c], float(rain_mm[r, c]))
    field = gaussian_filter(sparse, sigma=3.2)                 # point sources -> small radar blobs
    if field.max() > 1e-3:
        fm = np.ma.masked_less(field, field.max() * 0.06)
        ax.imshow(fm, cmap=RADAR, vmin=0, vmax=field.max(), origin="upper",
                  aspect=ASPECT, alpha=0.72, zorder=4)
    # tiny station dots for location (don't hide the map)
    rows = [p[0] for p in station_pixels]; cols = [p[1] for p in station_pixels]
    ax.scatter(cols, rows, s=2.0, c="#cfd8e0", alpha=0.5, zorder=5, linewidths=0)
    _coast(ax)
    _crop(ax)
    _style_ax(ax, title)


def render_case(idx, ds, preds, norm, station_pixels, show, out_dir, timestamp):
    """show: list subset of ['DriverFirst','EndToEnd'] -> determines variant."""
    disp = ds[idx]
    x_disp = disp["x"]                      # (12,H,W)
    gt_sat = disp["y_sat"]                  # (16,2,H,W)
    gt_rain = disp["y_rain"]               # (16,H,W)

    # ONE fixed IR scale from the 4 input history frames (matches training viz) so
    # GT + all predictions render on the same scale and look comparable.
    ir_hist = [denorm_ir(x_disp[c*3:c*3+1], norm) for c in range(4)]
    IR_VMIN = min(float(a.min()) for a in ir_hist)
    IR_VMAX = max(float(a.max()) for a in ir_hist)

    # driver (wind/temp) ground truth (ERA5 dense) + shared scales across GT+predictions
    gt_thermo = disp["y_thermo"]                       # (16, 2, H, W) wind, temp
    wm, ws = norm["wind"]; tm, ts = norm["temp"]
    def _dn(t, m, s): return t.squeeze().numpy() * s + m
    # scale from GROUND TRUTH only so its real structure shows at full contrast.
    # predictions share the same scale (legend) for comparability and may clip if off-range.
    _wv = [_dn(gt_thermo[st-1, 0:1], wm, ws) for st in STEPS]
    _tv = [_dn(gt_thermo[st-1, 1:2], tm, ts) for st in STEPS]
    WIND_VMIN, WIND_VMAX = np.percentile(np.concatenate([a.ravel() for a in _wv]), (2, 98))
    TEMP_VMIN, TEMP_VMAX = np.percentile(np.concatenate([a.ravel() for a in _tv]), (2, 98))

    ncol = len(STEPS)
    rows = []                               # (label, kind, source)
    rows.append(("INPUT IR (t-45..t0)", "inputhist", None))
    rows.append(("GROUND TRUTH — wind (ERA5)", "wind_gt", None))
    for m in show: rows.append((f"{SHORT[m]} — predicted wind", "wind_pred", m))
    rows.append(("GROUND TRUTH — temperature (ERA5)", "temp_gt", None))
    for m in show: rows.append((f"{SHORT[m]} — predicted temp", "temp_pred", m))
    rows.append(("GROUND TRUTH — clouds", "cloud_gt", None))
    for m in show: rows.append((f"{LABEL[m]} — predicted clouds", "cloud_pred", m))
    rows.append(("GROUND TRUTH — rain (IMS stations)", "rain_gt", None))
    for m in show: rows.append((f"{LABEL[m]} — predicted rain", "rain_pred", m))

    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.0 * ncol, 1.9 * len(rows)),
                             facecolor=_BG)
    if len(rows) == 1: axes = axes[None, :]
    variant = "_".join(show).lower() if len(show) < 2 else "combined"
    fig.suptitle(f"Horizon Forecast — case {idx}  |  {timestamp}  |  {variant}",
                 fontsize=12, color=_TEXT, fontweight="bold")

    for ri, (label, kind, src) in enumerate(rows):
        # single fixed IR scale from input frames (training-style) for all cloud panels
        vmin, vmax = IR_VMIN, IR_VMAX
        for ci, s in enumerate(STEPS):
            ax = axes[ri, ci]
            if kind == "inputhist":
                # show 4 history frames in first 4 cols, blank rest (same IR scale)
                if ci < 4:
                    ir = denorm_ir(x_disp[ci*3:ci*3+1], norm)
                    panel_ir(ax, ir, IR_VMIN, IR_VMAX,
                             f"IR  t-{45-ci*15}min" if ci < 3 else "IR  t-0")
                else:
                    ax.axis("off")
                continue
            if kind == "drivers":
                if ci == 0:
                    panel_field(ax, preds[src][0]["wind"].squeeze().numpy(), "YlOrRd", "wind t0")
                elif ci == 1:
                    panel_field(ax, preds[src][0]["temp"].squeeze().numpy(), "RdBu_r", "temp t0")
                else:
                    ax.axis("off")
                continue
            lead = LEADS[ci]
            if kind == "wind_gt":
                panel_field(ax, _dn(gt_thermo[s-1,0:1],wm,ws), CMAP_WIND, f"GT wind T+{lead}", WIND_VMIN, WIND_VMAX)
            elif kind == "wind_pred":
                panel_field(ax, _dn(preds[src][s-1]["wind"],wm,ws), CMAP_WIND, f"{SHORT[src]} T+{lead}", WIND_VMIN, WIND_VMAX)
            elif kind == "temp_gt":
                panel_field(ax, _dn(gt_thermo[s-1,1:2],tm,ts), CMAP_TEMP, f"GT temp T+{lead}", TEMP_VMIN, TEMP_VMAX)
            elif kind == "temp_pred":
                panel_field(ax, _dn(preds[src][s-1]["temp"],tm,ts), CMAP_TEMP, f"{SHORT[src]} T+{lead}", TEMP_VMIN, TEMP_VMAX)
            elif kind == "cloud_gt":
                panel_ir(ax, denorm_ir(gt_sat[s-1,0:1],norm), vmin, vmax, f"GT T+{lead}")
            elif kind == "cloud_pred":
                pc = denorm_ir(preds[src][s-1]["cloud"],norm)
                panel_ir(ax, pc, vmin, vmax, f"{SHORT[src]} T+{lead}")
                # SSIM on NORMALIZED IR (matches src/eval/evaluate.py compute_ssim_cloud)
                pcn = preds[src][s-1]["cloud"].squeeze().numpy()
                gcn = gt_sat[s-1, 0:1].squeeze().numpy()
                _metric_label(ax, f"SSIM {ssim_cloud(pcn, gcn):.2f}")
            elif kind == "rain_gt":
                bg = denorm_ir(gt_sat[s-1,0:1],norm)
                panel_rain(ax, bg, vmin, vmax, rain_mm_field(gt_rain[s-1]), station_pixels, f"GT T+{lead}")
            elif kind == "rain_pred":
                bg = denorm_ir(preds[src][s-1]["cloud"],norm)
                panel_rain(ax, bg, vmin, vmax, rain_mm_field(preds[src][s-1]["rain_class"]), station_pixels, f"{SHORT[src]} T+{lead}")
                _metric_label(ax, f"CSI {csi_at_stations(preds[src][s-1]['rain_class'], gt_rain[s-1], station_pixels):.2f}")
    # NOTE: row labels are NOT set as ylabels — a wide ylabel/labelpad makes tight_layout
    # shove the whole grid right (panels overflow, blank cells fall off-canvas). Instead we
    # reserve a left margin via rect and write labels with fig.text (doesn't perturb layout).
    # reserve left band for row labels (fig.text doesn't perturb the grid layout)
    fig.tight_layout(rect=[0.16, 0, 0.99, 0.96])
    for ri, (label, kind, src) in enumerate(rows):
        yp = axes[ri, 0].get_position()
        fig.text(0.155, yp.y0 + yp.height/2, label, ha="right", va="center",
                 fontsize=6.5, color=CYAN)
    out = out_dir / f"case{idx:06d}_{variant}.png"
    fig.savefig(out, dpi=110, facecolor=_BG); plt.close(fig)

    # composite a colour-key strip on top (separate image -> immune to panel overflow)
    from PIL import Image
    entries = [(CMAP_WIND,  WIND_VMIN, WIND_VMAX, "wind (m/s)",       False),
               (CMAP_TEMP,  TEMP_VMIN, TEMP_VMAX, "temp (°C)",        False),
               (CMAP_CLOUD, IR_VMIN,   IR_VMAX,   "cloud IR (K)",     False),
               (RADAR,      0,         1,         "rain light→heavy", True)]
    with Image.open(out) as _m:        # context-close the read handle (Windows: else save no-ops)
        main = _m.convert("RGB")
    strip = _legend_strip_img(entries, main.width)
    canvas = Image.new("RGB", (main.width, main.height + strip.height), (13, 17, 23))
    canvas.paste(main, (0, 0)); canvas.paste(strip, (0, main.height))
    tmp = out.with_suffix(".tmp.png")
    canvas.save(str(tmp)); os.replace(str(tmp), str(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="comma idxs; empty=auto pick")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="scripts/figures/longhorizon")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hide-stations", action="store_true", help="hide station triangles on cloud/driver panels")
    ap.add_argument("--geo", action="store_true", help="real country borders + cities (georeferenced) instead of DEM coastline")
    ap.add_argument("--steps", default="", help="comma step indices to show, e.g. 1,2,3,4 = T+15..T+60 (default 1,2,3,4,6,8 = to T+120)")
    args = ap.parse_args()
    global SHOW_STATIONS, GEO, GEO_BORDERS, STEPS, LEADS
    if args.steps:
        STEPS = [int(s) for s in args.steps.split(",")]
        LEADS = [s * 15 for s in STEPS]
    if args.hide_stations:
        SHOW_STATIONS = False
    if args.geo:
        GEO = True
        GEO_BORDERS = _load_borders()
        print(f"loaded {len(GEO_BORDERS)} border rings", flush=True)

    proc = ROOT / "data" / "processed"
    with (proc / "norm_stats.json").open() as f:
        norm = {k: tuple(v) for k, v in json.load(f).items()}
    mask = torch.load(proc / "station_mask.pt", weights_only=True)
    station_pixels = [(r, c) for r, c, *_ in mask["pixels"]]
    global DEM_RAW, DEM_NP, STATIONS
    DEM_RAW = np.load(proc / "dem_256.npy").astype(np.float32)   # meters (coastline)
    DEM_NP = (DEM_RAW - DEM_RAW.min()) / (DEM_RAW.max() - DEM_RAW.min() + 1e-8)  # hillshade
    STATIONS = station_pixels

    ds = HorizonDataset(str(proc / "index_test.csv"), str(proc / "dem_256.npy"),
                        str(proc / "station_mask.pt"), norm_stats=norm, augment=False,
                        project_root=str(ROOT), era5_npy_dir=str(proc.parent / "era5_npy"))
    print(f"test samples: {len(ds)}", flush=True)

    models = {n: load_model_for_inference(MODEL_CKPT[n], device=args.device, fp16=True)
              for n in MODEL_ORDER}

    if args.cases:
        idxs = [int(c) for c in args.cases.split(",")]
    else:
        import random; random.seed(args.seed)
        idxs = random.sample(range(len(ds)), args.n)

    out_dir = ROOT / args.out; out_dir.mkdir(parents=True, exist_ok=True)
    for k, idx in enumerate(idxs):
        sample = ds[idx]
        x = sample["x"].unsqueeze(0).to(args.device)
        ts = ds.index.iloc[idx]["timestamp"]
        preds = {n: run_multi_step_inference(models[n], x, n_steps=max(STEPS)) for n in MODEL_ORDER}
        for show in (MODEL_ORDER, ["DriverFirst"], ["EndToEnd"]):
            p = render_case(idx, ds, preds, norm, station_pixels, show, out_dir, ts)
            print(f"[{k+1}/{len(idxs)}] {p.name}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
