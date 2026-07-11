#!/usr/bin/env python
"""
Render the demo asset bank for the standalone .exe viewer.

For each curated case x {cascade=Driver-First, ablation=End-to-End} x horizon T+15..120:
  - static inputs (IR t-45..t0, WV, DEM)
  - prediction clouds+rain  vs  ground-truth clouds+rain
  - prediction wind/temp    vs  ground-truth wind/temp (ERA5 dense)
  - metrics: CSI, SSIM, RMSE_wind, RMSE_temp  ->  manifest.json

Reuses render_longhorizon.py machinery (inference, panels, geo overlay, metrics).
All panels share the georeferenced Israel crop + styling. Channel-fix on (fixed models).
"""
import json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))   # so `import render_longhorizon` resolves
import render_longhorizon as R
from src.data.dataset import HorizonDataset
from src.train.train import load_model_for_inference
from src.eval.inference import run_multi_step_inference

STEPS = [1, 2, 3, 4, 5, 6, 7, 8]                  # T+15 .. T+120
LEADS = [s * 15 for s in STEPS]
MODELS = {"cascade": R.CASCADE_CKPT, "ablation": R.ABLATION_CKPT}
MODEL_LABEL = {"cascade": "Driver-First Cascade", "ablation": "End-to-End (Ablation)"}

# The curated cases live in demo/cases.json (data, not code) so they can be edited without
# touching this script — each entry is {id, idx, type, label, time, caption}. The idx values are
# rows in data/processed/index_test.csv (deterministic for the delivered 2024-07..2025-12 test
# set). To change which cases the demo shows, edit that file. see the Maintenance Guide (B.8).
# Original selection: 3 DOMINANT cases (cascade strong, ablation misses the convective rain) +
# 5 CREDIBLE wins (ablation catches some rain, cascade clearly better) — comparison reads as real.
CASES_JSON = ROOT / "demo" / "cases.json"
CASES = json.loads(CASES_JSON.read_text(encoding="utf-8"))

OUT = ROOT / "demo" / "assets"
PANEL_W, PANEL_H = 2.0, 3.9      # tall-narrow Israel panel (inches)


def _new_ax(cbar=False):
    """Create a navy-background figure + full-bleed axis, reserving bottom space when a colorbar is needed."""
    fig = plt.figure(figsize=(PANEL_W, PANEL_H + (0.5 if cbar else 0.0)), facecolor=R._BG)
    ax = fig.add_axes([0.02, (0.13 if cbar else 0.02), 0.96, (0.83 if cbar else 0.96)])
    return fig, ax


def _save(fig, path):
    """Write the figure to `path` at 100 dpi on the navy background, then close it to free memory."""
    fig.savefig(path, dpi=100, facecolor=R._BG); plt.close(fig)


def _cbar(fig, ax, cmap, vmin, vmax, label, qual=False):
    """Add a slim horizontal colorbar under the panel. `qual=True` labels the ends light/heavy (rain)."""
    cax = fig.add_axes([0.12, 0.06, 0.76, 0.035])
    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_edgecolor(R._TEXT); cb.outline.set_linewidth(0.3)
    if qual:
        cb.set_ticks([vmin, vmax]); cax.set_xticklabels(["light", "heavy"])
    else:
        t = np.linspace(vmin, vmax, 3); cb.set_ticks(t)
        cax.set_xticklabels([f"{v:.0f}" for v in t])
    cax.tick_params(labelsize=6, colors=R._TEXT, length=1.5, width=0.3, pad=1)
    cb.set_label(label, fontsize=7, color=R.CYAN, labelpad=1)


def main():
    """Render the demo asset bank: for each curated case run both models and save input/GT/
    prediction PNGs (clouds+rain and wind/temp per horizon) plus a metrics manifest.json that
    demo/app.py reads. Optional arg re-renders a subset (see below)."""
    # arg: N (first N cases, smoke) OR comma-list of case ids (e.g. "case8" to re-render one).
    if len(sys.argv) > 1:
        a = sys.argv[1]
        cases = [c for c in CASES if c["id"] in a.split(",")] if a.startswith("case") else CASES[:int(a)]
    else:
        cases = CASES
    proc = ROOT / "data" / "processed"
    norm = {k: tuple(v) for k, v in json.load(open(proc / "norm_stats.json")).items()}
    mask = torch.load(proc / "station_mask.pt", weights_only=True)
    station_pixels = [(r, c) for r, c, *_ in mask["pixels"]]

    # wire render_longhorizon globals (geo overlay + coastline + stations)
    R.DEM_RAW = np.load(proc / "dem_256.npy").astype(np.float32)
    R.DEM_NP = (R.DEM_RAW - R.DEM_RAW.min()) / (R.DEM_RAW.max() - R.DEM_RAW.min() + 1e-8)
    R.STATIONS = station_pixels
    R.GEO = True
    R.GEO_BORDERS = R._load_borders()
    R.SHOW_STATIONS = True
    R.STATION_SIZE = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = HorizonDataset(str(proc / "index_test.csv"), str(proc / "dem_256.npy"),
                        str(proc / "station_mask.pt"), norm_stats=norm, augment=False,
                        project_root=str(ROOT), era5_npy_dir=str(proc.parent / "era5_npy"))
    models = {m: load_model_for_inference(ck, device=device, fp16=True) for m, ck in MODELS.items()}

    wm, ws = norm["wind"]; tm, ts = norm["temp"]
    def dn(t, m, s): return t.squeeze().numpy() * s + m
    manifest = {"models": MODEL_LABEL, "horizons": LEADS, "cases": []}

    for C in cases:
        idx = C["idx"]; cdir = OUT / C["id"]; cdir.mkdir(parents=True, exist_ok=True)
        disp = ds[idx]
        x = disp["x"].unsqueeze(0).to(device)
        gt_sat = disp["y_sat"]; gt_rain = disp["y_rain"]; gt_thermo = disp["y_thermo"]

        # one fixed IR scale from input history frames (training style)
        ir_hist = [R.denorm_ir(disp["x"][c*3:c*3+1], norm) for c in range(4)]
        IRV = (min(float(a.min()) for a in ir_hist), max(float(a.max()) for a in ir_hist))
        # driver scales from GT across horizons (full contrast)
        gw = [dn(gt_thermo[s-1, 0:1], wm, ws) for s in STEPS]
        gt = [dn(gt_thermo[s-1, 1:2], tm, ts) for s in STEPS]
        WV_ = np.percentile(np.concatenate([a.ravel() for a in gw]), (2, 98))
        TV_ = np.percentile(np.concatenate([a.ravel() for a in gt]), (2, 98))

        # ---- static inputs: IR + WV BOTH span all 4 history frames (model sees both every
        #      timestep — channels 3c=IR, 3c+1=WV, 3c+2=DEM). DEM is static. ----
        wvm, wvs = norm.get("wv", norm["ir"])
        wv_hist = [disp["x"][c*3+1:c*3+2].squeeze().numpy()*wvs+wvm for c in range(4)]
        WVV = (min(float(a.min()) for a in wv_hist), max(float(a.max()) for a in wv_hist))
        for c, lab in zip(range(4), ["t-45", "t-30", "t-15", "t0"]):
            tag = lab.replace('-', 'm')
            fig, ax = _new_ax(); R.panel_ir(ax, R.denorm_ir(disp["x"][c*3:c*3+1], norm), *IRV, f"IR {lab}")
            _save(fig, cdir / f"input_ir_{tag}.png")
            fig, ax = _new_ax(); R.panel_ir(ax, disp["x"][c*3+1:c*3+2].squeeze().numpy()*wvs+wvm, *WVV, f"WV {lab}")
            _save(fig, cdir / f"input_wv_{tag}.png")
        fig, ax = _new_ax(); ax.imshow(R.DEM_RAW, cmap="terrain", origin="upper", aspect=R.ASPECT)
        R._coast(ax); R._crop(ax); R._style_ax(ax, "DEM")
        _save(fig, cdir / "input_dem.png")

        # predictions
        preds = {m: run_multi_step_inference(models[m], x, n_steps=max(STEPS)) for m in MODELS}
        metrics = {m: {} for m in MODELS}
        for si, s in enumerate(STEPS):
            lead = LEADS[si]; H = f"T{lead:03d}"
            # GT clouds+rain
            bg = R.denorm_ir(gt_sat[s-1, 0:1], norm)
            fig, ax = _new_ax(); R.panel_rain(ax, bg, *IRV, R.rain_mm_field(gt_rain[s-1]), station_pixels, f"GT  T+{lead}")
            _save(fig, cdir / f"gt_cloudrain_{H}.png")
            fig, ax = _new_ax(cbar=True); R.panel_field(ax, dn(gt_thermo[s-1,0:1],wm,ws), R.CMAP_WIND, f"GT wind T+{lead}", *WV_)
            _cbar(fig, ax, R.CMAP_WIND, *WV_, "wind (m/s)"); _save(fig, cdir / f"gt_wind_{H}.png")
            fig, ax = _new_ax(cbar=True); R.panel_field(ax, dn(gt_thermo[s-1,1:2],tm,ts), R.CMAP_TEMP, f"GT temp T+{lead}", *TV_)
            _cbar(fig, ax, R.CMAP_TEMP, *TV_, "temp (°C)"); _save(fig, cdir / f"gt_temp_{H}.png")

            for m in MODELS:
                P = preds[m][s-1]
                pbg = R.denorm_ir(P["cloud"], norm)
                fig, ax = _new_ax(); R.panel_rain(ax, pbg, *IRV, R.rain_mm_field(P["rain_class"]), station_pixels, f"{MODEL_LABEL[m].split(' ')[0]}  T+{lead}")
                _save(fig, cdir / f"{m}_cloudrain_{H}.png")
                fig, ax = _new_ax(cbar=True); R.panel_field(ax, dn(P["wind"],wm,ws), R.CMAP_WIND, f"pred wind T+{lead}", *WV_)
                _cbar(fig, ax, R.CMAP_WIND, *WV_, "wind (m/s)"); _save(fig, cdir / f"{m}_wind_{H}.png")
                fig, ax = _new_ax(cbar=True); R.panel_field(ax, dn(P["temp"],tm,ts), R.CMAP_TEMP, f"pred temp T+{lead}", *TV_)
                _cbar(fig, ax, R.CMAP_TEMP, *TV_, "temp (°C)"); _save(fig, cdir / f"{m}_temp_{H}.png")
                # metrics
                csi = R.csi_at_stations(P["rain_class"], gt_rain[s-1], station_pixels)
                ssim = R.ssim_cloud(P["cloud"].squeeze().numpy(), gt_sat[s-1,0:1].squeeze().numpy())
                rw = float(np.sqrt(np.mean((dn(P["wind"],wm,ws) - dn(gt_thermo[s-1,0:1],wm,ws))**2)))
                rt = float(np.sqrt(np.mean((dn(P["temp"],tm,ts) - dn(gt_thermo[s-1,1:2],tm,ts))**2)))
                metrics[m][str(lead)] = {"CSI": round(csi,3), "SSIM": round(ssim,3),
                                         "RMSE_wind": round(rw,2), "RMSE_temp": round(rt,2)}
        manifest["cases"].append({**{k: C[k] for k in ("id","idx","type","label","time","caption")},
                                  "horizons": LEADS, "metrics": metrics})
        print(f"[done] {C['id']} idx={idx}  cascadeCSI@15={metrics['cascade']['15']['CSI']} "
              f"ablationCSI@15={metrics['ablation']['15']['CSI']}", flush=True)

    # Merge into any existing manifest (so a partial re-render updates only its cases),
    # then re-order to match the full CASES list.
    (OUT.parent).mkdir(parents=True, exist_ok=True)
    mpath = OUT.parent / "manifest.json"
    rendered = {c["id"]: c for c in manifest["cases"]}
    if len(cases) < len(CASES) and mpath.exists():
        old = {c["id"]: c for c in json.load(open(mpath)).get("cases", [])}
        old.update(rendered); rendered = old
    manifest["cases"] = [rendered[c["id"]] for c in CASES if c["id"] in rendered]
    json.dump(manifest, open(mpath, "w"), indent=2)
    print(f"\nMANIFEST -> {mpath}  ({len(manifest['cases'])} cases)", flush=True)


if __name__ == "__main__":
    main()
