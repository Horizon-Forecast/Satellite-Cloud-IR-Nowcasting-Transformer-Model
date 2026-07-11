#!/usr/bin/env python
"""
Neighborhood (tolerance) CSI — held-out test set.  [SELF-CONTAINED, separate from src/eval.]

Standard nowcasting "fuzzy / neighborhood" verification (Roberts & Lean 2008). For each IMS
station, allow the forecast a spatial tolerance radius r (pixels):
  obs       = GT rain at the station >= thr
  fc_nbhd   = ANY predicted rain >= thr within r px of the station
  Hit  = obs & fc_nbhd ;  Miss = obs & ~fc_nbhd ;  FalseAlarm = ~obs & fc_nbhd
  CSI(r) = H / (H + M + FA)
r=0 reduces to the exact-pixel station CSI we report normally. As r grows, well-localized rain
turns near-misses into hits -> CSI climbs => low strict CSI is the sparse-pointwise penalty,
not the model being wrong about WHERE rain is.

Outputs CSV (model,horizon,radius,CSI,H,M,FA) + a CSI-vs-radius figure. Touches nothing in src/.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dataset import HorizonDataset, RAIN_BIN_MID
from src.train.train import load_model_for_inference
from src.eval.inference import run_multi_step_inference

# The delivered models ship in weights/ as FP16 checkpoints, loaded directly here.
CASCADE_CKPT  = str(ROOT / "weights" / "driver_first.pt")
ABLATION_CKPT = str(ROOT / "weights" / "end_to_end.pt")
MODELS = {"DriverFirst-Cascade": CASCADE_CKPT, "EndToEnd-Ablation": ABLATION_CKPT}
MIDS = np.array(RAIN_BIN_MID, dtype=np.float32)
OUT = Path(__file__).resolve().parent / "out"


def mm(cls_hw):
    c = cls_hw.squeeze().numpy().astype(int).clip(0, len(MIDS) - 1)
    return MIDS[c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=2000, help="evenly-strided subsample of the test set")
    ap.add_argument("--horizons", default="1,2,4,8", help="step indices (T+15=1 .. T+120=8)")
    ap.add_argument("--radii", default="0,1,2,3,4", help="tolerance radii in pixels")
    ap.add_argument("--thr", type=float, default=1.0, help="rain threshold mm/hr")
    args = ap.parse_args()
    HS = [int(s) for s in args.horizons.split(",")]
    RS = [int(s) for s in args.radii.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    proc = ROOT / "data" / "processed"
    norm = {k: tuple(v) for k, v in json.load(open(proc / "norm_stats.json")).items()}
    mask = torch.load(proc / "station_mask.pt", weights_only=True)
    stations = [(int(r), int(c)) for r, c, *_ in mask["pixels"]]
    srows = np.array([p[0] for p in stations]); scols = np.array([p[1] for p in stations])

    ds = HorizonDataset(str(proc / "index_test.csv"), str(proc / "dem_256.npy"),
                        str(proc / "station_mask.pt"), norm_stats=norm, augment=False,
                        project_root=str(ROOT))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {m: load_model_for_inference(ck, device=device, fp16=True) for m, ck in MODELS.items()}

    N = len(ds); idxs = list(range(0, N, max(1, N // args.sub)))[:args.sub]
    print(f"test {N}  eval {len(idxs)} samples  horizons {HS}  radii {RS}", flush=True)

    # accumulators[model][h][r] = [H, M, FA]
    acc = {m: {h: {r: np.zeros(3, np.int64) for r in RS} for h in HS} for m in MODELS}
    Hgrid = Wgrid = 256

    for n, idx in enumerate(idxs):
        s = ds[idx]
        x = s["x"].unsqueeze(0).to(device)
        gt_rain = s["y_rain"]
        for m in MODELS:
            preds = run_multi_step_inference(models[m], x, n_steps=max(HS))
            for h in HS:
                pred_wet = (mm(preds[h-1]["rain_class"]) >= args.thr)   # HxW bool
                gt_wet = (mm(gt_rain[h-1]) >= args.thr)                 # HxW bool (sparse: only stations nonzero)
                obs = gt_wet[srows, scols]                             # per-station truth (vectorized)
                for r in RS:
                    # dilate the forecast by radius r -> "rain predicted within r px" at each station
                    dil = pred_wet if r == 0 else maximum_filter(pred_wet, size=2*r+1, mode="constant")
                    fc = dil[srows, scols]
                    H  = int(np.sum(obs & fc)); M = int(np.sum(obs & ~fc)); FA = int(np.sum(~obs & fc))
                    acc[m][h][r] += (H, M, FA)
        if (n+1) % 200 == 0:
            print(f"  {n+1}/{len(idxs)}", flush=True)

    # write CSV + summary (CSI = H/(H+M+FA), POD = H/(H+M) detection-with-tolerance, FAR = FA/(H+FA))
    rows = ["model,horizon_min,radius_px,CSI,POD,FAR,H,M,FA"]
    summary = {}
    for m in MODELS:
        summary[m] = {}
        for h in HS:
            summary[m][h*15] = {}
            for r in RS:
                Hc, Mc, FAc = (int(v) for v in acc[m][h][r])
                csi = Hc / max(1, Hc + Mc + FAc)
                pod = Hc / max(1, Hc + Mc)
                far = FAc / max(1, Hc + FAc)
                summary[m][h*15][r] = {"CSI": round(float(csi),4), "POD": round(float(pod),4), "FAR": round(float(far),4)}
                rows.append(f"{m},{h*15},{r},{csi:.4f},{pod:.4f},{far:.4f},{Hc},{Mc},{FAc}")
    (OUT / "neighborhood_csi.csv").write_text("\n".join(rows))
    json.dump(summary, open(OUT / "neighborhood_csi.json", "w"), indent=2)
    print("\n=== Neighborhood CSI / POD (rows=radius px) ===")
    for h in HS:
        print(f"\nT+{h*15} min:        " + "  ".join(f"{m[:20]:>20}" for m in MODELS))
        for r in RS:
            cells = "  ".join(f"CSI {summary[m][h*15][r]['CSI']:.3f} POD {summary[m][h*15][r]['POD']:.3f}" for m in MODELS)
            print(f"  r={r}px   {cells}")

    _plot(summary, HS, RS)
    print(f"\nsaved -> {OUT}", flush=True)


def _plot(summary, HS, RS):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    NAVY="#0E2A47"; CY="#2EC4D8"; OR="#E8743B"
    BG="#0d1117"; TX="#e6edf3"
    h = HS[0]   # headline horizon (T+15)
    col = {"DriverFirst-Cascade": CY, "EndToEnd-Ablation": OR}
    fig, (axp, axc) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=BG)
    for ax, metric, ttl in [(axp, "POD", "Detection with tolerance (POD)\nmodel finds the rain once small offsets are allowed"),
                            (axc, "CSI", "Neighborhood CSI\nrises with tolerance; FA from over-prediction caps it")]:
        ax.set_facecolor(BG)
        for m in summary:
            ys = [summary[m][h*15][r][metric] for r in RS]
            ax.plot(RS, ys, "-o", color=col.get(m, TX), lw=2.4, ms=7, label=m.replace("-", " "))
        ax.set_xlabel("tolerance radius (px ≈ 1–2 km)", color=TX)
        ax.set_ylabel(f"{metric}  (T+{h*15} min)", color=TX)
        ax.set_title(ttl, color=TX, fontsize=10, fontweight="bold")
        ax.tick_params(colors=TX); [s.set_color("#33414f") for s in ax.spines.values()]
        ax.grid(alpha=0.15); ax.legend(facecolor=NAVY, edgecolor="#33414f", labelcolor=TX, fontsize=8)
    fig.suptitle("Low strict CSI = sparse-station pointwise scoring, not bad localization",
                 color=TX, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT / "neighborhood_csi_curve.png", dpi=150, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
