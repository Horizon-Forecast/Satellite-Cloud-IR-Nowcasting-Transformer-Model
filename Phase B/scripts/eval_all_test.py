"""
Held-out TEST evaluation — the delivered models, one shared (subsampled) test loader.

Test split = index_test.csv (2024-07..2025-12), never seen in train or selection.
The dataset normalizes the IR/WV band order by default (see HorizonDataset._load_sat),
so no channel flag is needed. Override the model list with HORIZON_EVAL_MODELS=
"Name:path,Name2:path2" to evaluate other checkpoints.

Usage:
  python scripts/eval_all_test.py --sub 10000 --horizons 1,2,3,4,6,8
Out: eval_results/test/<run>.json  +  SUMMARY.csv
"""
import argparse, json, sys, csv as _csv, random, os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.data.dataset as dset
from src.data.dataset  import get_dataloaders, HorizonDataset
from src.train.train   import load_model_for_inference
from src.eval.evaluate import evaluate_checkpoint_multihorizon, format_multihorizon_table

PROCESSED = ROOT / "data" / "processed"
THRESHOLDS = (1, 3, 6, 24)

# The two delivered models, committed in weights/ as FP16 checkpoints.
# To evaluate other/local checkpoints instead, set:
#   HORIZON_EVAL_MODELS="Cascade:checkpoints/phase2/gpu0_best.pt,Ablation:checkpoints/ablation/gpu0_best.pt"
MODELS_DEFAULT = [
    ("DriverFirst-Cascade", "weights/driver_first.pt"),
    ("EndToEnd-Ablation",   "weights/end_to_end.pt"),
]

# Optional override: HORIZON_EVAL_MODELS="Name:path,Name2:path2"
_ov = os.environ.get("HORIZON_EVAL_MODELS")
if _ov:
    MODELS = [(s.split(":", 1)[0], s.split(":", 1)[1]) for s in _ov.split(",") if ":" in s]
else:
    MODELS = MODELS_DEFAULT


def make_subset(n, seed=42):
    """Return a cached `index_test_sub{n}.csv` — a deterministic n-row random subset of the
    held-out test index (seeded, sorted). Builds it on first call, reuses it afterwards."""
    sub = PROCESSED / f"index_test_sub{n}.csv"
    if sub.exists():
        return sub
    import pandas as pd
    df = pd.read_csv(PROCESSED / "index_test.csv")
    random.seed(seed)
    idx = sorted(random.sample(range(len(df)), min(n, len(df))))
    df.iloc[idx].to_csv(sub, index=False)
    print(f"built {sub.name} ({len(idx)} rows)")
    return sub


def main():
    """CLI entry: evaluate every configured checkpoint on the held-out test set and print a
    comparison table. Flags select the subset size and the horizons to report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=6000, help="subsample size (0 = full)")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--horizons", default="1,2,3", help="comma steps, e.g. 1 or 1,2,3")
    ap.add_argument("--tag", default="", help="suffix for output files (avoid clobber)")
    args = ap.parse_args()
    HORIZONS = tuple(int(s) for s in args.horizons.split(","))

    # Channel order is normalized by default in HorizonDataset. No flag needed.
    suffix = args.tag

    test_csv = (PROCESSED / "index_test.csv") if args.sub == 0 else make_subset(args.sub)

    with (PROCESSED / "norm_stats.json").open() as f:
        norm_stats = {k: tuple(v) for k, v in json.load(f).items()}

    print(f"building test loader from {test_csv.name} ...", flush=True)
    _, test_loader = get_dataloaders(
        train_csv=str(PROCESSED / "index_train_subset.csv"),
        val_csv=str(test_csv),
        dem_path=str(PROCESSED / "dem_256.npy"),
        mask_path=str(PROCESSED / "station_mask.pt"),
        norm_stats=norm_stats, batch_size=args.batch_size, num_workers=4,
    )
    print(f"  test batches: {len(test_loader)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_data = torch.load(PROCESSED / "station_mask.pt", weights_only=True)
    station_mask = mask_data["mask"].to(device)
    dem_arr = torch.from_numpy(np.load(PROCESSED / "dem_256.npy").astype("float32"))
    dm, ds = norm_stats["dem"]
    dem = ((dem_arr - dm) / (ds + 1e-8)).unsqueeze(0).to(device)

    out_dir = ROOT / "eval_results" / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, rel in MODELS:
        ckpt = ROOT / rel
        if not ckpt.exists():
            print(f"[SKIP] {name}", flush=True); continue
        print(f"\n[EVAL{suffix}] {name}", flush=True)
        try:
            model = load_model_for_inference(str(ckpt), device=str(device), fp16=False)
            model.eval()
            res = evaluate_checkpoint_multihorizon(
                model=model, val_loader=test_loader, device=device,
                amp_dtype=torch.bfloat16, station_mask=station_mask,
                horizons_steps=HORIZONS, rain_thresholds=THRESHOLDS, dem=dem,
                out_path=str(out_dir / f"{name}{suffix}.json"),
            )
            h = res.get(HORIZONS[0] * 15) or res.get(str(HORIZONS[0] * 15)) or {}
            rows.append({"model": name, "CSI@1": h.get("csi@1"), "POD@1": h.get("pod@1"),
                         "FAR@1": h.get("far@1"), "CSI@24": h.get("csi@24"),
                         "SSIM": h.get("ssim_cloud"), "RMSE_wind": h.get("rmse_wind"),
                         "RMSE_temp": h.get("rmse_temp")})
            _c=h.get('csi@1'); _f=h.get('far@1'); _s=h.get('ssim_cloud')
            print(f"  CSI@1={_c} FAR={_f} SSIM={_s}", flush=True)
            del model; torch.cuda.empty_cache()
        except Exception as e:
            import traceback; print(f"[ERR] {name}: {e}"); traceback.print_exc()

    if rows:
        cols = ["model","CSI@1","POD@1","FAR@1","CSI@24","SSIM","RMSE_wind","RMSE_temp"]
        sp = out_dir / f"SUMMARY{suffix}.csv"
        with sp.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow({c: r.get(c) for c in cols})
        print(f"\nSUMMARY{suffix} -> {sp}", flush=True)
        for r in rows:
            print(f"  {r['model']:22s} CSI={r['CSI@1']} FAR={r['FAR@1']} SSIM={r['SSIM']}", flush=True)


if __name__ == "__main__":
    main()
