#!/usr/bin/env python3
"""
runpod_rebase_csv.py
Rewrites absolute Windows paths in index CSVs to Linux RunPod paths.

Usage (run ON RunPod after extracting archives):
    python runpod_rebase_csv.py --data-root /workspace/data

The script rewrites all path columns in-place.
"""
import argparse
import pandas as pd
from pathlib import Path

WINDOWS_ROOT = "G:\\dev\\Horizon Forecast\\Phase B\\data\\processed"
WINDOWS_ROOT_ALT = "G:/dev/Horizon Forecast/Phase B/data/processed"

def rebase(path_str: str, linux_root: str) -> str:
    if not isinstance(path_str, str):
        return path_str
    p = path_str.replace("\\", "/")
    if WINDOWS_ROOT_ALT in p:
        return p.replace(WINDOWS_ROOT_ALT, linux_root)
    return p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/workspace/data/processed",
                        help="Linux path where processed data was extracted")
    args = parser.parse_args()

    linux_root = args.data_root.rstrip("/")
    csv_dir = Path(linux_root)

    for csv_file in ["index_train.csv", "index_train_subset.csv", "index_val.csv"]:
        path = csv_dir / csv_file
        if not path.exists():
            print(f"SKIP {csv_file} — not found at {path}")
            continue

        df = pd.read_csv(path)
        path_cols = [c for c in df.columns if "path" in c.lower()]
        for col in path_cols:
            df[col] = df[col].apply(lambda x: rebase(x, linux_root))

        df.to_csv(path, index=False)
        print(f"Rebased {csv_file} ({len(path_cols)} path columns)")

    print("Done. All CSVs point to Linux paths.")

if __name__ == "__main__":
    main()
