"""
Build index_test.csv — held-out chronological test split (2024-07-01 .. 2025-12-31).
Mirrors src/data/prep.py build_indices() test-row logic exactly (history=4, rollout=16)
but emits ONLY the test split, so it does not touch index_train/index_val.

Run: python scripts/build_test_index.py
"""
import sys
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SAT_ROOT  = PROCESSED / "sat_npy"
SNAP_ROOT = PROCESSED / "ims_snapshots"
OUT       = PROCESSED / "index_test.csv"

HISTORY = 4
ROLLOUT = 16
DT      = pd.Timedelta("15min")


def index_by_ts(root: Path, suffix: str):
    """Map UTC timestamp → file path for every `*.{suffix}` under `root`, parsing the
    `YYYYMMDD_HHMM` stem. Files with an unparseable name are skipped."""
    out = {}
    for f in sorted(root.glob(f"**/*.{suffix}")):
        try:
            d, t = f.stem.split("_")[:2]
            ts = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}", tz="UTC")
            out[ts] = f
        except Exception:
            continue
    return out


def main():
    """Build `index_test.csv`: join satellite/thermo/rain frames by timestamp over the
    held-out window (history=4, rollout=16) and write one row per valid sample. No-op if it
    already exists."""
    if OUT.exists():
        print(f"{OUT.name} already exists ({len(pd.read_csv(OUT))} rows). Delete to rebuild.")
        return

    print("indexing sat_npy ...")
    sat_files = index_by_ts(SAT_ROOT, "npy")
    print(f"  sat timestamps: {len(sat_files)}")

    print("indexing ims_snapshots ...")
    snap_files = index_by_ts(SNAP_ROOT, "csv")
    print(f"  snapshot timestamps: {len(snap_files)}")

    rows_test = []
    for anchor_ts in tqdm(sorted(sat_files), desc="test indices"):
        yr, mo = anchor_ts.year, anchor_ts.month
        in_test = (yr == 2024 and mo >= 7) or yr == 2025
        if not in_test:
            continue

        history_ts = [anchor_ts - (HISTORY - i) * DT for i in range(HISTORY)]
        if not all(ts in sat_files for ts in history_ts):
            continue

        row = {"timestamp": anchor_ts.isoformat()}
        for i, ts in enumerate(history_ts):
            row[f"sat_path_t{i}"] = str(sat_files[ts])

        n_valid = 0
        for step in range(1, ROLLOUT + 1):
            future_ts = anchor_ts + (step - 1) * DT
            if future_ts in sat_files and future_ts in snap_files:
                row[f"sat_target_path_t{step}"] = str(sat_files[future_ts])
                row[f"ims_target_path_t{step}"] = str(snap_files[future_ts])
                n_valid = step
            else:
                for rem in range(step, ROLLOUT + 1):
                    row[f"sat_target_path_t{rem}"] = None
                    row[f"ims_target_path_t{rem}"] = None
                break
        else:
            n_valid = ROLLOUT

        row["valid_steps"] = n_valid
        if n_valid < 1:
            continue
        rows_test.append(row)

    df = pd.DataFrame(rows_test)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df)} rows)")
    if len(df):
        print(f"date range: {df.timestamp.min()}  ->  {df.timestamp.max()}")


if __name__ == "__main__":
    main()
