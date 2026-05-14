# src/data/prep.py
# Horizon Forecast — Data Preparation Pipeline (Phase B)
# Authors: Or Mordechay Hod, Gilad Boudman | Braude College, CODE: 26-1-R-1
#
# Pure-function module. Each step is idempotent and resumable: re-running
# a step skips work that is already on disk. Drivers (data_prep.ipynb on Colab
# and data_prep_local.py locally) import these functions and run them in order.
#
# Pipeline stages (call in this order):
#   1. fix_station_csv      raw/stations_locations.csv -> raw/ims_stations.csv
#   2. build_dem            raw/ElevationData(NASA)    -> processed/dem_256.npy
#   3. convert_sat_tifs     raw/IR_108 with WV_062 Tif -> processed/sat_npy/YYYYMM/*.npy
#   4. merge_ims_to_parquet raw/GroundTruth(IMS)/*.csv -> processed/ims_parquet/ims_train_YYYY.parquet
#   5. build_ims_snapshots  processed/ims_parquet      -> processed/ims_snapshots/YYYYMM/*.csv
#   6. build_station_mask_step                         -> processed/station_mask.pt
#   7. build_indices        processed/{sat_npy,ims_snapshots} -> processed/index_{train,val,test}.csv
#   8. compute_norm_stats   processed/* -> processed/norm_stats.json
#   9. compute_and_cache_rain_weights                  -> processed/rain_weights.pt
#  10. verify_artifacts     processed/*

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Reuse: do NOT re-implement these in prep.py
from src.data.dataset import (
    build_station_mask,
    compute_rain_class_weights,
    rain_mm_to_class,
    GRID_BOUNDS,
    GRID_H,
    GRID_W,
    N_RAIN_BINS,
    T_IN,
    T_ROLLOUT,
)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Station CSV column rename
# ══════════════════════════════════════════════════════════════════════════════
def fix_station_csv(raw_csv: str, out_csv: str) -> int:
    """
    Rename StationID/Latitude/Longitude → station_id/lat/lon and save.
    Idempotent: skips if out_csv already exists.
    Returns row count.
    """
    out = Path(out_csv)
    if out.exists():
        n = len(pd.read_csv(out))
        logger.info(f"[1] station csv exists ({n} rows) - skip")
        return n

    df = pd.read_csv(raw_csv)
    df = df.rename(columns={
        "StationID":  "station_id",
        "Latitude":   "lat",
        "Longitude":  "lon",
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["station_id", "lat", "lon"]].to_csv(out, index=False)
    logger.info(f"[1] wrote {len(df)} stations → {out}")
    return len(df)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — SRTM DEM mosaic → 256×256 .npy
# ══════════════════════════════════════════════════════════════════════════════
def _read_hgt(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=">i2").astype(np.float32)
    n = int(round(data.size ** 0.5))
    return data.reshape(n, n)


def _parse_hgt_name(stem: str) -> Tuple[int, int]:
    lat = int(stem[1:3]) * (1 if stem[0] == "N" else -1)
    lon = int(stem[4:7]) * (1 if stem[3] == "E" else -1)
    return lat, lon


def build_dem(dem_dir: str, out_path: str) -> Dict:
    """
    Mosaic SRTM3 .hgt tiles covering [29N-34N, 34E-36E], downsample to 256×256.
    Idempotent: skips if out_path exists.
    """
    from scipy.ndimage import zoom

    out = Path(out_path)
    if out.exists():
        arr = np.load(out)
        logger.info(f"[2] dem exists shape={arr.shape} — skip")
        return {"shape": arr.shape, "min": float(arr.min()), "max": float(arr.max())}

    dem_path = Path(dem_dir)
    LAT_MIN, LAT_MAX = int(GRID_BOUNDS["lat_min"]), int(GRID_BOUNDS["lat_max"])
    LON_MIN, LON_MAX = int(GRID_BOUNDS["lon_min"]), int(GRID_BOUNDS["lon_max"])
    STEP   = 1200
    N_LAT  = LAT_MAX - LAT_MIN
    N_LON  = LON_MAX - LON_MIN
    mosaic = np.zeros((N_LAT * STEP, N_LON * STEP), dtype=np.float32)

    loaded = 0
    for hgt_path in sorted(dem_path.glob("*.hgt")):
        try:
            lat, lon = _parse_hgt_name(hgt_path.stem)
        except Exception:
            continue
        if not (LAT_MIN <= lat < LAT_MAX and LON_MIN <= lon < LON_MAX):
            continue
        tile = _read_hgt(hgt_path)
        row_start = (LAT_MAX - 1 - lat) * STEP
        col_start = (lon - LON_MIN) * STEP
        mosaic[row_start:row_start + STEP, col_start:col_start + STEP] = tile[:STEP, :STEP]
        loaded += 1

    mosaic[mosaic < -1000] = 0.0  # void fill
    dem_256 = zoom(mosaic, (256 / mosaic.shape[0], 256 / mosaic.shape[1]), order=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), dem_256)
    logger.info(
        f"[2] dem mosaic: {loaded} tiles, "
        f"shape={dem_256.shape}, min={dem_256.min():.0f}m, max={dem_256.max():.0f}m"
    )
    return {"shape": dem_256.shape, "min": float(dem_256.min()), "max": float(dem_256.max())}


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Satellite TIF → .npy conversion
# ══════════════════════════════════════════════════════════════════════════════
def _tif_to_npy(tif_path: Path, h: int = 256, w: int = 256) -> np.ndarray:
    """
    Read a SEVIRI TIF and return a (C, H, W) float32 array.

    Void sentinels (raw values < -999 from EUMETVIEW edge / scan-gap pixels)
    are inpainted with the nearest valid pixel per channel via
    distance_transform_edt. Plain zero-fill was avoided because it produces a
    bimodal pixel distribution that inflates norm-stats std ~2-3x and gives
    the encoder a fake "void plateau" to memorize.
    """
    import rasterio
    from scipy.ndimage import zoom, distance_transform_edt

    with rasterio.open(tif_path) as src:
        arr = src.read().astype(np.float32)

    for ch in range(arr.shape[0]):
        plane = arr[ch]
        mask  = plane < -999
        if not mask.any() or mask.all():
            continue
        _, (ii, jj) = distance_transform_edt(mask, return_indices=True)
        arr[ch] = plane[ii, jj]

    if arr.shape[1] != h or arr.shape[2] != w:
        arr = zoom(arr, (1, h / arr.shape[1], w / arr.shape[2]), order=1)
    return arr


def convert_sat_tifs(
    sat_raw: str,
    out_root: str,
    local_cache: Optional[str] = None,
) -> Dict:
    """
    Convert each YYYYMMDD_HHMM_*.tif to (2,256,256) float32 .npy
    under out_root/YYYYMM/.

    Idempotent: skips files already present at out_root.

    On Colab (Drive write is slow + flaky), pass local_cache="data/tmp_cache/sat_npy"
    so files are written locally first then copied to Drive. PermissionError on Drive
    copy is non-fatal — left for next run to retry.
    """
    from tqdm.auto import tqdm

    sat_path = Path(sat_raw)
    out_path = Path(out_root)
    out_path.mkdir(parents=True, exist_ok=True)
    cache = Path(local_cache) if local_cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    months = sorted([d for d in sat_path.iterdir()
                     if d.is_dir() and re.fullmatch(r"\d{6}", d.name)])
    if not months:
        logger.warning(f"[3] no YYYYMM dirs found under {sat_path}")
        return {"ok": 0, "skip": 0, "err": 0, "missing_months": []}

    total_ok = total_skip = total_err = 0

    for month_dir in months:
        out_month   = out_path / month_dir.name
        cache_month = cache / month_dir.name if cache else None
        out_month.mkdir(exist_ok=True)
        if cache_month:
            cache_month.mkdir(exist_ok=True)

        tifs = sorted(month_dir.glob("*.tif"))
        m_ok = m_skip = m_err = 0

        for tif in tqdm(tifs, desc=month_dir.name, leave=False):
            stem = tif.stem
            final_out = out_month / f"{stem}.npy"
            if final_out.exists():
                m_skip += 1
                continue

            cache_out = cache_month / f"{stem}.npy" if cache_month else None
            if cache_out and cache_out.exists():
                try:
                    shutil.copy2(str(cache_out), str(final_out))
                    m_ok += 1
                except PermissionError:
                    m_err += 1
                continue

            try:
                arr = _tif_to_npy(tif)
                target = cache_out if cache_out else final_out
                np.save(str(target), arr)
                if cache_out:
                    try:
                        shutil.copy2(str(cache_out), str(final_out))
                    except PermissionError:
                        pass
                m_ok += 1
            except Exception as e:
                m_err += 1
                logger.error(f"[3] {tif.name}: {e}")

        total_ok   += m_ok
        total_skip += m_skip
        total_err  += m_err
        if m_ok or m_err:
            logger.info(f"[3] {month_dir.name}: {m_ok} new, {m_skip} skip, {m_err} err")

    # Per-file completeness diff (catches partial months too)
    raw_months = {m.name for m in months}
    out_months = {d.name for d in out_path.iterdir()
                  if d.is_dir() and re.fullmatch(r"\d{6}", d.name)}
    missing_months = sorted(raw_months - out_months)

    missing_files: List[str] = []
    per_month_missing: Dict[str, int] = {}
    for month_dir in months:
        tif_stems = {p.stem for p in month_dir.glob("*.tif")}
        out_month = out_path / month_dir.name
        npy_stems = ({p.stem for p in out_month.glob("*.npy")}
                     if out_month.exists() else set())
        miss = sorted(tif_stems - npy_stems)
        if miss:
            per_month_missing[month_dir.name] = len(miss)
            for stem in miss:
                missing_files.append(f"{month_dir.name}/{stem}.tif")

    if missing_files:
        logger.warning(f"[3] {len(missing_files)} TIF(s) missing matching .npy. "
                       f"Per-month: {per_month_missing}")
    else:
        logger.info("[3] sat completeness: every TIF has a matching .npy")

    logger.info(f"[3] sat: ok={total_ok} skip={total_skip} err={total_err} "
                f"months_raw={len(raw_months)} months_out={len(out_months)}")
    return {
        "ok": total_ok, "skip": total_skip, "err": total_err,
        "missing_months": missing_months,
        "missing_files":  missing_files,
        "per_month_missing": per_month_missing,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — IMS station CSVs → per-year parquet (OOM-safe streaming)
# ══════════════════════════════════════════════════════════════════════════════
_ID_RE = re.compile(r"_(\d+)\.csv$")


def _station_id_from_name(name: str) -> Optional[int]:
    m = _ID_RE.search(name)
    return int(m.group(1)) if m else None


def merge_ims_to_parquet(
    ims_dir: str,
    out_dir: str,
    start: str = "2020-01-01",
    end:   str = "2026-01-01",
) -> Dict:
    """
    Stream-convert IMS station CSVs into per-year parquet files.

    OOM-safe: opens one ParquetWriter per year, streams each station's rows
    in chunks. Never holds the full 28M-row table in memory.

    Filters out *_1m_*.csv files (rain-only schema, irrelevant for thermo+rain
    combined supervision). Date range default covers full project span 2020 → 2026.

    Idempotent: skips if all parquet files already exist.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm

    src = Path(ims_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC")
    years = list(range(start_ts.year, end_ts.year))  # 2020..2025 incl

    paths = {y: out / f"ims_train_{y}.parquet" for y in years}
    if all(p.exists() for p in paths.values()):
        rows = 0
        for p in paths.values():
            rows += pq.ParquetFile(p).metadata.num_rows
        logger.info(f"[4] all year parquets exist ({rows} rows) — skip")
        return {"rows": rows, "stations": 0, "skipped_files": 0, "years": years}

    # Filter out 1-minute rain-only files (only Date,Rain_1_min schema)
    csv_files = sorted(p for p in src.glob("*.csv") if "_1m_" not in p.name)
    logger.info(f"[4] processing {len(csv_files)} IMS CSV files (filtered out _1m_)")

    schema = pa.schema([
        ("timestamp",          pa.timestamp("ns", tz="UTC")),
        ("station_id",         pa.int32()),
        ("wind_speed_ms",      pa.float32()),
        ("temperature_c",      pa.float32()),
        ("precipitation_mmhr", pa.float32()),
    ])
    writers = {y: pq.ParquetWriter(str(paths[y]), schema, compression="snappy")
               for y in years}

    rows_total = 0
    skipped = 0
    stations_used = 0

    try:
        for csv_path in tqdm(csv_files, desc="ims"):
            sid = _station_id_from_name(csv_path.name)
            if sid is None:
                skipped += 1
                continue

            try:
                df = pd.read_csv(
                    csv_path,
                    usecols=["Date", "Rain", "WS", "TD"],
                    parse_dates=["Date"],
                    low_memory=False,
                )
            except (ValueError, KeyError) as e:
                logger.debug(f"[4] skip {csv_path.name}: {e}")
                skipped += 1
                continue
            except Exception as e:
                logger.warning(f"[4] err {csv_path.name}: {e}")
                skipped += 1
                continue

            df = df.replace(-9999.0, np.nan)
            # 10-minute reporting period after _1m_ filter; convert mm/10min → mm/hr
            df["precipitation_mmhr"] = df["Rain"].clip(lower=0) * 6.0
            df["wind_speed_ms"]      = df["WS"].clip(lower=0)
            df["temperature_c"]      = df["TD"]
            df["station_id"]         = np.int32(sid)
            df["timestamp"]          = pd.to_datetime(df["Date"], utc=True)

            df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)]
            # Drop rows missing wind OR temp (required for thermo supervision).
            # Precipitation NaN is allowed — treated as dry (class 0) downstream.
            df = df.dropna(subset=["wind_speed_ms", "temperature_c"], how="any")
            df = df.dropna(subset=["precipitation_mmhr"], how="all")
            if df.empty:
                continue

            df = df[["timestamp", "station_id",
                     "wind_speed_ms", "temperature_c", "precipitation_mmhr"]]

            # Cast to writer schema
            df["wind_speed_ms"]      = df["wind_speed_ms"].astype("float32")
            df["temperature_c"]      = df["temperature_c"].astype("float32")
            df["precipitation_mmhr"] = df["precipitation_mmhr"].astype("float32")

            for year, group in df.groupby(df["timestamp"].dt.year):
                year = int(year)
                if year not in writers:
                    continue
                table = pa.Table.from_pandas(
                    group, schema=schema, preserve_index=False
                )
                writers[year].write_table(table)
                rows_total += len(group)
            stations_used += 1
    finally:
        for w in writers.values():
            w.close()

    logger.info(f"[4] wrote {rows_total} rows across {len(years)} year files; "
                f"stations={stations_used} skipped_files={skipped}")
    return {"rows": rows_total, "stations": stations_used,
            "skipped_files": skipped, "years": years}


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — IMS snapshots (one CSV per 15-min target timestamp)
# ══════════════════════════════════════════════════════════════════════════════
def build_ims_snapshots(parquet_dir: str, out_dir: str) -> int:
    """
    Read each per-year parquet, round timestamps to 15min, group by timestamp+station,
    average duplicates, write one CSV per timestamp under out_dir/YYYYMM/{ts}.csv.

    Idempotent per-month: skips a YYYYMM dir if it already exists and is non-empty.
    """
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm

    pq_root = Path(parquet_dir)
    out     = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    parquets = sorted(pq_root.glob("ims_train_*.parquet"))
    if not parquets:
        logger.warning(f"[5] no parquets in {pq_root}")
        return 0

    total_ts = 0
    for pq_path in parquets:
        df = pq.read_table(pq_path).to_pandas()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["ts_15min"]  = df["timestamp"].dt.round("15min")
        df["ym"]        = df["ts_15min"].dt.strftime("%Y%m")

        for ym, ym_df in df.groupby("ym"):
            ym_dir = out / ym
            if ym_dir.exists() and any(ym_dir.iterdir()):
                continue
            ym_dir.mkdir(parents=True, exist_ok=True)

            for ts, grp in tqdm(ym_df.groupby("ts_15min"),
                                desc=f"snap {ym}", leave=False):
                snap = (
                    grp.groupby("station_id")
                       .agg(wind_speed_ms     =("wind_speed_ms",      "mean"),
                            temperature_c    =("temperature_c",      "mean"),
                            precipitation_mmhr=("precipitation_mmhr", "mean"))
                       .reset_index()
                )
                snap = snap.dropna(subset=["wind_speed_ms", "temperature_c"])
                if snap.empty:
                    continue
                ts_str = pd.Timestamp(ts).strftime("%Y%m%d_%H%M")
                snap.to_csv(str(ym_dir / f"{ts_str}.csv"), index=False)
                total_ts += 1
            logger.info(f"[5] snapshots {ym} done")

    logger.info(f"[5] snapshots written: {total_ts} new timestamps")
    return total_ts


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Station mask (delegates to dataset.build_station_mask)
# ══════════════════════════════════════════════════════════════════════════════
def build_station_mask_step(stations_csv: str, out_path: str) -> int:
    """Wrap build_station_mask for use from prep drivers. Idempotent."""
    out = Path(out_path)
    if out.exists():
        data = torch.load(out, weights_only=True)
        n = int(data["mask"].sum().item())
        logger.info(f"[6] station_mask exists ({n} pixels) — skip")
        return n
    mask = build_station_mask(stations_csv, str(out))
    n = int(mask.sum().item())
    logger.info(f"[6] station_mask built: {n} active station pixels")
    return n


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Build train/val/test index CSVs
# ══════════════════════════════════════════════════════════════════════════════
def build_indices(
    sat_npy_root: str,
    snap_root:    str,
    out_dir:      str,
    history:      int  = T_IN,
    rollout:      int  = T_ROLLOUT,
    skip_test:    bool = False,
) -> Dict:
    """
    For every anchor timestamp T where history frames + at least 1 future step exist,
    emit a row with:
      - sat_path_t0..t{history-1}        : input history frame paths
      - sat_target_path_t1..t{rollout}   : future sat frame paths (None if unavailable)
      - ims_target_path_t1..t{rollout}   : future IMS snapshot paths (None if unavailable)
      - valid_steps                      : count of consecutive steps with both sat+IMS

    Idempotent: skips if output files already contain rollout columns.

    Splits (chronological — no data leakage):
      train: 2020-01-01 .. 2023-12-31
      val:   2024-01-01 .. 2024-06-30
      test:  2024-07-01 .. 2025-12-31  (omitted if skip_test=True)
    """
    from tqdm.auto import tqdm

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Idempotency: skip if rollout columns already present
    out_train = out / "index_train.csv"
    if out_train.exists():
        try:
            sample = pd.read_csv(out_train, nrows=1)
            if f"sat_target_path_t{rollout}" in sample.columns:
                counts = {}
                for split in ("train", "val", "test"):
                    p = out / f"index_{split}.csv"
                    counts[split] = len(pd.read_csv(p)) if p.exists() else 0
                logger.info(f"[7] rollout indices exist (t1..t{rollout}) — skip {counts}")
                return counts
        except Exception:
            pass  # corrupt file — rebuild

    sat_root  = Path(sat_npy_root)
    snap_path = Path(snap_root)

    # Index all sat .npy files by timestamp
    sat_files: Dict[pd.Timestamp, Path] = {}
    for npy in sorted(sat_root.glob("**/*.npy")):
        try:
            d, t = npy.stem.split("_")[:2]
            ts = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}", tz="UTC")
            sat_files[ts] = npy
        except Exception:
            continue
    logger.info(f"[7] sat timestamps indexed: {len(sat_files)}")

    snap_files: Dict[pd.Timestamp, Path] = {}
    for csv in sorted(snap_path.glob("**/*.csv")):
        try:
            d, t = csv.stem.split("_")[:2]
            ts = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}", tz="UTC")
            snap_files[ts] = csv
        except Exception:
            continue
    logger.info(f"[7] snapshot timestamps indexed: {len(snap_files)}")

    DT = pd.Timedelta("15min")
    rows_train: List[Dict] = []
    rows_val:   List[Dict] = []
    rows_test:  List[Dict] = []

    for anchor_ts in tqdm(sorted(sat_files), desc="indices"):
        # Timestamp convention:
        #   history frames = [anchor_ts - 4·DT, ..., anchor_ts - 1·DT]   (4 frames, T-60 to T-15)
        #   target step k  = anchor_ts + (k-1)·DT                         (k=1 is T+0, k=16 is T+225)
        # i.e. anchor_ts is the first target frame, immediately following the 4 inputs.
        history_ts = [anchor_ts - (history - i) * DT for i in range(history)]
        if not all(ts in sat_files for ts in history_ts):
            continue

        row = {"timestamp": anchor_ts.isoformat()}
        for i, ts in enumerate(history_ts):
            row[f"sat_path_t{i}"] = str(sat_files[ts])

        # Find consecutive future steps: step 1 = anchor_ts, step 2 = anchor_ts+DT, ...
        n_valid = 0
        for step in range(1, rollout + 1):
            future_ts = anchor_ts + (step - 1) * DT  # step 1 = anchor_ts itself
            sat_ok = future_ts in sat_files
            ims_ok = future_ts in snap_files
            if sat_ok and ims_ok:
                row[f"sat_target_path_t{step}"] = str(sat_files[future_ts])
                row[f"ims_target_path_t{step}"] = str(snap_files[future_ts])
                n_valid = step
            else:
                for remaining in range(step, rollout + 1):
                    row[f"sat_target_path_t{remaining}"] = None
                    row[f"ims_target_path_t{remaining}"] = None
                break
        else:
            n_valid = rollout

        row["valid_steps"] = n_valid
        if n_valid < 1:
            continue  # no usable target at all

        yr, mo = anchor_ts.year, anchor_ts.month
        if yr <= 2023:
            rows_train.append(row)
        elif yr == 2024 and mo <= 6:
            rows_val.append(row)
        elif (yr == 2024 and mo >= 7) or yr == 2025:
            if not skip_test:
                rows_test.append(row)

    pd.DataFrame(rows_train).to_csv(out / "index_train.csv", index=False)
    pd.DataFrame(rows_val  ).to_csv(out / "index_val.csv",   index=False)
    if not skip_test:
        pd.DataFrame(rows_test).to_csv(out / "index_test.csv", index=False)

    counts = {"train": len(rows_train), "val": len(rows_val), "test": len(rows_test)}
    logger.info(
        f"[7] rollout indices (t1..t{rollout}): "
        f"train={counts['train']} val={counts['val']} test={counts['test']}"
    )
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Normalization stats (for entry_point.py NORM_STATS)
# ══════════════════════════════════════════════════════════════════════════════
def compute_norm_stats(
    index_train: str,
    parquet_dir: str,
    dem_path:    str,
    out_path:    str,
    ir_wv_sample: int = 2000,
    seed: int = 0,
) -> Dict:
    """
    Compute (mean, std) per channel using only training-period data:
      ir, wv: from a random sample of train target sat .npy files
      dem:    from data/processed/dem_256.npy
      wind, temp: from per-year IMS parquets (years <= 2023)

    Writes a JSON of {channel: [mean, std]} to out_path.
    Idempotent: skips if out_path exists.
    """
    out = Path(out_path)
    if out.exists():
        with out.open() as f:
            stats = json.load(f)
        logger.info(f"[8] norm_stats exists — skip ({list(stats.keys())})")
        return stats

    from tqdm.auto import tqdm
    rng = np.random.default_rng(seed)

    df = pd.read_csv(index_train)
    n_sample = min(ir_wv_sample, len(df))
    sat_col = "sat_target_path_t1" if "sat_target_path_t1" in df.columns else "sat_target_path"
    sample_paths = df.sample(n_sample, random_state=seed)[sat_col].dropna().tolist()

    ir_sum = ir_sq = 0.0; ir_n = 0
    wv_sum = wv_sq = 0.0; wv_n = 0
    for p in tqdm(sample_paths, desc="norm sat"):
        arr = np.load(p).astype(np.float64)  # (2, 256, 256)
        ir, wv = arr[0], arr[1]
        ir_sum += ir.sum(); ir_sq += (ir ** 2).sum(); ir_n += ir.size
        wv_sum += wv.sum(); wv_sq += (wv ** 2).sum(); wv_n += wv.size

    ir_mean = ir_sum / ir_n
    ir_std  = float(np.sqrt(max(ir_sq / ir_n - ir_mean ** 2, 1e-12)))
    wv_mean = wv_sum / wv_n
    wv_std  = float(np.sqrt(max(wv_sq / wv_n - wv_mean ** 2, 1e-12)))

    dem = np.load(dem_path).astype(np.float64)
    dem_mean = float(dem.mean())
    dem_std  = float(dem.std() + 1e-8)

    import pyarrow.parquet as pq
    pq_root = Path(parquet_dir)
    train_pqs = [pq_root / f"ims_train_{y}.parquet" for y in (2020, 2021, 2022, 2023)]
    train_pqs = [p for p in train_pqs if p.exists()]
    wind_vals: List[np.ndarray] = []
    temp_vals: List[np.ndarray] = []
    for p in train_pqs:
        tab = pq.read_table(p, columns=["wind_speed_ms", "temperature_c"]).to_pandas()
        w = tab["wind_speed_ms"].dropna()
        wind_vals.append(w[(w >= 0) & (w <= 60)].to_numpy(dtype=np.float64))
        t = tab["temperature_c"].dropna()
        temp_vals.append(t[(t >= -20) & (t <= 55)].to_numpy(dtype=np.float64))
    wind = np.concatenate(wind_vals) if wind_vals else np.array([4.5])
    temp = np.concatenate(temp_vals) if temp_vals else np.array([18.0])

    stats = {
        "ir":   [float(ir_mean),  float(ir_std)],
        "wv":   [float(wv_mean),  float(wv_std)],
        "dem":  [float(dem_mean), float(dem_std)],
        "wind": [float(wind.mean()), float(wind.std() + 1e-8)],
        "temp": [float(temp.mean()), float(temp.std() + 1e-8)],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"[8] norm_stats → {out}: {stats}")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Step 9 — Cache rain class weights
# ══════════════════════════════════════════════════════════════════════════════
def compute_and_cache_rain_weights(parquet_dir: str, out_path: str) -> torch.Tensor:
    """
    Compute 64-bin class weights from IMS training years (2020-2023) and cache.
    Idempotent: skips if out_path exists.
    """
    out = Path(out_path)
    if out.exists():
        w = torch.load(out, weights_only=True)
        logger.info(f"[9] rain_weights exists shape={tuple(w.shape)} — skip")
        return w

    import pyarrow.parquet as pq
    pq_root = Path(parquet_dir)
    train_pqs = [pq_root / f"ims_train_{y}.parquet" for y in (2020, 2021, 2022, 2023)]
    train_pqs = [p for p in train_pqs if p.exists()]
    if not train_pqs:
        raise FileNotFoundError(f"No training parquets in {pq_root}")

    frames = [pq.read_table(p, columns=["precipitation_mmhr"]).to_pandas()
              for p in train_pqs]
    df = pd.concat(frames, ignore_index=True)

    weights = compute_rain_class_weights(df, rain_col="precipitation_mmhr")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out)
    logger.info(f"[9] rain_weights cached: dry={weights[0]:.3f} "
                f"rain_mean={weights[1:].mean():.3f} → {out}")
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# Step 10 — Verification
# ══════════════════════════════════════════════════════════════════════════════
REQUIRED_ARTIFACTS = [
    "data/processed/dem_256.npy",
    "data/processed/station_mask.pt",
    "data/processed/index_train.csv",
    "data/processed/index_val.csv",
    "data/processed/norm_stats.json",
    "data/processed/rain_weights.pt",
]


def check_sat_completeness(
    sat_raw: str,
    sat_out: str,
) -> Dict:
    """
    Per-TIF completeness diff: list every raw .tif that lacks a matching .npy.
    Pure read-only check. Returns dict with totals + missing list.
    """
    raw = Path(sat_raw)
    out = Path(sat_out)
    months = sorted([d for d in raw.iterdir()
                     if d.is_dir() and re.fullmatch(r"\d{6}", d.name)])
    total_tif = total_npy = 0
    missing: List[str] = []
    per_month: Dict[str, Dict[str, int]] = {}
    for m in months:
        tif_stems = {p.stem for p in m.glob("*.tif")}
        out_m = out / m.name
        npy_stems = ({p.stem for p in out_m.glob("*.npy")}
                     if out_m.exists() else set())
        total_tif += len(tif_stems)
        total_npy += len(npy_stems)
        miss = sorted(tif_stems - npy_stems)
        if miss:
            per_month[m.name] = {"tif": len(tif_stems), "npy": len(npy_stems),
                                 "missing": len(miss)}
            for stem in miss:
                missing.append(f"{m.name}/{stem}.tif")
    return {
        "total_tif":   total_tif,
        "total_npy":   total_npy,
        "missing":     missing,
        "per_month":   per_month,
        "complete":    len(missing) == 0,
    }


def verify_artifacts(processed_root: str = "data/processed",
                     require_test: bool = False,
                     sat_raw: Optional[str] = None,
                     strict_sat: bool = False) -> None:
    """
    Raise FileNotFoundError listing every missing artifact, with hints.
    Use as a preflight check before training.

    If sat_raw is provided, also verify per-TIF completeness in sat_npy/.
    With strict_sat=True, any missing TIF→npy mapping causes an error;
    otherwise it logs a warning.
    """
    missing: List[str] = []
    required = list(REQUIRED_ARTIFACTS)
    if require_test:
        required.append("data/processed/index_test.csv")

    for rel in required:
        if not Path(rel).exists():
            missing.append(rel)

    sat_root = Path(processed_root) / "sat_npy"
    if not sat_root.exists() or not any(sat_root.iterdir()):
        missing.append(str(sat_root) + "/ (run convert_sat_tifs)")

    snap_root = Path(processed_root) / "ims_snapshots"
    if not snap_root.exists() or not any(snap_root.iterdir()):
        missing.append(str(snap_root) + "/ (run build_ims_snapshots)")

    if missing:
        msg = "Missing data prep artifacts:\n  - " + "\n  - ".join(missing)
        msg += "\n\nRun data_prep_local.py (local) or data_prep.ipynb (Colab) first."
        raise FileNotFoundError(msg)

    mask_path = Path("data/processed/station_mask.pt")
    if mask_path.exists():
        import torch as _torch
        mask_data = _torch.load(mask_path, weights_only=True)
        pixels = [(r, c) for r, c, *_ in mask_data["pixels"]]
        n_dup = len(pixels) - len(set(pixels))
        if n_dup > 0:
            logger.warning(
                f"station_mask.pt has {n_dup} duplicate pixel mappings "
                "(multiple stations in same grid cell). Rebuilding now..."
            )
            from src.data.dataset import build_station_mask
            stations_csv = str(Path("data/raw") / "ims_stations.csv")
            if Path(stations_csv).exists():
                build_station_mask(stations_csv, str(mask_path))
            else:
                logger.warning(
                    "data/raw/ims_stations.csv not found — cannot auto-rebuild. "
                    "Run build_station_mask('data/raw/ims_stations.csv', 'data/processed/station_mask.pt') manually."
                )
        if not all(0 <= r < 256 and 0 <= c < 256 for r, c in pixels):
            raise ValueError(
                "station_mask.pt has pixels outside 256×256 grid — check GRID_BOUNDS"
            )

    if sat_raw is not None and Path(sat_raw).exists():
        report = check_sat_completeness(sat_raw, str(sat_root))
        if report["complete"]:
            logger.info(f"[verify] sat: {report['total_npy']}/{report['total_tif']} "
                        f"TIFs have matching .npy")
        else:
            preview = report["missing"][:10]
            msg = (f"[verify] sat INCOMPLETE: {len(report['missing'])} TIFs missing .npy. "
                   f"Per-month: {report['per_month']}\n"
                   f"  first 10: {preview}")
            if strict_sat:
                raise FileNotFoundError(msg)
            logger.warning(msg)

    logger.info("[verify] all required artifacts present")


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: run all steps
# ══════════════════════════════════════════════════════════════════════════════
def run_all(
    project_root: str = ".",
    skip_test: bool = False,
    sat_local_cache: Optional[str] = None,
) -> Dict:
    """
    Execute every step in order with default paths under project_root.
    Each step is idempotent — safe to re-run after partial completion.
    """
    root = Path(project_root)

    raw         = root / "data" / "raw"
    processed   = root / "data" / "processed"
    parquet_dir = processed / "ims_parquet"

    summary: Dict[str, object] = {}

    summary["fix_station_csv"] = fix_station_csv(
        str(raw / "stations_locations.csv"),
        str(raw / "ims_stations.csv"),
    )
    summary["build_dem"] = build_dem(
        str(raw / "ElevationData(NASA)"),
        str(processed / "dem_256.npy"),
    )
    summary["convert_sat_tifs"] = convert_sat_tifs(
        str(raw / "IR_108 with WV_062 Tif (EUMETVIEW)" / "Raw"),
        str(processed / "sat_npy"),
        local_cache=sat_local_cache,
    )
    summary["merge_ims_to_parquet"] = merge_ims_to_parquet(
        str(raw / "GroundTruth(IMS)"),
        str(parquet_dir),
    )
    summary["build_ims_snapshots"] = build_ims_snapshots(
        str(parquet_dir),
        str(processed / "ims_snapshots"),
    )
    summary["build_station_mask"] = build_station_mask_step(
        str(raw / "ims_stations.csv"),
        str(processed / "station_mask.pt"),
    )
    summary["build_indices"] = build_indices(
        str(processed / "sat_npy"),
        str(processed / "ims_snapshots"),
        str(processed),
        skip_test=skip_test,
    )
    summary["compute_norm_stats"] = compute_norm_stats(
        str(processed / "index_train.csv"),
        str(parquet_dir),
        str(processed / "dem_256.npy"),
        str(processed / "norm_stats.json"),
    )
    summary["rain_weights"] = compute_and_cache_rain_weights(
        str(parquet_dir),
        str(processed / "rain_weights.pt"),
    )
    verify_artifacts(str(processed), require_test=not skip_test)
    return summary
