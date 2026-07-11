#!/usr/bin/env python
"""
process_era5.py
Converts downloaded ERA5 NetCDF files -> per-timestamp .npy arrays
aligned with existing satellite data (15-min intervals, 256×256 Israel grid).

ERA5 is hourly -> we nearest-neighbor interpolate to 15-min timestamps.
Each output .npy has shape (2, 256, 256):
  channel 0: wind speed  sqrt(u10² + v10²)  [m/s]
  channel 1: temperature t2m                 [°C]

Output: data/era5_npy/YYYYMM/YYYYMMDD_HHMM.npy

Usage:
    venv\\Scripts\\python data\\process_era5.py
"""

import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT        = Path(__file__).parent.parent   # project root
ERA5_DIR    = ROOT / "data" / "era5"
OUT_DIR     = ROOT / "data" / "era5_npy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Israel domain — must match GRID_BOUNDS in src/data/dataset.py EXACTLY
# GRID_BOUNDS = dict(lat_min=29.0, lat_max=34.0, lon_min=34.0, lon_max=36.0)
LAT_MIN, LAT_MAX = 29.0, 34.0
LON_MIN, LON_MAX = 34.0, 36.0
GRID_SIZE        = 256

# Target lat/lon grid (same as satellite 256×256 projection)
target_lats = np.linspace(LAT_MAX, LAT_MIN, GRID_SIZE)   # north->south (34.0 -> 29.0)
target_lons = np.linspace(LON_MIN, LON_MAX, GRID_SIZE)   # west->east   (34.0 -> 36.0)


def regrid_to_256(da: xr.DataArray) -> np.ndarray:
    """Bilinear interpolation from ERA5 0.25° grid -> 256×256 Israel grid.
    Interpolates lat and lon independently to produce a true 2D (256, 256) grid.
    """
    return da.interp(
        latitude=target_lats,
        longitude=target_lons,
        method="linear",
    ).values.astype(np.float32)  # shape (256, 256)


def process_year(nc_path: Path):
    print(f"\n[LOAD] {nc_path.name}")
    ds = xr.open_dataset(nc_path)

    # Rename coords if needed (CDS sometimes uses 'lat'/'lon')
    rename = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)

    # New CDS API uses 'valid_time' instead of 'time'
    time_key = "valid_time" if "valid_time" in ds else "time"
    times = ds[time_key].values  # numpy datetime64 array (hourly)
    n_times = len(times)
    print(f"  Timestamps: {n_times}  ({times[0]} -> {times[-1]})")

    for i, t in enumerate(times):
        # Convert to Python datetime (UTC)
        dt = (t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        ts = datetime.fromtimestamp(dt, tz=timezone.utc)

        # ERA5 is hourly — generate 4 × 15-min sub-timestamps (HH:00, HH:15, HH:30, HH:45)
        for minute_offset in [0, 15, 30, 45]:
            ts_out = ts.replace(minute=minute_offset, second=0, microsecond=0)

            yyyymm  = ts_out.strftime("%Y%m")
            fname   = ts_out.strftime("%Y%m%d_%H%M") + ".npy"
            out_sub = OUT_DIR / yyyymm
            out_sub.mkdir(parents=True, exist_ok=True)
            out_path = out_sub / fname

            if out_path.exists():
                continue  # idempotent

            # u10, v10 -> wind speed magnitude
            u10 = ds["u10"].isel({time_key: i})
            v10 = ds["v10"].isel({time_key: i})
            t2m = ds["t2m"].isel({time_key: i})

            u_grid   = regrid_to_256(u10)
            v_grid   = regrid_to_256(v10)
            wind_spd = np.sqrt(u_grid**2 + v_grid**2)   # (256, 256) m/s
            temp_c   = regrid_to_256(t2m) - 273.15       # K -> °C (256, 256)

            arr = np.stack([wind_spd, temp_c], axis=0)   # (2, 256, 256)
            np.save(out_path, arr)

        if i % 500 == 0:
            print(f"  [{i}/{n_times}] {ts.strftime('%Y-%m-%d %H:%M')} done")

    ds.close()
    print(f"[DONE] {nc_path.name}")


if __name__ == "__main__":
    nc_files = sorted(ERA5_DIR.glob("era5_*_*.nc"))  # monthly files: era5_YYYY_MM.nc
    if not nc_files:
        print("No ERA5 .nc files found in data/era5/. Run download_era5.py first.")
        raise SystemExit(1)

    for nc_path in nc_files:
        process_year(nc_path)

    print("\nAll ERA5 data processed -> data/era5_npy/")
