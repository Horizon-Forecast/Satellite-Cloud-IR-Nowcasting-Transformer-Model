#!/usr/bin/env python
"""
download_era5.py
Downloads ERA5 hourly wind + temperature for the Israel domain (2020-2026).

Variables downloaded:
  - u10: 10m U-component of wind  (m/s, eastward)
  - v10: 10m V-component of wind  (m/s, northward)
  - t2m: 2m air temperature        (K → convert to °C later)

Domain: Israel + surroundings
  North: 34.0°N, South: 29.0°N, West: 33.5°E, East: 37.0°E

Output: data/era5/era5_{YEAR}.nc  (one NetCDF per year, ~500MB each)

Setup:
  1. Register at https://cds.climate.copernicus.eu/user/register
  2. After login go to https://cds.climate.copernicus.eu/user → copy UID + API Key
  3. Create file ~/.cdsapirc  (Windows: %USERPROFILE%\\.cdsapirc)  with content:
       url: https://cds.climate.copernicus.eu/api
       key: YOUR_API_KEY
  4. Install cdsapi:
       venv\\Scripts\\pip install cdsapi netCDF4 xarray
  5. Run:
       venv\\Scripts\\python data\\download_era5.py

After download run:
       venv\\Scripts\\python data\\process_era5.py   (converts NetCDF → .npy per timestamp)
"""

import cdsapi
from pathlib import Path

# Config
YEARS       = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
OUT_DIR     = Path(__file__).parent / "era5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Israel bounding box [North, West, South, East] (CDS format)
# Downloaded with wider buffer (33.5->37.0 lon) to ensure full coverage.
# process_era5.py regrids to exact training domain (34.0->36.0 lon, 29.0->34.0 lat).
AREA        = [34.0, 33.5, 29.0, 37.0]

VARIABLES   = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
]

DAYS        = [f"{d:02d}" for d in range(1, 32)]
HOURS       = [f"{h:02d}:00" for h in range(24)]   # hourly (ERA5 is 1h resolution)

# Download — one file per month to stay within CDS size limits
client = cdsapi.Client()

for year in YEARS:
    for month in range(1, 13 if year < 2026 else 2):  # 2026: Jan only
        month_str = f"{month:02d}"
        out_path  = OUT_DIR / f"era5_{year}_{month_str}.nc"

        if out_path.exists():
            print(f"[SKIP] {out_path.name} already exists")
            continue

        print(f"[DOWN] Requesting ERA5 {year}-{month_str} ...")
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type":   "reanalysis",
                "variable":       VARIABLES,
                "year":           str(year),
                "month":          month_str,
                "day":            DAYS,
                "time":           HOURS,
                "area":           AREA,
                "format":         "netcdf",
            },
            str(out_path),
        )
        print(f"[DONE] Saved -> {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")

print("\nAll years downloaded. Run process_era5.py next.")
