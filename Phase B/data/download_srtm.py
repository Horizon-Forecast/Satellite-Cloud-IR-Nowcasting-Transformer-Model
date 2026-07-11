"""
Download the SRTM1 elevation tiles covering the Israel domain.

Uses the AWS public mirror (s3.amazonaws.com/elevation-tiles-prod/skadi/), which serves
NASA SRTM1 1-arcsec tiles as .hgt.gz without authentication (no NASA Earthdata token).
The tiles land in data/raw/ElevationData(NASA)/ and are mosaicked into the 256x256 DEM by
prep.py (stage 2, build_dem).

Usage: python data/download_srtm.py
"""
import gzip
import sys
from io import BytesIO
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
DEM_DIR = ROOT / "raw" / "ElevationData(NASA)"
DEM_DIR.mkdir(parents=True, exist_ok=True)

# Required tiles for GRID_BOUNDS (lat 29-34, lon 34-36).
REQUIRED = [
    "N29E034", "N29E035",
    "N30E034", "N30E035",
    "N31E034", "N31E035",
    "N32E034", "N32E035",
    "N33E034", "N33E035",
]

missing = [t for t in REQUIRED if not (DEM_DIR / f"{t}.hgt").exists()]
print(f"required: {len(REQUIRED)} tiles")
print(f"present : {len(REQUIRED) - len(missing)}")
print(f"missing : {len(missing)} -> {missing}")
print()

if not missing:
    print("All tiles present. Nothing to download.")
    sys.exit(0)

# AWS Skadi-format mirror: gzipped .hgt under /skadi/<lat_dir>/<tile>.hgt.gz
BASE = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"

for tile in missing:
    lat_dir = tile[:3]  # "N31"
    url = f"{BASE}/{lat_dir}/{tile}.hgt.gz"
    print(f"downloading {tile} from {url}")
    try:
        r = requests.get(url, timeout=120)
    except Exception as e:
        print(f"  FAILED: {e}")
        continue
    if r.status_code != 200:
        print(f"  FAILED status={r.status_code}")
        continue
    print(f"  got {len(r.content)/1e6:.1f} MB .gz")

    try:
        data = gzip.decompress(r.content)
    except Exception as e:
        print(f"  ERROR decompress: {e}")
        continue

    out_path = DEM_DIR / f"{tile}.hgt"
    out_path.write_bytes(data)
    print(f"  saved {out_path.name}  ({out_path.stat().st_size/1e6:.1f} MB)")

print()
print("done. next: rebuild the processed dataset with  python -m src.data.prep")
