"""
Download EUMETSAT SEVIRI IR/WV satellite tiles for the Israel domain.

Uses the EUMETSAT Data Store + Data Tailor (via the `eumdac` client) to search the
MSG High-Rate SEVIRI collection, crop each frame to the region of interest, and save
IR 10.8 um and WV 6.2 um GeoTIFFs into data/raw/. These raw tiles are the model input;
they are converted to the training .npy frames by prep.py (stage 3, convert_sat_tifs).

Credentials (never hard-code them):
  1. Create a free EUMETSAT account and generate a consumer key + secret at
     https://api.eumetsat.int/api-key/
  2. In the Data Store Data Tailor, create a customisation preset that crops to the ROI
     and outputs GeoTIFF, named to match PRESET_NAME below (default
     "Horizon Forecast Dataset Satiate").
  3. Export the key/secret before running:
       set EUMETSAT_KEY=your_consumer_key
       set EUMETSAT_SECRET=your_consumer_secret
     (optionally a second account for more parallel jobs:
       set EUMETSAT_KEY2=... / set EUMETSAT_SECRET2=...)

Output dir (override with HORIZON_SEVIRI_DIR): data/raw/IR_108 with WV_062 Tif (EUMETVIEW)/
Elevation overlays are optional and only generated if HORIZON_HGT_DIR points at SRTM .hgt tiles.
"""
import eumdac
import datetime
import os
import sys
import time
import shutil
import logging
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from pyproj import CRS
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Event, Lock
import matplotlib.pyplot as plt

# CREDENTIALS (from environment — see module docstring)
def _load_credentials():
    creds = []
    k1, s1 = os.environ.get("EUMETSAT_KEY"), os.environ.get("EUMETSAT_SECRET")
    if k1 and s1:
        creds.append((k1, s1))
    k2, s2 = os.environ.get("EUMETSAT_KEY2"), os.environ.get("EUMETSAT_SECRET2")
    if k2 and s2:
        creds.append((k2, s2))
    if not creds:
        sys.exit(
            "No EUMETSAT credentials found. Set EUMETSAT_KEY and EUMETSAT_SECRET "
            "(get them at https://api.eumetsat.int/api-key/). See the module docstring."
        )
    return creds

API_CREDENTIALS = _load_credentials()

# Region Of Interest (N, S, W, E)
ROI_NORTH = 34.0
ROI_SOUTH = 29.0
ROI_WEST = 34.0
ROI_EAST = 36.0
ROI_BBOX = (ROI_WEST, ROI_SOUTH, ROI_EAST, ROI_NORTH)

# Desired MSG channels
CHANNELS = ["IR_108", "WV_062"]
PRESET_NAME = os.environ.get("EUMETSAT_PRESET", "Horizon Forecast Dataset Satiate")

# FOLDER SETTINGS (repo-relative defaults, override via env)
_DATA = Path(__file__).resolve().parent
BASE_DIR = os.environ.get(
    "HORIZON_SEVIRI_DIR",
    str(_DATA / "raw" / "IR_108 with WV_062 Tif (EUMETVIEW)"),
)
RAW_DIR = os.path.join(BASE_DIR, "Raw")
OVERLAY_DIR = os.path.join(BASE_DIR, "ElevationOverlay")

# Optional: folder of SRTM .hgt tiles for the (purely cosmetic) elevation overlay.
HGT_DIR = os.environ.get("HORIZON_HGT_DIR", "")

# Date Range
START_DATE = datetime.datetime(2020, 1, 1)
END_DATE = datetime.datetime(2026, 1, 1)

# SYSTEM SETTINGS
JOBS_PER_KEY = 3
SEARCH_WINDOW_DAYS = 7
MAX_SUBMIT_RETRIES = 20
RETRY_BASE_SLEEP = 20
RETRY_MAX_SLEEP = 300
STOP_EVENT = Event()
_print_lock = Lock()

logging.getLogger("eumdac").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs, flush=True)


def _try_get_chains(tailor):
    """Robustly retrieve the list of Data Tailor chains/presets across eumdac versions."""
    if hasattr(tailor, "chains"):
        chains_obj = tailor.chains
        if hasattr(chains_obj, "search") and callable(chains_obj.search):
            try:
                return list(chains_obj.search())
            except Exception:
                pass
        try:
            return list(chains_obj)
        except Exception:
            pass
    if hasattr(tailor, "list_chains"):
        try:
            return list(tailor.list_chains())
        except Exception:
            pass
    return []


def process_elevation_overlay(satellite_tif_path, hgt_folder, output_overlay_path):
    """Optional cosmetic overlay: reproject SRTM contours onto a satellite frame and save a PNG."""
    try:
        hgt_files = [os.path.join(hgt_folder, f) for f in os.listdir(hgt_folder) if f.endswith(".hgt")]
        if not hgt_files:
            return
        src_files_to_mosaic = [rasterio.open(fp) for fp in hgt_files]
        mosaic, out_trans = merge(src_files_to_mosaic)
        mosaic_meta = src_files_to_mosaic[0].meta.copy()
        mosaic_meta.update({"driver": "GTiff", "height": mosaic.shape[1],
                            "width": mosaic.shape[2], "transform": out_trans, "crs": "EPSG:4326"})
        with rasterio.open(satellite_tif_path) as sat_src:
            sat_data = sat_src.read(1)
            sat_transform = sat_src.transform
            sat_crs = sat_src.crs or CRS.from_string(
                "+proj=geos +h=35785831 +lon_0=0 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +sweep=y +no_defs")
            elevation_projected = np.zeros((sat_src.height, sat_src.width), dtype=np.float32)
            reproject(source=mosaic, destination=elevation_projected,
                      src_transform=out_trans, src_crs=mosaic_meta["crs"],
                      dst_transform=sat_transform, dst_crs=sat_crs, resampling=Resampling.bilinear)
            fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
            valid_sat = sat_data[sat_data > 0]
            vmin, vmax = (np.percentile(valid_sat, 5), np.percentile(valid_sat, 95)) if valid_sat.size else (0, 255)
            extent = [sat_src.bounds.left, sat_src.bounds.right, sat_src.bounds.bottom, sat_src.bounds.top]
            ax.imshow(sat_data, cmap="gray_r", vmin=vmin, vmax=vmax, origin="upper", extent=extent)
            elevation_projected[elevation_projected < 10] = np.nan
            ax.contour(elevation_projected, levels=[600, 900, 1200, 2000], colors="cyan",
                       linewidths=0.6, alpha=0.8, origin="upper", extent=extent)
            ax.axis("off")
            plt.savefig(output_overlay_path, bbox_inches="tight", pad_inches=0, dpi=150)
            plt.close()
        for src in src_files_to_mosaic:
            src.close()
    except Exception as e:
        safe_print(f"Overlay failed for {satellite_tif_path}: {e}")


def search_with_roi(collection, start_dt, end_dt):
    try:
        return collection.search(dtstart=start_dt, dtend=end_dt, bbox=ROI_BBOX)
    except Exception:
        return collection.search(dtstart=start_dt, dtend=end_dt)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    print("--- SEVIRI DOWNLOADER ---")
    print(f"Raw Data Dir: {RAW_DIR}")

    accounts = []
    print(f"\n--- AUTHENTICATING {len(API_CREDENTIALS)} ACCOUNT(S) ---")
    for i, (k, s) in enumerate(API_CREDENTIALS):
        try:
            token = eumdac.AccessToken((k, s))
            tailor = eumdac.DataTailor(token)
            user_chain = next((c for c in _try_get_chains(tailor) if c.name == PRESET_NAME), None)
            if user_chain:
                print(f"   Account {i+1}: found preset '{user_chain.name}'")
                accounts.append({"tailor": tailor, "gate": Semaphore(JOBS_PER_KEY), "chain": user_chain})
            else:
                print(f"   Account {i+1}: preset '{PRESET_NAME}' NOT found — skipping.")
        except Exception as e:
            print(f"   Account {i+1} failed: {e}")

    if not accounts:
        print("No valid accounts / presets. Exiting.")
        return

    datastore = eumdac.DataStore(accounts[0]["tailor"].token)
    collection = datastore.get_collection("EO:EUM:DAT:MSG:HRSEVIRI")
    account_counter = [0]
    account_lock = Lock()

    def process_file(args):
        product, date_str, raw_month_dir, overlay_month_dir = args
        if STOP_EVENT.is_set():
            return
        with account_lock:
            acc = accounts[account_counter[0] % len(accounts)]
            account_counter[0] += 1
        with acc["gate"]:
            try:
                submit_attempt = 0
                while True:
                    try:
                        cust = acc["tailor"].new_customisation(product, acc["chain"])
                        break
                    except Exception as e:
                        msg = str(e).lower()
                        if "exceeding your maximum number" in msg or "queued+running" in msg:
                            submit_attempt += 1
                            if submit_attempt > MAX_SUBMIT_RETRIES:
                                return [f"Error: {e}"]
                            time.sleep(min(RETRY_BASE_SLEEP * submit_attempt, RETRY_MAX_SLEEP))
                            continue
                        raise
                while cust.status in ["QUEUED", "RUNNING"]:
                    if STOP_EVENT.is_set():
                        return
                    time.sleep(2)
                    cust = acc["tailor"].get_customisation(cust._id)
                if cust.status != "DONE":
                    return [f"Failed: {cust.status} [{date_str}]"]
                messages = []
                for out in cust.outputs:
                    fname = getattr(out, "name", f"output_{date_str}.tif")
                    ch_tag = next((c for c in CHANNELS
                                   if c.lower().replace("_", "") in fname.lower().replace("_", "")), "UNK")
                    raw_path = os.path.join(raw_month_dir, f"{date_str}_{ch_tag}.tif")
                    with cust.stream_output(out) as s, open(raw_path, "wb") as f:
                        shutil.copyfileobj(s, f)
                    messages.append(f"Saved: {os.path.basename(raw_path)}")
                    if "IR" in ch_tag and HGT_DIR and os.path.isdir(HGT_DIR):
                        overlay_path = os.path.join(overlay_month_dir, f"{date_str}_{ch_tag}_elevation.png")
                        if not os.path.exists(overlay_path):
                            process_elevation_overlay(raw_path, HGT_DIR, overlay_path)
                try:
                    cust.delete()
                except Exception:
                    pass
                return messages
            except Exception as e:
                return [f"Error: {e} [{date_str}]"]

    curr = START_DATE
    while curr < END_DATE:
        nxt = min(curr + datetime.timedelta(days=SEARCH_WINDOW_DAYS), END_DATE)
        print(f"\nWindow: {curr.date()} -> {nxt.date()}")
        products = search_with_roi(collection, curr, nxt)
        tasks = []
        for p in products:
            d_str = p.sensing_start.strftime("%Y%m%d_%H%M")
            month_str = p.sensing_start.strftime("%Y%m")
            raw_month_dir = os.path.join(RAW_DIR, month_str)
            overlay_month_dir = os.path.join(OVERLAY_DIR, month_str)
            os.makedirs(raw_month_dir, exist_ok=True)
            os.makedirs(overlay_month_dir, exist_ok=True)
            if [f for f in os.listdir(raw_month_dir) if f.startswith(f"{d_str}_") and f.endswith(".tif")]:
                continue
            tasks.append((p, d_str, raw_month_dir, overlay_month_dir))
        if tasks:
            print(f"Processing {len(tasks)} files...")
            with ThreadPoolExecutor(max_workers=5) as exc:
                for f in as_completed({exc.submit(process_file, t): t for t in tasks}):
                    safe_print(f.result())
        curr = nxt


if __name__ == "__main__":
    main()
