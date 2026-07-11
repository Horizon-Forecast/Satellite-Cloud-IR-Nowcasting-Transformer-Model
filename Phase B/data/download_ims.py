"""
Download IMS (Israel Meteorological Service) ground-station time series.

Pulls every active channel (wind, temperature, precipitation, ...) for every station on
the IMS network across the project date range and saves one wide CSV per station into
data/raw/GroundTruth(IMS)/. These are the sparse rain (and driver-monitoring) ground truth;
prep.py (stages 4-5) merges them into per-timestamp station snapshots.

Credentials (never hard-code the token):
  Request a free API token from the IMS data portal
  (https://ims.gov.il/en/ObservationDataAPI), then export it before running:
      set IMS_API_TOKEN=your_token

Output dir (override with HORIZON_IMS_DIR): data/raw/GroundTruth(IMS)/
Also run download_ims_stations.py once to fetch the station-coordinate CSV.
"""
import os
import sys
import time
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# CREDENTIALS (from environment — see module docstring)
API_TOKEN = os.environ.get("IMS_API_TOKEN")
if not API_TOKEN:
    sys.exit(
        "No IMS token found. Set IMS_API_TOKEN (request one at "
        "https://ims.gov.il/en/ObservationDataAPI). See the module docstring."
    )

BASE_URL = "https://api.ims.gov.il/v1/envista"
HEADERS = {
    "Authorization": f"ApiToken {API_TOKEN}",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) python-requests/2.31",
}

# Date range and parallelism
START_DATE = "2020/01/01"
END_DATE = "2026/01/01"
MAX_WORKERS = 5

# Output dir (repo-relative default, override via env)
OUTPUT_DIR = os.environ.get(
    "HORIZON_IMS_DIR",
    str(Path(__file__).resolve().parent / "raw" / "GroundTruth(IMS)"),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Robust session
session = requests.Session()
retry_strategy = Retry(total=5, backoff_factor=2,
                       status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)


def get_data_chunk(station_id, channel_id, from_date, to_date):
    """Download one station/channel/date-range chunk. Returns a list of readings (empty on any error)."""
    url = f"{BASE_URL}/stations/{station_id}/data/{channel_id}?from={from_date}&to={to_date}"
    try:
        response = session.get(url, headers=HEADERS, timeout=30)
        if response.status_code == 200:
            return response.json().get("data", []) if response.text.strip() else []
        return []
    except Exception:
        time.sleep(5)
        return []


def process_station_agent(station):
    """Download all active channels for one station and save a wide CSV (index=Date, columns=Channel)."""
    name = station["name"]
    st_id = station["stationId"]
    monitors = station.get("monitors", [])

    expected_file = os.path.join(OUTPUT_DIR, f"{name}_{st_id}.csv")
    if os.path.exists(expected_file):
        print(f"[skip] {name} (file exists)")
        return

    print(f"[start] {name}...")
    all_data = []
    curr = datetime.strptime(START_DATE, "%Y/%m/%d")
    end = datetime.strptime(END_DATE, "%Y/%m/%d")
    while curr < end:
        next_date = min(curr + timedelta(days=30), end)
        d_from, d_to = curr.strftime("%Y/%m/%d"), next_date.strftime("%Y/%m/%d")
        for mon in monitors:
            if mon["active"]:
                ch_id, ch_name = mon["channelId"], mon["name"]
                for r in get_data_chunk(st_id, ch_id, d_from, d_to):
                    val = next((c["value"] for c in r["channels"] if c["id"] == ch_id), None)
                    if val is not None:
                        all_data.append({"Date": r["datetime"], "Channel": ch_name, "Value": val})
        curr = next_date

    if all_data:
        try:
            df = pd.DataFrame(all_data)
            df.pivot_table(index="Date", columns="Channel", values="Value",
                           aggfunc="first").to_csv(expected_file)
            print(f"[done] {name}: saved {len(df)} readings.")
        except Exception as e:
            print(f"[error] saving {name}: {e}")
    else:
        print(f"[warn] no data for {name}.")


def main():
    print("--- IMS ALL-STATION DOWNLOADER ---")
    try:
        stations = session.get(f"{BASE_URL}/stations", headers=HEADERS).json()
        print(f"Found {len(stations)} stations on the network.")
    except Exception as e:
        print(f"CRITICAL: failed to get station list: {e}")
        return

    target = [s for s in stations if s.get("active", True)]
    print(f"Queueing {len(target)} active stations across {MAX_WORKERS} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_station_agent, s): s for s in target}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"[error] agent crashed: {exc}")
    print("--- DONE ---")


if __name__ == "__main__":
    main()
