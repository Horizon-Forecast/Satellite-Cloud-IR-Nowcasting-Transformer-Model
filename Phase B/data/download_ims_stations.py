"""
Fetch IMS station coordinates -> data/raw/stations_locations.csv.

Downloads the IMS station list and extracts (StationID, Name, Latitude, Longitude, Active)
for every station with valid coordinates. prep.py (stage 1) uses this file to map each
station's GPS location to a pixel on the 256x256 grid (the station mask).

Credentials (never hard-code the token):
  Request a free API token from the IMS data portal
  (https://ims.gov.il/en/ObservationDataAPI), then export it before running:
      set IMS_API_TOKEN=your_token

Output file (override with HORIZON_IMS_STATIONS_CSV): data/raw/stations_locations.csv
"""
import os
import sys
from pathlib import Path

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

OUTPUT_FILE = os.environ.get(
    "HORIZON_IMS_STATIONS_CSV",
    str(Path(__file__).resolve().parent / "raw" / "stations_locations.csv"),
)

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])))


def main():
    print("--- FETCHING IMS STATION LOCATIONS ---")
    try:
        response = session.get(f"{BASE_URL}/stations", headers=HEADERS)
        response.raise_for_status()
        stations_json = response.json()
        print(f"Found {len(stations_json)} stations. Extracting coordinates...")

        rows = []
        for station in stations_json:
            loc = station.get("location", {})
            lat, lon = loc.get("latitude"), loc.get("longitude")
            if lat is not None and lon is not None:
                rows.append({
                    "StationID": station["stationId"],
                    "Name": station["name"],
                    "Latitude": lat,
                    "Longitude": lon,
                    "Active": station.get("active", False),
                })

        if rows:
            os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
            df = pd.DataFrame(rows).sort_values(by="StationID")
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"Saved {len(df)} locations to '{OUTPUT_FILE}'.")
        else:
            print("No location data found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
