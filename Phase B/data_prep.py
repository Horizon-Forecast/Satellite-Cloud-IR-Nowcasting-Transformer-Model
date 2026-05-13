#!/usr/bin/env python
# data_prep.py
# Horizon Forecast — Local Data Preparation Driver (runs on Windows PC).
# Authors: Or Mordechay Hod, Gilad Boudman | Braude College, CODE: 26-1-R-1
#
# Runs every data-prep step from src/data/prep.py against the local project tree.
# Writes artifacts under data/processed/. Idempotent: re-running skips finished work.
#
# Usage:
#   python data_prep.py                       # all steps
#   python data_prep.py --steps sat,ims       # only listed steps
#   python data_prep.py --skip-test           # build train+val only
#   python data_prep.py --steps verify        # just check artifacts
#
# Available step names: station, dem, sat, ims, snapshots, mask, index, norm,
# rain, verify, all

import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable when run from repo root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import prep

STEP_ORDER = [
    "station", "dem", "sat", "ims", "snapshots",
    "mask", "index", "norm", "rain", "verify",
]


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _resolve_steps(arg: str) -> list:
    if arg == "all":
        return STEP_ORDER[:-1]  # everything except final verify (added implicitly)
    requested = [s.strip() for s in arg.split(",") if s.strip()]
    bad = [s for s in requested if s not in STEP_ORDER]
    if bad:
        raise SystemExit(f"Unknown step(s): {bad}. Valid: {STEP_ORDER}")
    return requested


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Horizon Forecast local data prep — runs every step from src/data/prep.py."
    )
    parser.add_argument("--steps", default="all",
                        help=f"Comma-separated subset of {STEP_ORDER}, or 'all'")
    parser.add_argument("--skip-test", action="store_true",
                        help="Do not build index_test.csv (useful if 2025 raw incomplete)")
    parser.add_argument("--project-root", default=str(ROOT),
                        help="Project root containing data/ and src/")
    parser.add_argument("--ir-wv-sample", type=int, default=2000,
                        help="Random sample size for IR/WV norm stats")
    parser.add_argument("--strict-sat", action="store_true",
                        help="Fail verify if any TIF lacks a matching .npy "
                             "(default: warn only)")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    raw = project / "data" / "raw"
    processed = project / "data" / "processed"
    parquet_dir = processed / "ims_parquet"

    _setup_logging(processed / "prep.log")
    log = logging.getLogger("prep_local")
    log.info(f"project_root={project}")
    log.info(f"steps={args.steps}  skip_test={args.skip_test}")

    steps = _resolve_steps(args.steps)
    do = lambda name: name in steps  # noqa: E731

    try:
        if do("station"):
            prep.fix_station_csv(
                str(raw / "stations_locations.csv"),
                str(raw / "ims_stations.csv"),
            )
        if do("dem"):
            prep.build_dem(
                str(raw / "ElevationData(NASA)"),
                str(processed / "dem_256.npy"),
            )
        if do("sat"):
            prep.convert_sat_tifs(
                str(raw / "IR_108 with WV_062 Tif (EUMETVIEW)" / "Raw"),
                str(processed / "sat_npy"),
                local_cache=None,  # local PC: no Drive write detour needed
            )
        if do("ims"):
            prep.merge_ims_to_parquet(
                str(raw / "GroundTruth(IMS)"),
                str(parquet_dir),
            )
        if do("snapshots"):
            prep.build_ims_snapshots(
                str(parquet_dir),
                str(processed / "ims_snapshots"),
            )
        if do("mask"):
            prep.build_station_mask_step(
                str(raw / "ims_stations.csv"),
                str(processed / "station_mask.pt"),
            )
        if do("index"):
            prep.build_indices(
                str(processed / "sat_npy"),
                str(processed / "ims_snapshots"),
                str(processed),
                skip_test=args.skip_test,
            )
        if do("norm"):
            prep.compute_norm_stats(
                str(processed / "index_train.csv"),
                str(parquet_dir),
                str(processed / "dem_256.npy"),
                str(processed / "norm_stats.json"),
                ir_wv_sample=args.ir_wv_sample,
            )
        if do("rain"):
            prep.compute_and_cache_rain_weights(
                str(parquet_dir),
                str(processed / "rain_weights.pt"),
            )
        if do("verify") or args.steps == "all":
            prep.verify_artifacts(
                str(processed),
                require_test=not args.skip_test,
                sat_raw=str(raw / "IR_108 with WV_062 Tif (EUMETVIEW)" / "Raw"),
                strict_sat=args.strict_sat,
            )

        log.info("data prep DONE")
        return 0
    except Exception as e:
        log.exception(f"data prep FAILED at: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
