@echo off
REM Phase 1 -- train the encoder + driver head with the manifestation head frozen
REM (dense ERA5 wind/temp supervision). Resolves the repo root relative to this script,
REM so it runs from any machine. Requires the venv created per the Maintenance Guide (B.1).
cd /d "%~dp0..\.."
if not exist logs mkdir logs
venv\Scripts\python.exe entry_point.py --device-id 0 --precision bf16 --batch-size 12 --grad-accum 3 --max-epochs 30 --num-workers 2 --rollout-max 2 --multihorizon-every 0 --train-csv data\processed\index_train_subset.csv --era5-path data\era5_npy --ckpt-dir checkpoints\h5_phase1 --freeze-stage mani --lambda-cloud 0.0 --lambda-rain 0.0 >> logs\h5_phase1.log 2>&1
echo EXIT_CODE %ERRORLEVEL% >> logs\h5_phase1.log
