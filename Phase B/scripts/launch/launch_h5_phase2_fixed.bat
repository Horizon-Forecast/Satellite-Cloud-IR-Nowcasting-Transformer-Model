@echo off
REM Phase 2 -- freeze the encoder + driver head, train the manifestation head (cloud + rain).
REM Resumes the Phase-1 best weights (epoch/LR reset) and selects by CSI (multihorizon every 5).
REM Resolves the repo root relative to this script, so it runs from any machine.
cd /d "%~dp0..\.."
if not exist logs mkdir logs
venv\Scripts\python.exe entry_point.py --device-id 0 --precision bf16 --batch-size 12 --val-batch-size 64 --grad-accum 3 --max-epochs 30 --num-workers 2 --rollout-max 2 --multihorizon-every 5 --train-csv data\processed\index_train_subset.csv --era5-path data\era5_npy --resume-ckpt checkpoints\h5_phase1\gpu0_best.pt --resume-weights-only --ckpt-dir checkpoints\h5_phase2 --freeze-stage encoder_phys --lambda-cloud 1.0 --lambda-rain 0.5 --lambda-thermo 0.0 >> logs\h5_phase2.log 2>&1
echo EXIT_CODE %ERRORLEVEL% >> logs\h5_phase2.log
