#!/bin/bash
# runpod_run_all.sh
# Sequential full training: H4 → H4b → H5-Phase1 → H5-Phase2
# Optimized for H100 80GB. Full dataset. ERA5 dense supervision only.
#
# Usage (from /workspace/project):
#   bash runpod_run_all.sh 2>&1 | tee logs/runpod_run_all.log
#
# Each run appends to its own log. Script halts on any failure (set -e).
# H5-Phase2 auto-loads Phase1 best checkpoint — no manual intervention needed.

set -e

PROJECT=/workspace/project
PYTHON=$PROJECT/venv/bin/python
LOGS=$PROJECT/logs
mkdir -p $LOGS $PROJECT/checkpoints

# ── H100 80GB parameters ──────────────────────────────────────────────────────
# batch_size=48, grad_accum=2 → effective batch=96
# val_batch_size=256 (no grads, 80GB free during val)
# num_workers=8 (H100 pods have 16-32 vCPUs)
# rollout_max=2 (consistent with local runs)
# max_epochs=30 (matching local runs for fair comparison)
# multihorizon_every=10 (CSI eval at ep10/20/30)
COMMON="--device-id 0 --precision bf16
        --batch-size 48 --grad-accum 2 --val-batch-size 256
        --num-workers 8 --rollout-max 2
        --max-epochs 30 --multihorizon-every 10
        --train-csv data/processed/index_train.csv
        --era5-path data/era5_npy
        --wandb --wandb-project horizon-forecast"

cd $PROJECT

echo "================================================================"
echo " H4  — cascade + ERA5  (full dataset, H100)"
echo " Start: $(date)"
echo "================================================================"
$PYTHON entry_point.py $COMMON \
    --ckpt-dir checkpoints/h4_runpod \
    >> $LOGS/h4_runpod.log 2>&1
echo "H4 EXIT_CODE $?" >> $LOGS/h4_runpod.log
echo "H4 done: $(date)"

echo "================================================================"
echo " H4b — no-cascade ablation + ERA5  (full dataset, H100)"
echo " Start: $(date)"
echo "================================================================"
$PYTHON entry_point.py $COMMON \
    --ckpt-dir checkpoints/h4b_runpod \
    --no-cascade \
    >> $LOGS/h4b_runpod.log 2>&1
echo "H4b EXIT_CODE $?" >> $LOGS/h4b_runpod.log
echo "H4b done: $(date)"

echo "================================================================"
echo " H5 Phase1 — freeze mani_head, train encoder+phys_head"
echo " Start: $(date)"
echo "================================================================"
$PYTHON entry_point.py $COMMON \
    --ckpt-dir checkpoints/h5_phase1_runpod \
    --freeze-stage mani \
    --lambda-cloud 0.0 --lambda-rain 0.0 --lambda-thermo 1.0 \
    >> $LOGS/h5_phase1_runpod.log 2>&1
echo "H5-Phase1 EXIT_CODE $?" >> $LOGS/h5_phase1_runpod.log
echo "H5 Phase1 done: $(date)"

# Verify Phase1 checkpoint exists before launching Phase2
PHASE1_CKPT=$PROJECT/checkpoints/h5_phase1_runpod/gpu0_best.pt
if [ ! -f "$PHASE1_CKPT" ]; then
    echo "ERROR: Phase1 best checkpoint not found at $PHASE1_CKPT — aborting."
    exit 1
fi
echo "Phase1 checkpoint verified: $PHASE1_CKPT"

echo "================================================================"
echo " H5 Phase2 — freeze encoder+phys_head, train mani_head only"
echo " Loads Phase1 weights, resets epoch/LR (--resume-weights-only)"
echo " Start: $(date)"
echo "================================================================"
$PYTHON entry_point.py $COMMON \
    --ckpt-dir checkpoints/h5_phase2_runpod \
    --freeze-stage encoder_phys \
    --resume-ckpt $PHASE1_CKPT \
    --resume-weights-only \
    --lambda-cloud 1.0 --lambda-rain 0.5 --lambda-thermo 0.0 \
    >> $LOGS/h5_phase2_runpod.log 2>&1
echo "H5-Phase2 EXIT_CODE $?" >> $LOGS/h5_phase2_runpod.log
echo "H5 Phase2 done: $(date)"

echo "================================================================"
echo " ALL RUNS COMPLETE: $(date)"
echo " Checkpoints:"
echo "   H4:         checkpoints/h4_runpod/gpu0_best.pt"
echo "   H4b:        checkpoints/h4b_runpod/gpu0_best.pt"
echo "   H5-Phase1:  checkpoints/h5_phase1_runpod/gpu0_best.pt"
echo "   H5-Phase2:  checkpoints/h5_phase2_runpod/gpu0_best.pt"
echo "================================================================"
