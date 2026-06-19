#!/bin/bash
# runpod_run_all.sh — robust sequential trainer (H6 → H6b → H7-Phase1 → H7-Phase2)
# Full dataset, H100 80GB, ERA5 dense supervision.
#
# Features:
#   - Each run trains all 30 epochs, checkpoints isolated per run.
#   - Phone notifications on start/done/fail via ntfy.sh (set NTFY_TOPIC).
#   - OOM auto-recovery: retry once with batch-size 32.
#   - On non-OOM failure: writes Claude diagnosis to PROGRESS_LOG, notifies, halts.
#   - Maintains PROGRESS_LOG.md the user can read or download.
#   - On full completion: prints exact scp pull commands + notifies.
#
# Setup before running:
#   export NTFY_TOPIC=horizon-<your-random-string>   # subscribe same topic in ntfy app on phone
#   export ANTHROPIC_API_KEY=sk-ant-...         # for failure diagnosis (optional)
#
# Usage (from /workspace/project):
#   cd /workspace/project && mkdir -p logs && nohup bash /workspace/runpod_run_all.sh > logs/runpod_run_all.log 2>&1 &
#
# Naming: H6=local-H4 (cascade+ERA5), H6b=local-H4b (no-cascade), H7=local-H5 (two-phase).

PROJECT=/workspace/project
# Use venv python if it exists, else system python (RunPod base image installs deps globally)
if [ -x "$PROJECT/venv/bin/python" ]; then PYTHON="$PROJECT/venv/bin/python"; else PYTHON="$(command -v python || command -v python3)"; fi
LOGS=$PROJECT/logs
PROGRESS=$PROJECT/PROGRESS_LOG.md
NTFY_TOPIC="${NTFY_TOPIC:-}"

mkdir -p "$LOGS" "$PROJECT/checkpoints"
cd "$PROJECT" || exit 1

# ── helpers ─────────────────────────────────────────────────────────────────
notify() {
    echo "[$(date '+%F %T')] $1"
    if [ -n "$NTFY_TOPIC" ]; then
        curl -s -H "Title: Horizon Training" -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
    fi
}

log() {
    echo "$1" >> "$PROGRESS"
}

# Claude diagnoses a failure (writes analysis, does NOT auto-edit code).
diagnose() {
    local name="$1" logf="$2"
    if command -v claude >/dev/null 2>&1 && [ -n "$ANTHROPIC_API_KEY" ]; then
        local tail_log
        tail_log=$(tail -60 "$logf")
        local analysis
        analysis=$(claude -p "Training run '$name' for the Horizon Forecast nowcasting model crashed on a RunPod H100. Read /workspace/CLAUDE_CONTEXT.md for project context. Here are the last 60 log lines:

$tail_log

In 3-5 lines: what failed, the likely root cause, and the exact fix. Be specific and terse." 2>/dev/null)
        log ""
        log "### Claude diagnosis for $name failure"
        log '```'
        log "$analysis"
        log '```'
        notify "Claude diagnosis ($name): $(echo "$analysis" | head -c 250)"
    fi
}

# ── run one experiment: all 30 epochs, OOM auto-retry ────────────────────────
# args: <name> <extra args for entry_point.py...>
run_one() {
    local name="$1"; shift
    local logf="$LOGS/$name.log"
    local bs=48

    notify "▶ $name starting — 30 epochs (batch=$bs)"
    log ""
    log "## $name"
    log "- started: $(date '+%F %T')"

    while :; do
        $PYTHON entry_point.py \
            --device-id 0 --precision bf16 \
            --batch-size $bs --grad-accum 2 --val-batch-size 64 \
            --num-workers 32 --rollout-max 2 \
            --max-epochs 30 --multihorizon-every 10 \
            --train-csv data/processed/index_train.csv \
            --era5-path data/era5_npy \
            "$@" >> "$logf" 2>&1
        local code=$?

        if [ $code -eq 0 ]; then
            notify "✔ $name DONE — checkpoint saved"
            log "- finished: $(date '+%F %T') — SUCCESS (batch=$bs)"
            return 0
        fi

        # OOM? retry once at batch 32.
        if grep -qiE "out of memory|CUDA out of memory|OutOfMemory" "$logf" && [ $bs -gt 32 ]; then
            bs=32
            notify "⚠ $name hit OOM — auto-retrying at batch=$bs"
            log "- OOM detected, retry at batch=$bs — $(date '+%F %T')"
            continue
        fi

        # Non-recoverable failure.
        notify "✖ $name FAILED (exit $code). $(tail -3 "$logf" | tr '\n' ' ' | head -c 200)"
        log "- **FAILED** exit $code — $(date '+%F %T')"
        diagnose "$name" "$logf"
        return $code
    done
}

# ── PROGRESS_LOG header ──────────────────────────────────────────────────────
if [ ! -f "$PROGRESS" ]; then
    cat > "$PROGRESS" << 'HDR'
# RunPod Training Progress — Horizon Forecast (H6 / H6b / H7)

Full-dataset H100 runs. Authors: Or Mordechay Hod + Gilad Boudman. Code 26-1-R-1.

Naming: H6 = cascade+ERA5 (local H4), H6b = no-cascade ablation (local H4b),
H7 = two-phase frozen (local H5). Each run = 30 epochs. CSI@1mm baseline = 0.167.

---
HDR
fi

notify "=== Horizon training pipeline started on H100 ==="
log "Pipeline started: $(date '+%F %T')"

# ── H6 — cascade + ERA5 ───────────────────────────────────────────────────────
run_one h6 --ckpt-dir checkpoints/h6 || exit 1

# ── H6b — no-cascade ablation ─────────────────────────────────────────────────
run_one h6b --ckpt-dir checkpoints/h6b --no-cascade || exit 1

# ── H7 Phase1 — freeze mani_head, train encoder+phys ──────────────────────────
run_one h7_phase1 --ckpt-dir checkpoints/h7_phase1 \
    --freeze-stage mani \
    --lambda-cloud 0.0 --lambda-rain 0.0 --lambda-thermo 1.0 || exit 1

# Verify Phase1 checkpoint before Phase2.
PHASE1_CKPT=$PROJECT/checkpoints/h7_phase1/gpu0_best.pt
if [ ! -f "$PHASE1_CKPT" ]; then
    notify "✖ H7-Phase1 checkpoint missing at $PHASE1_CKPT — cannot start Phase2"
    log "- **ABORT** H7-Phase1 checkpoint not found — $(date '+%F %T')"
    exit 1
fi

# ── H7 Phase2 — freeze encoder+phys, train mani_head; load Phase1 weights ─────
run_one h7_phase2 --ckpt-dir checkpoints/h7_phase2 \
    --freeze-stage encoder_phys \
    --resume-ckpt "$PHASE1_CKPT" --resume-weights-only \
    --lambda-cloud 1.0 --lambda-rain 0.5 --lambda-thermo 0.0 || exit 1

# ── all done ──────────────────────────────────────────────────────────────────
log ""
log "## ALL RUNS COMPLETE — $(date '+%F %T')"
log "Checkpoints ready to download:"
log "- checkpoints/h6/gpu0_best.pt"
log "- checkpoints/h6b/gpu0_best.pt"
log "- checkpoints/h7_phase1/gpu0_best.pt"
log "- checkpoints/h7_phase2/gpu0_best.pt"

notify "ALL 4 RUNS COMPLETE. Download checkpoints+logs to your PC, then stop the pod. See PROGRESS_LOG.md for scp commands."

cat << 'PULL'
================================================================
ALL RUNS COMPLETE.

Download from your LOCAL PowerShell (pod cannot push to your PC).
Replace <PORT>, <POD_IP>, <SSH_KEY>, <LOCAL_DEST> with your values:

  scp -P <PORT> -i "<SSH_KEY>" -r root@<POD_IP>:/workspace/project/checkpoints "<LOCAL_DEST>"
  scp -P <PORT> -i "<SSH_KEY>" -r root@<POD_IP>:/workspace/project/logs "<LOCAL_DEST>"
  scp -P <PORT> -i "<SSH_KEY>" root@<POD_IP>:/workspace/project/PROGRESS_LOG.md "<LOCAL_DEST>"

After download verified -> STOP the pod in RunPod dashboard to stop billing.
================================================================
PULL
