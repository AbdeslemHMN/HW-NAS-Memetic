#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_full_pipeline.sh  —  HW-NAS-Memetic end-to-end research pipeline
#
# Runs a full 3-dataset × 3-hardware sweep:
#   1. Data health-check
#   2. Proposed search   (run_search.py)
#   3. Baselines         (run_baselines.py)
#   4. Ablation study    (run_ablations.py)
#   5. Incremental build (run_incremental_build.py)
#   6. Waterfall sweep   (run_waterfall.py)  — 30 seeds × stages 1–3
#
# All per-run stdout+stderr are captured to:
#   logs/<dataset>/<hardware>/<script>_<TIMESTAMP>.log
#
# Usage:
#   chmod +x run_full_pipeline.sh
#   ./run_full_pipeline.sh                          # full sweep
#   DATASETS="cifar10" HARDWARE="edgegpu_latency" ./run_full_pipeline.sh
#   DRY_RUN=1 ./run_full_pipeline.sh                # print commands, don't run
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configurable sweep dimensions ────────────────────────────────────────────
DATASETS="${DATASETS:-cifar10 cifar100 ImageNet16-120}"
HARDWARE="${HARDWARE:-edgegpu_latency raspi4_latency eyeriss_latency}"

# ── Optional flags ────────────────────────────────────────────────────────────
DRY_RUN="${DRY_RUN:-0}"           # Set to 1 to print commands without running

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOGS_ROOT="$PROJECT_ROOT/logs"
PYTHON="${PYTHON:-python}"        # override with venv python if needed

# ── Timestamp for this pipeline run ──────────────────────────────────────────
RUN_TS="$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="$LOGS_ROOT/pipeline_${RUN_TS}.log"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_log() {
    local level="$1"; shift
    local msg="$*"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf "[%s] [%s] %s\n" "$ts" "$level" "$msg" | tee -a "$PIPELINE_LOG"
}
info()  { _log "INFO " "$@"; }
warn()  { _log "WARN " "$@"; }
error() { _log "ERROR" "$@"; }
step()  { printf "\n"; _log "STEP " "══════  %s  ══════" "$*"; }

# Run a Python command, capture output to a log file, and stream progress
run_py() {
    local log_path="$1"; shift
    local cmd=("$PYTHON" "$@")

    mkdir -p "$(dirname "$log_path")"

    if [[ "$DRY_RUN" == "1" ]]; then
        info "[DRY-RUN] ${cmd[*]}  >> $log_path"
        return 0
    fi

    info "Running: ${cmd[*]}"
    info "Log    : $log_path"

    # Tee stdout+stderr to the log file AND to our pipeline log
    # We intentionally do NOT use set -e inside the subprocess — let caller
    # decide whether a failure is fatal.
    if "${cmd[@]}" >> "$log_path" 2>&1; then
        info "OK — $(basename "$log_path")"
    else
        local rc=$?
        error "FAILED (exit $rc) — see $log_path"
        return $rc
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Environment bootstrap
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p "$LOGS_ROOT"
info "Pipeline run ID : $RUN_TS"
info "Project root    : $PROJECT_ROOT"
info "Datasets        : $DATASETS"
info "Hardware        : $HARDWARE"
info "Pipeline log    : $PIPELINE_LOG"

# Activate venv if present and not already active
if [[ -z "${VIRTUAL_ENV:-}" && -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/venv/bin/activate"
    info "Activated venv: $VIRTUAL_ENV"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Data health-check
# ─────────────────────────────────────────────────────────────────────────────
step "Data health-check"

HW_PICKLE="$PROJECT_ROOT/data/HW-NAS-Bench-v1_0.pickle"
ACC_CACHE="$PROJECT_ROOT/data/nas201_accuracy_cache.npz"
NATS_DIR="$PROJECT_ROOT/data/NATS-tss-v1_0-3ffb9-simple"

_check_file() {
    local path="$1"
    local label="$2"
    if [[ -e "$path" ]]; then
        info "  [OK]      $label"
    else
        warn "  [MISSING] $label → attempting download…"
        return 1
    fi
}

HEALTHCHECK_FAILED=0

if ! _check_file "$HW_PICKLE" "HW-NAS-Bench-v1_0.pickle"; then
    if [[ "$DRY_RUN" != "1" ]]; then
        run_py "$LOGS_ROOT/download_hw_nas_${RUN_TS}.log" \
            scripts/download_data.py || HEALTHCHECK_FAILED=1
    fi
fi

if ! _check_file "$NATS_DIR" "NATS-tss-v1_0-3ffb9-simple/"; then
    if [[ "$DRY_RUN" != "1" ]]; then
        run_py "$LOGS_ROOT/download_nats_${RUN_TS}.log" \
            scripts/download_data.py --nats-bench || HEALTHCHECK_FAILED=1
    fi
fi

if ! _check_file "$ACC_CACHE" "nas201_accuracy_cache.npz"; then
    if [[ -e "$NATS_DIR" && "$DRY_RUN" != "1" ]]; then
        info "Extracting accuracy cache from NATS-Bench…"
        run_py "$LOGS_ROOT/extract_accuracy_${RUN_TS}.log" \
            scripts/extract_nas201_accuracy.py || HEALTHCHECK_FAILED=1
    else
        warn "  Skipping cache extraction (NATS dir missing or dry-run)."
    fi
fi

if [[ "$HEALTHCHECK_FAILED" -eq 1 ]]; then
    error "Health-check failed. Fix data issues before continuing."
    exit 1
fi

# Mandatory: HW-NAS-Bench pickle must exist before any algorithm runs
if [[ ! -e "$HW_PICKLE" && "$DRY_RUN" != "1" ]]; then
    error "HW-NAS-Bench-v1_0.pickle not found after download attempt. Aborting."
    exit 1
fi

info "Health-check passed."

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2-5 — Algorithm sweep
# ─────────────────────────────────────────────────────────────────────────────
step "Algorithm sweep  (${DATASETS// /,} × ${HARDWARE// /,})"

TOTAL_COMBOS=0
for _ds in $DATASETS; do for _hw in $HARDWARE; do
    TOTAL_COMBOS=$(( TOTAL_COMBOS + 1 ))
done; done

COMBO_IDX=0
for DATASET in $DATASETS; do
    for HW in $HARDWARE; do
        COMBO_IDX=$(( COMBO_IDX + 1 ))
        LOG_DIR="$LOGS_ROOT/$DATASET/$HW"
        mkdir -p "$LOG_DIR"

        step "[$COMBO_IDX/$TOTAL_COMBOS] dataset=$DATASET  hardware=$HW"

        # ── Proposed search ───────────────────────────────────────────────
        info "  → run_search.py"
        run_py "$LOG_DIR/search_${RUN_TS}.log" \
            scripts/run_search.py \
            --dataset   "$DATASET" \
            --hardware  "$HW"

        # ── Baselines ─────────────────────────────────────────────────────
        info "  → run_baselines.py"
        run_py "$LOG_DIR/baselines_${RUN_TS}.log" \
            scripts/run_baselines.py \
            --dataset   "$DATASET" \
            --hardware  "$HW"

        # ── Ablation study ────────────────────────────────────────────────
        info "  → run_ablations.py"
        run_py "$LOG_DIR/ablations_${RUN_TS}.log" \
            scripts/run_ablations.py \
            --dataset   "$DATASET" \
            --hardware  "$HW"
        # ── Component-isolation baselines (30 seeds × 4 variants) ────────────────
        info "  → run_component_baselines.py"
        run_py "$LOG_DIR/component_baselines_${RUN_TS}.log" \
            scripts/run_component_baselines.py \
            --dataset   "$DATASET" \
            --hardware  "$HW"
        # ── Incremental build ─────────────────────────────────────────────
        info "  → run_incremental_build.py"
        run_py "$LOG_DIR/incremental_${RUN_TS}.log" \
            scripts/run_incremental_build.py \
            --dataset   "$DATASET" \
            --hardware  "$HW"

        # ── Waterfall sweep (30 seeds × stages 1–3) ───────────────────────
        info "  → run_waterfall.py"
        run_py "$LOG_DIR/waterfall_${RUN_TS}.log" \
            scripts/run_waterfall.py \
            --dataset   "$DATASET" \
            --hardware  "$HW"

        info "  Combo $COMBO_IDX/$TOTAL_COMBOS complete."
    done
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Summary
# ─────────────────────────────────────────────────────────────────────────────
step "Pipeline complete"
info "Run timestamp  : $RUN_TS"
info "Results root   : $PROJECT_ROOT/results/"
info "Logs root      : $LOGS_ROOT/"
info "Pipeline log   : $PIPELINE_LOG"

if command -v tree &>/dev/null; then
    info "Log tree:"
    tree -L 4 "$LOGS_ROOT" | tee -a "$PIPELINE_LOG"
fi

info "Done."
# ─────────────────────────────────────────────────────────────────────────────
step "Pipeline complete"
info "Run timestamp  : $RUN_TS"
info "Results root   : $PROJECT_ROOT/results/"
info "Logs root      : $LOGS_ROOT/"
info "Pipeline log   : $PIPELINE_LOG"

if command -v tree &>/dev/null; then
    info "Log tree:"
    tree -L 4 "$LOGS_ROOT" | tee -a "$PIPELINE_LOG"
fi

info "Done."
