#!/bin/bash

# Stage 3.0 scoped launcher: runs 10 models (all but `kfre`, which has no
# published equation for this scenario) for TWENTY_FEATURES_HETEROGENEOUS
# only, via pkgs/scripts/run_stage3_0_twenty_features.py. Only run this
# after the user has explicitly approved twenty_features_heterogeneous per
# EXPERIMENT_PLAN_DETAILS.md's Stage 3.0 scenario-ordering rule.
#
# Otherwise identical in structure/conventions to run_rep_stage3_0_four_eight.sh
# (and, by extension, run_rep.sh).
#
# Usage: pkgs/scripts/run_rep_stage3_0_twenty.sh <rep_number>

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ $# -ne 1 ] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <rep_number>"
    exit 1
fi

REP_NUM="$1"
LOG_FILE="${SCRIPT_DIR}/eval_all_rep${REP_NUM}_stage3_0_twenty.log"
MASTER_LOG="${SCRIPT_DIR}/run_rep${REP_NUM}_stage3_0_twenty_master.log"
PID_FILE="${SCRIPT_DIR}/run_rep${REP_NUM}_stage3_0_twenty.pid"

EXPERIMENTS=("cox" "dynamic_deephit" "hazard_transformer" "logistic_hazard" "rnnsurv" "deepsurv" "gbsa" "srf" "survival_svm" "weibul")

if [ "${RUN_REP_DETACHED:-0}" != "1" ]; then
    export RUN_REP_DETACHED=1
    setsid "$0" "$REP_NUM" > "$MASTER_LOG" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"

    echo "Started rep${REP_NUM} (twenty_features_heterogeneous only) in background with PID: $(cat "$PID_FILE")"
    echo "Rep log:    $LOG_FILE"
    echo "Master log: $MASTER_LOG"
    echo "Monitor with: tail -f $LOG_FILE"
    echo "Stop with:    kill -TERM -$(cat "$PID_FILE")"
    exit 0
fi

export PYTHONPATH="${PROJECT_ROOT}"
export CKD_REP="${REP_NUM}"

mkdir -p "${PROJECT_ROOT}/generated_data/rep${REP_NUM}"

echo "Starting rep${REP_NUM} (CKD_REP=${CKD_REP}), TWENTY_FEATURES_HETEROGENEOUS only..." | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Running experiments IN PARALLEL; per-experiment logs: ${SCRIPT_DIR}/eval_rep${REP_NUM}_stage3_0_twenty_<experiment>.log" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

declare -A exp_pids
declare -A exp_logs

for experiment in "${EXPERIMENTS[@]}"; do
    exp_log="${SCRIPT_DIR}/eval_rep${REP_NUM}_stage3_0_twenty_${experiment}.log"
    exp_logs["$experiment"]="$exp_log"

    (
        echo "Start time: $(date)" > "$exp_log"
        if python -m pkgs.scripts.run_stage3_0_twenty_features "${experiment}" >> "$exp_log" 2>&1; then
            echo "✓ $experiment completed successfully" >> "$exp_log"
        else
            echo "✗ $experiment failed with exit code $?" >> "$exp_log"
        fi
        echo "End time: $(date)" >> "$exp_log"
    ) &
    exp_pids["$experiment"]=$!
    echo "==================== Launched $experiment for rep${REP_NUM} (PID ${exp_pids[$experiment]}, log: $exp_log) ====================" | tee -a "$LOG_FILE"
done

failed_experiments=()

for experiment in "${!exp_pids[@]}"; do
    if wait "${exp_pids[$experiment]}"; then
        echo "✓ $experiment completed successfully (log: ${exp_logs[$experiment]})" | tee -a "$LOG_FILE"
    else
        echo "✗ $experiment failed (log: ${exp_logs[$experiment]})" | tee -a "$LOG_FILE"
        failed_experiments+=("$experiment")
    fi
done

echo "=========================================" | tee -a "$LOG_FILE"
echo "Rep${REP_NUM} (twenty_features_heterogeneous only) Summary:" | tee -a "$LOG_FILE"
echo "Total experiments run: ${#exp_pids[@]}" | tee -a "$LOG_FILE"
echo "Failed experiments: ${#failed_experiments[@]}" | tee -a "$LOG_FILE"

if [ ${#failed_experiments[@]} -eq 0 ]; then
    echo "✓ All experiments completed successfully!" | tee -a "$LOG_FILE"
else
    echo "✗ Failed experiments: ${failed_experiments[*]}" | tee -a "$LOG_FILE"
fi

echo "End time: $(date)" | tee -a "$LOG_FILE"
