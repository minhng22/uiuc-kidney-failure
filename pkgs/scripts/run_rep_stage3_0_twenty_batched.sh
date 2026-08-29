#!/bin/bash

# Stage 3.0.2 remediation driver: same 10 models / same
# pkgs.scripts.run_stage3_0_twenty_features entry point as
# run_rep_stage3_0_twenty.sh, but launched in small BATCHES (parallel
# within a batch, sequential across batches) instead of all 10 at once.
#
# Why: the original all-10-parallel launch (run_rep_stage3_0_twenty.sh,
# PID 3011811, 2026-08-29 02:37 CDT on sunlab-serv-01) OOM-killed at least
# gbsa/hazard_transformer/weibul (confirmed exit 137 in their own
# per-experiment logs) and very likely rnnsurv (killed per the master
# log's job-control notice) within ~20 minutes, with the other 6 logs
# still frozen (zero bytes past "Start time") 78+ minutes later --
# consistent with 10 processes each independently loading
# TWENTY_FEATURES_HETEROGENEOUS's full get_train_test_data() frame
# (~455.8 rows/patient, by far the largest of the 3 scenarios --
# see stage2_2_debug_report.txt Finding #3) simultaneously exhausting
# host RAM before most of them even reached training. See
# generated_data/rep1/stage3_0_background_process_log.txt for the full
# diagnosis. Not a per-model code bug -- a resource-limit issue in the
# original launcher's all-parallel design for this specific scenario.
#
# Usage: pkgs/scripts/run_rep_stage3_0_twenty_batched.sh <rep_number>

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ $# -ne 1 ] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <rep_number>"
    exit 1
fi

REP_NUM="$1"
LOG_FILE="${SCRIPT_DIR}/eval_all_rep${REP_NUM}_stage3_0_twenty_batched.log"
MASTER_LOG="${SCRIPT_DIR}/run_rep${REP_NUM}_stage3_0_twenty_batched_master.log"
PID_FILE="${SCRIPT_DIR}/run_rep${REP_NUM}_stage3_0_twenty_batched.pid"

BATCH_1=("cox" "logistic_hazard" "deepsurv")
BATCH_2=("dynamic_deephit" "hazard_transformer" "srf")
BATCH_3=("rnnsurv" "gbsa" "survival_svm")
BATCH_4=("weibul")

if [ "${RUN_REP_DETACHED:-0}" != "1" ]; then
    export RUN_REP_DETACHED=1
    setsid "$0" "$REP_NUM" > "$MASTER_LOG" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"

    echo "Started rep${REP_NUM} (twenty_features_heterogeneous, batched) in background with PID: $(cat "$PID_FILE")"
    echo "Rep log:    $LOG_FILE"
    echo "Master log: $MASTER_LOG"
    echo "Monitor with: tail -f $LOG_FILE"
    echo "Stop with:    kill -TERM -$(cat "$PID_FILE")"
    exit 0
fi

export PYTHONPATH="${PROJECT_ROOT}"
export CKD_REP="${REP_NUM}"

mkdir -p "${PROJECT_ROOT}/generated_data/rep${REP_NUM}"

echo "Starting rep${REP_NUM} (CKD_REP=${CKD_REP}), TWENTY_FEATURES_HETEROGENEOUS, BATCHED..." | tee "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

overall_failed=()
batch_num=0
for batch_name in BATCH_1 BATCH_2 BATCH_3 BATCH_4; do
    batch_num=$((batch_num + 1))
    eval "experiments=(\"\${${batch_name}[@]}\")"
    echo "--- Batch ${batch_num}: ${experiments[*]} ---" | tee -a "$LOG_FILE"

    declare -A exp_pids
    for experiment in "${experiments[@]}"; do
        exp_log="${SCRIPT_DIR}/eval_rep${REP_NUM}_stage3_0_twenty_${experiment}.log"
        (
            echo "Start time: $(date)" > "$exp_log"
            python -m pkgs.scripts.run_stage3_0_twenty_features "${experiment}" >> "$exp_log" 2>&1
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                echo "✓ $experiment completed successfully" >> "$exp_log"
            else
                echo "✗ $experiment failed with exit code $exit_code" >> "$exp_log"
            fi
            echo "End time: $(date)" >> "$exp_log"
            exit $exit_code
        ) &
        exp_pids["$experiment"]=$!
        echo "Launched $experiment (PID ${exp_pids[$experiment]}, log: $exp_log)" | tee -a "$LOG_FILE"
    done

    for experiment in "${!exp_pids[@]}"; do
        if wait "${exp_pids[$experiment]}"; then
            echo "✓ $experiment completed successfully" | tee -a "$LOG_FILE"
        else
            echo "✗ $experiment failed" | tee -a "$LOG_FILE"
            overall_failed+=("$experiment")
        fi
    done
    unset exp_pids
    echo "--- Batch ${batch_num} done ---" | tee -a "$LOG_FILE"
done

echo "=========================================" | tee -a "$LOG_FILE"
echo "Rep${REP_NUM} (twenty_features_heterogeneous, batched) Summary:" | tee -a "$LOG_FILE"
echo "Failed experiments: ${#overall_failed[@]}" | tee -a "$LOG_FILE"
if [ ${#overall_failed[@]} -eq 0 ]; then
    echo "✓ All experiments completed successfully!" | tee -a "$LOG_FILE"
else
    echo "✗ Failed experiments: ${overall_failed[*]}" | tee -a "$LOG_FILE"
fi
echo "End time: $(date)" | tee -a "$LOG_FILE"
