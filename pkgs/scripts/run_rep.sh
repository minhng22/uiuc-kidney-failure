#!/bin/bash

# Run all experiments for a single repetition in the background.
# Unlike run_all_reps.sh (which mutates commons.py in place and thus can only
# run one rep at a time), this script selects the rep via the CKD_REP
# environment variable, so multiple reps can be launched concurrently
# (e.g. `run_rep.sh 2` then `run_rep.sh 3`) without racing each other.
#
# Usage: pkgs/scripts/run_rep.sh <rep_number>

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ $# -ne 1 ] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <rep_number>"
    exit 1
fi

REP_NUM="$1"
LOG_FILE="${SCRIPT_DIR}/eval_all_rep${REP_NUM}.log"
MASTER_LOG="${SCRIPT_DIR}/run_rep${REP_NUM}_master.log"
PID_FILE="${SCRIPT_DIR}/run_rep${REP_NUM}.pid"

EXPERIMENTS=("cox" "dynamic_deephit" "hazard_transformer" "logistic_hazard" "rnnsurv")

# On the first invocation, re-exec ourselves under setsid so the whole run
# survives the launching shell exiting, then return immediately. The
# RUN_REP_DETACHED guard makes the re-exec'd process fall through to the
# actual work below instead of forking again.
if [ "${RUN_REP_DETACHED:-0}" != "1" ]; then
    export RUN_REP_DETACHED=1
    setsid "$0" "$REP_NUM" > "$MASTER_LOG" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"

    echo "Started rep${REP_NUM} in background with PID: $(cat "$PID_FILE")"
    echo "Rep log:    $LOG_FILE"
    echo "Master log: $MASTER_LOG"
    echo "Monitor with: tail -f $LOG_FILE"
    echo "Stop with:    kill -TERM -$(cat "$PID_FILE")"
    exit 0
fi

export PYTHONPATH="${PROJECT_ROOT}"
export CKD_REP="${REP_NUM}"

mkdir -p "${PROJECT_ROOT}/generated_data/rep${REP_NUM}"

echo "Starting rep${REP_NUM} (CKD_REP=${CKD_REP})..." | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

failed_experiments=()

for experiment in "${EXPERIMENTS[@]}"; do
    echo "==================== Running $experiment for rep${REP_NUM} ====================" | tee -a "$LOG_FILE"
    echo "Start time: $(date)" | tee -a "$LOG_FILE"

    if python -m "pkgs.experiments.${experiment}" >> "$LOG_FILE" 2>&1; then
        echo "✓ $experiment completed successfully" | tee -a "$LOG_FILE"
    else
        echo "✗ $experiment failed with exit code $?" | tee -a "$LOG_FILE"
        failed_experiments+=("$experiment")
    fi
    echo "End time: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "=========================================" | tee -a "$LOG_FILE"
echo "Rep${REP_NUM} Summary:" | tee -a "$LOG_FILE"
echo "Total experiments: ${#EXPERIMENTS[@]}" | tee -a "$LOG_FILE"
echo "Failed experiments: ${#failed_experiments[@]}" | tee -a "$LOG_FILE"

if [ ${#failed_experiments[@]} -eq 0 ]; then
    echo "✓ All experiments completed successfully!" | tee -a "$LOG_FILE"
else
    echo "✗ Failed experiments: ${failed_experiments[*]}" | tee -a "$LOG_FILE"
fi

echo "End time: $(date)" | tee -a "$LOG_FILE"
