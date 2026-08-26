#!/bin/bash

# Run all experiments for a single repetition in the background, IN PARALLEL
# (each experiment as its own subprocess, each with its own per-experiment
# log so parallel output doesn't interleave; the main LOG_FILE gets a
# launch/completion line per experiment plus the final summary).
# Unlike run_all_reps.sh (which mutates commons.py in place and thus can only
# run one rep at a time), this script selects the rep via the CKD_REP
# environment variable, so multiple reps can be launched concurrently
# (e.g. `run_rep.sh 2` then `run_rep.sh 3`) without racing each other.
#
# Env vars (optional):
#   EXCLUDE_EXPERIMENTS - space-separated experiment names to skip this run
#                         (e.g. `EXCLUDE_EXPERIMENTS="dynamic_deephit"`),
#                         for when one is already running elsewhere for the
#                         same rep and shouldn't be started a second time.
#   RUN_TAG             - suffix appended to this invocation's log/pid file
#                         names (e.g. `RUN_TAG=resume`), so a second
#                         `run_rep.sh` invocation for a rep that already has
#                         a live process doesn't clobber its log/pid files.
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
TAG_SUFFIX="${RUN_TAG:+_${RUN_TAG}}"
LOG_FILE="${SCRIPT_DIR}/eval_all_rep${REP_NUM}${TAG_SUFFIX}.log"
MASTER_LOG="${SCRIPT_DIR}/run_rep${REP_NUM}${TAG_SUFFIX}_master.log"
PID_FILE="${SCRIPT_DIR}/run_rep${REP_NUM}${TAG_SUFFIX}.pid"

EXPERIMENTS=("cox" "dynamic_deephit" "hazard_transformer" "logistic_hazard" "rnnsurv" "kfre")
EXCLUDE_EXPERIMENTS="${EXCLUDE_EXPERIMENTS:-}"

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
echo "Running experiments IN PARALLEL; per-experiment logs: ${SCRIPT_DIR}/eval_rep${REP_NUM}${TAG_SUFFIX}_<experiment>.log" | tee -a "$LOG_FILE"
if [ -n "$EXCLUDE_EXPERIMENTS" ]; then
    echo "Excluded this run: ${EXCLUDE_EXPERIMENTS}" | tee -a "$LOG_FILE"
fi
echo "=========================================" | tee -a "$LOG_FILE"

declare -A exp_pids
declare -A exp_logs

for experiment in "${EXPERIMENTS[@]}"; do
    skip=0
    for ex in $EXCLUDE_EXPERIMENTS; do
        if [ "$ex" == "$experiment" ]; then
            skip=1
            break
        fi
    done
    if [ "$skip" -eq 1 ]; then
        echo "Skipping $experiment (in EXCLUDE_EXPERIMENTS)" | tee -a "$LOG_FILE"
        continue
    fi

    exp_log="${SCRIPT_DIR}/eval_rep${REP_NUM}${TAG_SUFFIX}_${experiment}.log"
    exp_logs["$experiment"]="$exp_log"

    (
        echo "Start time: $(date)" > "$exp_log"
        if python -m "pkgs.experiments.${experiment}" >> "$exp_log" 2>&1; then
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
echo "Rep${REP_NUM} Summary:" | tee -a "$LOG_FILE"
echo "Total experiments run: ${#exp_pids[@]}" | tee -a "$LOG_FILE"
echo "Failed experiments: ${#failed_experiments[@]}" | tee -a "$LOG_FILE"

if [ ${#failed_experiments[@]} -eq 0 ]; then
    echo "✓ All experiments completed successfully!" | tee -a "$LOG_FILE"
else
    echo "✗ Failed experiments: ${failed_experiments[*]}" | tee -a "$LOG_FILE"
fi

echo "End time: $(date)" | tee -a "$LOG_FILE"
