#!/bin/bash

# Script to run Survival SVM experiments
# Runs experiments for all scenarios: non-time-variant, time-variant, heterogeneous, and egfr components

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

# Get current rep number from commons.py to determine log file name
REP_NUM=$(grep "generate_data_path_latest_rep" "${PROJECT_ROOT}/pkgs/commons.py" | sed -n "s/.*rep\([0-9]*\).*/\1/p")
LOG_FILE="${SCRIPT_DIR}/survival_svm_rep${REP_NUM}.log"

echo "Running Survival SVM experiments..." | tee "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run the Survival SVM experiment
if python -m pkgs.experiments.survival_svm >> "$LOG_FILE" 2>&1; then
    echo "" | tee -a "$LOG_FILE"
    echo "✓ Survival SVM experiments completed successfully" | tee -a "$LOG_FILE"
    echo "End time: $(date)" | tee -a "$LOG_FILE"
    exit 0
else
    echo "" | tee -a "$LOG_FILE"
    echo "✗ Survival SVM experiments failed with exit code $?" | tee -a "$LOG_FILE"
    echo "End time: $(date)" | tee -a "$LOG_FILE"
    exit 1
fi
