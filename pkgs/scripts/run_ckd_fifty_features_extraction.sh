#!/bin/bash
# Script to run CKD_FIFTY_FEATURES_HETEROGENEOUS data extraction in background
# Usage: ./pkgs/scripts/run_ckd_fifty_features_extraction.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"

# Create logs directory in scripts folder if it doesn't exist
mkdir -p "${SCRIPT_DIR}/logs"

# Get timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${SCRIPT_DIR}/logs/ckd_fifty_features_extraction_${TIMESTAMP}.log"
PID_FILE="${SCRIPT_DIR}/logs/ckd_fifty_features_extraction.pid"

echo "Starting CKD_FIFTY_FEATURES_HETEROGENEOUS data extraction..."
echo "Log file: ${LOG_FILE}"
echo "To monitor progress: tail -f ${LOG_FILE}"

# Run in background with nohup, redirect both stdout and stderr to log file
# Use PYTHONUNBUFFERED=1 to disable output buffering so logs appear in real-time
nohup bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate minhn2 && cd ${PROJECT_DIR} && PYTHONUNBUFFERED=1 python -m pkgs.data_analysis.model_data_store" > "${LOG_FILE}" 2>&1 &

# Save the PID
PID=$!
echo "Process started with PID: ${PID}"
echo "${PID}" > "${PID_FILE}"

echo "Done. Process running in background."
echo "To check if still running: ps -p ${PID}"
echo "To kill: kill ${PID}"
