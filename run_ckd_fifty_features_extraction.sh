#!/bin/bash
# Script to run CKD_FIFTY_FEATURES_HETEROGENEOUS data extraction in background
# Usage: ./run_ckd_fifty_features_extraction.sh

# Activate conda environment and run the extraction
cd /home/minhn2/uiuc-kidney-failure

# Create logs directory if it doesn't exist
mkdir -p logs

# Get timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/ckd_fifty_features_extraction_${TIMESTAMP}.log"

echo "Starting CKD_FIFTY_FEATURES_HETEROGENEOUS data extraction..."
echo "Log file: ${LOG_FILE}"
echo "To monitor progress: tail -f ${LOG_FILE}"

# Run in background with nohup, redirect both stdout and stderr to log file
nohup bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate minhn2 && python -m pkgs.data_analysis.model_data_store" > "${LOG_FILE}" 2>&1 &

# Save the PID
PID=$!
echo "Process started with PID: ${PID}"
echo "${PID}" > logs/ckd_fifty_features_extraction.pid

echo "Done. Process running in background."
echo "To check if still running: ps -p ${PID}"
echo "To kill: kill ${PID}"
