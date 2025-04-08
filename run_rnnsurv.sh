#!/bin/bash
# Determine the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Run the command in the background using nohup
nohup python -m pkgs.experiments.rnnsurv > "${SCRIPT_DIR}/rnnsurv.log" 2>&1 &

# Save the process ID (PID) to a file within the same directory for later tracking or stopping
echo $! > "${SCRIPT_DIR}/rnnsurv.pid"

echo "rnnsurv started with PID $(cat ${SCRIPT_DIR}/rnnsurv.pid)"