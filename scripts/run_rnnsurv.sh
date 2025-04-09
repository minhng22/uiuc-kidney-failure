#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

nohup python -u -m pkgs.experiments.rnnsurv > "${SCRIPT_DIR}/rnnsurv.log" 2>&1 &

echo $! > "${SCRIPT_DIR}/rnnsurv.pid"

echo "rnnsurv started with PID $(cat ${SCRIPT_DIR}/rnnsurv.pid)"