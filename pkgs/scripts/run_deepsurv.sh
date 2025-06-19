#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

nohup python -u -m pkgs.experiments.deepsurv > "${SCRIPT_DIR}/deepsurv.log" 2>&1 &

echo $! > "${SCRIPT_DIR}/deepsurv.pid"

echo "deepsurv started with PID $(cat "${SCRIPT_DIR}/deepsurv.pid")"