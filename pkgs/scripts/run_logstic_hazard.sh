#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

nohup python -u -m pkgs.experiments.logistic_hazard > "${SCRIPT_DIR}/logistic_hazard_rep5.log" 2>&1 &

echo $! > "${SCRIPT_DIR}/logistic_hazard_rep5.pid"

echo "logistic_hazard started with PID $(cat "${SCRIPT_DIR}/logistic_hazard_rep5.pid")"