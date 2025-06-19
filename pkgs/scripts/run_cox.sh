#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

nohup python -u -m pkgs.experiments.cox > "${SCRIPT_DIR}/cox.log" 2>&1 &

echo $! > "${SCRIPT_DIR}/cox.pid"

echo "cox started with PID $(cat "${SCRIPT_DIR}/cox.pid")"