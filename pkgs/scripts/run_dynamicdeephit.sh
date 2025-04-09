#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

nohup python -u -m pkgs.experiments.dynamic_deephit > "${SCRIPT_DIR}/dynamic_deephit.log" 2>&1 &

echo $! > "${SCRIPT_DIR}/dynamic_deephit.pid"

echo "dynamic_deephit started with PID $(cat "${SCRIPT_DIR}/dynamic_deephit.pid")"