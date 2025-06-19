#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

nohup python -u -m pkgs.experiments.gbsa > "${SCRIPT_DIR}/gbsa.log" 2>&1 &

echo $! > "${SCRIPT_DIR}/gbsa.pid"

echo "gbsa started with PID $(cat "${SCRIPT_DIR}/gbsa.pid")"