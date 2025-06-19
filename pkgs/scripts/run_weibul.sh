#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

nohup python -u -m pkgs.experiments.weibul > "${SCRIPT_DIR}/weibul.log" 2>&1 &

echo $! > "${SCRIPT_DIR}/weibul.pid"

echo "weibul started with PID $(cat "${SCRIPT_DIR}/weibul.pid")"