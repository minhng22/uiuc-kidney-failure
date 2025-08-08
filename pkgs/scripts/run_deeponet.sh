#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

nohup python -u -m pkgs.experiments.deeponet > "${SCRIPT_DIR}/deeponet_rep-5.log" 2>&1 &

echo $! > "${SCRIPT_DIR}/deeponet-5.pid"

echo "deeponet started with PID $(cat "${SCRIPT_DIR}/deeponet-5.pid")"