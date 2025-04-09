#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

nohup python -u -m pkgs.experiments.hazard_transformer > "${SCRIPT_DIR}/hazard_transformer.log" 2>&1 &

echo $! > "${SCRIPT_DIR}/hazard_transformer.pid"

echo "hazard_transformer started with PID $(cat "${SCRIPT_DIR}/hazard_transformer.pid")"