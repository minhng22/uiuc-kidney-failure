#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

nohup python -u -m pkgs.experiments.hazard_transformer > "${SCRIPT_DIR}/hazard_transformer.log" 2>&1 &

echo $! > "${SCRIPT_DIR}/hazard_transformer.pid"

echo "hazard_transformer started with PID $(cat ${SCRIPT_DIR}/hazard_transformer.pid)"