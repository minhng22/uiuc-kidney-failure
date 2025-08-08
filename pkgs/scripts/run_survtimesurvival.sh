#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

nohup python -u -m pkgs.experiments.survtimesurvival > "${SCRIPT_DIR}/survtimesurvival_rep-1.log" 2>&1 &

echo $! > "${SCRIPT_DIR}/survtimesurvival-1.pid"

echo "survtimesurvival started with PID $(cat "${SCRIPT_DIR}/survtimesurvival-1.pid")"