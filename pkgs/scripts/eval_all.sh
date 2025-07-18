#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

nohup bash -c "python -m pkgs.experiments.cox && python -m pkgs.experiments.deepsurv && python -m pkgs.experiments.dynamic_deephit && python -m pkgs.experiments.gbsa && python -m pkgs.experiments.hazard_transformer && python -m pkgs.experiments.rnnsurv && python -m pkgs.experiments.weibul" > "${SCRIPT_DIR}/eval_all_rep_2.log" 2>&1 &

echo $! > "${SCRIPT_DIR}/eval_all.pid"

echo "Evaluation started with PID $(cat "${SCRIPT_DIR}/eval_all.pid")" >> "${SCRIPT_DIR}/eval_all_rep_2.log"