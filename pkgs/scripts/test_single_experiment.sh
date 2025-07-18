#!/bin/bash

# Usage: ./test_single_experiment.sh <experiment_name> <rep_number>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <experiment_name> <rep_number>"
    echo "Example: $0 cox 2"
    exit 1
fi

EXPERIMENT=$1
REP_NUM=$2

echo "Testing experiment: $EXPERIMENT for rep$REP_NUM"
echo "Project root: $PROJECT_ROOT"

echo "Updating commons.py for rep$REP_NUM..."
if python "${SCRIPT_DIR}/update_rep.py" "$REP_NUM"; then
    echo "✓ Commons.py updated successfully"
else
    echo "✗ Failed to update commons.py"
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/generated_data/rep${REP_NUM}"

echo "Running experiment: $EXPERIMENT"
echo "Start time: $(date)"

if python -m "pkgs.experiments.${EXPERIMENT}"; then
    echo "✓ $EXPERIMENT completed successfully"
    echo "End time: $(date)"
else
    echo "✗ $EXPERIMENT failed"
    echo "End time: $(date)"
    exit 1
fi
