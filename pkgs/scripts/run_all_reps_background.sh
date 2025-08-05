#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"


echo "Starting all experiments in background..."
echo "Script directory: $SCRIPT_DIR"

nohup "${SCRIPT_DIR}/run_all_reps.sh" > "${SCRIPT_DIR}/run_all_reps_master.log" 2>&1 &

echo $! > "${SCRIPT_DIR}/run_all_reps.pid"

echo "Background process started with PID: $(cat "${SCRIPT_DIR}/run_all_reps.pid")"
echo "Master log: ${SCRIPT_DIR}/run_all_reps_master.log"
echo ""
echo "Monitor progress with:"
echo "  tail -f ${SCRIPT_DIR}/run_all_reps_master.log"
echo ""
echo "Stop process with:"
echo "  kill \$(cat ${SCRIPT_DIR}/run_all_reps.pid)"
echo ""
echo "Individual rep logs will also be generated:"
for rep in {1..5}; do
    echo "  - ${SCRIPT_DIR}/eval_all_rep${rep}.log"
done
echo ""
echo "Check if process is still running:"
echo "  ps -p \$(cat ${SCRIPT_DIR}/run_all_reps.pid)"
