#!/bin/bash

# Script to run all experiments for repetitions 1-5 in the background
# Each experiment must finish before the next one starts
# Each repetition must finish before the next repetition starts

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

EXPERIMENTS=("cox" "deepsurv" "dynamic_deephit" "gbsa" "hazard_transformer" "rnnsurv" "weibul")

main_execution() {

update_rep_in_commons() {
    local rep_num=$1
    
    echo "Updating commons.py for rep${rep_num}..."
    
    if python "${SCRIPT_DIR}/update_rep.py" "$rep_num"; then
        echo "Successfully updated commons.py to use rep${rep_num}"
        return 0
    else
        echo "Failed to update commons.py"
        return 1
    fi
}

run_experiment() {
    local experiment=$1
    local rep_num=$2
    local log_file=$3
    
    echo "==================== Running $experiment for rep${rep_num} ====================" | tee -a "$log_file"
    echo "Start time: $(date)" | tee -a "$log_file"
    
    if python -m "pkgs.experiments.${experiment}" >> "$log_file" 2>&1; then
        echo "✓ $experiment completed successfully" | tee -a "$log_file"
        echo "End time: $(date)" | tee -a "$log_file"
        return 0
    else
        echo "✗ $experiment failed with exit code $?" | tee -a "$log_file"
        echo "End time: $(date)" | tee -a "$log_file"
        return 1
    fi
}

run_rep() {
    local rep_num=$1
    local log_file="${SCRIPT_DIR}/eval_all_rep${rep_num}.log"
    
    echo "Starting repetition ${rep_num}..." | tee "$log_file"
    echo "Log file: $log_file" | tee -a "$log_file"
    echo "Start time: $(date)" | tee -a "$log_file"
    echo "=========================================" | tee -a "$log_file"
    
    update_rep_in_commons "$rep_num"
    
    mkdir -p "${PROJECT_ROOT}/generated_data/rep${rep_num}"
    
    local failed_experiments=()
    
    for experiment in "${EXPERIMENTS[@]}"; do
        if ! run_experiment "$experiment" "$rep_num" "$log_file"; then
            failed_experiments+=("$experiment")
        fi
        echo "" | tee -a "$log_file"
    done
    
    echo "=========================================" | tee -a "$log_file"
    echo "Repetition ${rep_num} Summary:" | tee -a "$log_file"
    echo "Total experiments: ${#EXPERIMENTS[@]}" | tee -a "$log_file"
    echo "Failed experiments: ${#failed_experiments[@]}" | tee -a "$log_file"
    
    if [ ${#failed_experiments[@]} -eq 0 ]; then
        echo "✓ All experiments completed successfully!" | tee -a "$log_file"
    else
        echo "✗ Failed experiments: ${failed_experiments[*]}" | tee -a "$log_file"
    fi
    
    echo "End time: $(date)" | tee -a "$log_file"
    echo "=========================================" | tee -a "$log_file"
    
    return ${#failed_experiments[@]}
}

echo "Starting all repetitions (rep1 to rep5)..."
echo "Project root: $PROJECT_ROOT"
echo "Experiments to run: ${EXPERIMENTS[*]}"
echo ""

original_rep=$(grep "generate_data_path_latest_rep" "${PROJECT_ROOT}/pkgs/commons.py" | sed -n "s/.*rep\([0-9]*\).*/\1/p")
echo "Original rep setting: rep${original_rep}"

failed_reps=()

for rep in {1..1}; do
    echo "Starting rep${rep}..."
    if run_rep "$rep"; then
        echo "✓ Rep${rep} completed successfully"
    else
        echo "✗ Rep${rep} had failures"
        failed_reps+=("$rep")
    fi
    echo ""
done

echo "========================================="
echo "FINAL SUMMARY"
echo "========================================="
echo "Repetitions run: 1, 2, 3, 4, 5"
echo "Total repetitions: 5"
echo "Failed repetitions: ${#failed_reps[@]}"

if [ ${#failed_reps[@]} -eq 0 ]; then
    echo "✓ All repetitions completed successfully!"
else
    echo "✗ Failed repetitions: ${failed_reps[*]}"
fi

echo ""
echo "Log files generated:"
for rep in {1..1}; do
    echo "  - ${SCRIPT_DIR}/eval_all_rep${rep}.log"
done

if [ -n "$original_rep" ]; then
    echo ""
    echo "Restoring original rep setting (rep${original_rep})..."
    update_rep_in_commons "$original_rep"
fi

echo ""
echo "All done! Check the individual log files for detailed results."
}

if [ "$1" = "--background" ] || [ "$1" = "-b" ]; then
    echo "Starting all experiments in background..."
    echo "Project root: $PROJECT_ROOT"
    echo "Output will be logged to: ${SCRIPT_DIR}/run_all_reps_master.log"
    
    main_execution > "${SCRIPT_DIR}/run_all_reps_master.log" 2>&1 &
    
    echo $! > "${SCRIPT_DIR}/run_all_reps.pid"
    
    echo "Background process started with PID: $(cat "${SCRIPT_DIR}/run_all_reps.pid")"
    echo "Monitor progress with: tail -f ${SCRIPT_DIR}/run_all_reps_master.log"
    echo "Stop process with: kill $(cat "${SCRIPT_DIR}/run_all_reps.pid")"
    echo ""
    echo "Individual rep logs will also be generated:"
    for rep in {1..1}; do
        echo "  - ${SCRIPT_DIR}/eval_all_rep${rep}.log"
    done
else
    main_execution
fi
