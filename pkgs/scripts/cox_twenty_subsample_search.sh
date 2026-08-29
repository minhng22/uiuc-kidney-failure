#!/bin/bash
# Runs one cox_twenty_subsample_trial.py trial for N patients, polling its
# RSS every 15s. Kills it early (recorded as a "too-big" failure, faster
# than waiting for a real kernel OOM-kill) if RSS crosses SAFETY_CAP_KB, or
# if it runs past MAX_WALL_SECS. Logs to
# pkgs/scripts/cox_subsample_trial_<n>.log. Exit 0 = trial succeeded
# within the caps, exit 1 = failed/killed (see log for which).
#
# Usage: pkgs/scripts/cox_twenty_subsample_search.sh <n_patients> [seed]
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
N="$1"
SEED="${2:-42}"
LOG="${SCRIPT_DIR}/cox_subsample_trial_${N}.log"
SAFETY_CAP_KB=$((120 * 1024 * 1024))   # 120GB -- isolated-trial safety cap, well below this box's ~188GB
MAX_WALL_SECS=$((150 * 60))            # 150 min wall-clock cap per trial (raised: memory has stayed well-bounded through N=8000, time is now the more relevant limit to watch)

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
export CKD_REP=1

echo "Trial start: $(date), n_patients=$N, seed=$SEED, safety_cap=${SAFETY_CAP_KB}KB, max_wall=${MAX_WALL_SECS}s" > "$LOG"
python -m pkgs.scripts.cox_twenty_subsample_trial "$N" "$SEED" >> "$LOG" 2>&1 &
PYPID=$!

start_ts=$(date +%s)
result="unknown"
while kill -0 "$PYPID" 2>/dev/null; do
    sleep 15
    now_ts=$(date +%s)
    elapsed=$((now_ts - start_ts))
    rss_kb=$(ps -o rss= -p "$PYPID" 2>/dev/null | tr -d ' ')
    if [ -z "$rss_kb" ]; then
        break  # process already gone
    fi
    echo "  [poll] elapsed=${elapsed}s rss=${rss_kb}KB ($((rss_kb / 1024 / 1024))GB)" >> "$LOG"
    if [ "$rss_kb" -gt "$SAFETY_CAP_KB" ]; then
        echo "KILLED: RSS ${rss_kb}KB exceeded safety cap ${SAFETY_CAP_KB}KB at ${elapsed}s" >> "$LOG"
        kill -KILL "$PYPID" 2>/dev/null
        result="too_big"
        break
    fi
    if [ "$elapsed" -gt "$MAX_WALL_SECS" ]; then
        echo "KILLED: wall time ${elapsed}s exceeded cap ${MAX_WALL_SECS}s" >> "$LOG"
        kill -KILL "$PYPID" 2>/dev/null
        result="too_slow"
        break
    fi
done

if [ "$result" = "unknown" ]; then
    wait "$PYPID"
    exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        result="success"
    else
        result="failed_exit_${exit_code}"
    fi
fi

echo "Trial end: $(date), result=$result" >> "$LOG"
echo "$result" > "${SCRIPT_DIR}/cox_subsample_trial_${N}.result"
