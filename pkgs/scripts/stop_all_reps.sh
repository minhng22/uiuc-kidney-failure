#!/bin/bash
cd "$(dirname "$0")"

if ! [ -f run_all_reps.pid ]; then
  echo "ERROR: run_all_reps.pid not found!" >&2
  exit 1
fi

PID=$(cat run_all_reps.pid)
PGID=$(ps -o pgid= $PID | tr -d ' ')

echo "Killing process group $PGID (leader $PID)…"
kill -TERM -"$PGID"
sleep 10

echo "Recursively killing any stragglers…"
killtree() {
  for c in $(pgrep -P $1); do killtree $c; done
  kill -TERM $1 2>/dev/null
}
killtree $PID

echo "Done. Verify with: ps -o pid,ppid,pgid,cmd --forest | grep $PID"