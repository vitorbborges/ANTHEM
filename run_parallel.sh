#!/usr/bin/env bash
# run_parallel_extreme.sh - Script for extreme optimization
set -euo pipefail
IFS=$'\n\t'

# Number of workers to run (overrideable via env or cmdline)
NUM_WORKERS=${NUM_WORKERS:-10}

# Paths
METRICS_DIR="metrics"
JOBLOG="${METRICS_DIR}/extreme_parallel_joblog.txt"
SUMMARY_LOG="${METRICS_DIR}/extreme_run_summary_$(date +%Y%m%d_%H%M%S).txt"

# Ensure dependencies
command -v parallel >/dev/null 2>&1 \
  || { echo >&2 "Error: GNU parallel is not installed."; exit 1; }

# Prepare environment
chmod +x run_worker.sh
mkdir -p "$METRICS_DIR"

echo "Cleaning up old log files..."
# delete logs older than 3 days
find "$METRICS_DIR" -name "optimization_log_*" -type f -mtime +3 -print -delete

# Initialize summary
{
  echo "Starting extreme parallel run with $NUM_WORKERS workers at $(date)"
  echo "Using reduced feature set and data sampling for faster execution"
} > "$SUMMARY_LOG"

echo "Launching $NUM_WORKERS worker processes with extreme optimization..."
parallel --progress --joblog "$JOBLOG" \
  ./run_worker.sh {1} ::: $(seq 1 "$NUM_WORKERS")

# Summarize results
{
  echo "All worker processes completed at $(date)"
  echo "Finding best overall result..."
} >> "$SUMMARY_LOG"

BEST_MSE=9999999
BEST_FILE=""

for f in "$METRICS_DIR"/best_params_worker_*.json; do
  [[ -f "$f" ]] || continue

  # extract metrics
  MSE=$(grep -oP '"best_test_mse"\s*:\s*\K[0-9.]+(?=,?)' "$f" || echo "")
  R2=$(grep -oP '"best_test_r2"\s*:\s*\K[0-9.]+(?=,?)' "$f" || echo "")

  if [[ -n "$MSE" && -n "$R2" ]]; then
    echo "File $f: MSE=$MSE, R²=$R2" >> "$SUMMARY_LOG"
    # compare and update if better
    if awk "BEGIN {exit !($MSE < $BEST_MSE)}"; then
      BEST_MSE=$MSE
      BEST_FILE=$f
    fi
  fi
done

if [[ -n "${BEST_FILE:-}" ]]; then
  cp -- "$BEST_FILE" "${METRICS_DIR}/best_params_extreme_overall.json"
  {
    echo "Best overall result: $BEST_FILE with MSE=$BEST_MSE"
    echo "Best parameters copied to ${METRICS_DIR}/best_params_extreme_overall.json"
    echo
    echo "Best model configuration:"
    grep -E "feature_step|final_model" "$BEST_FILE"
  } >> "$SUMMARY_LOG"
else
  echo "No valid result files found." >> "$SUMMARY_LOG"
fi

echo "Run completed. See $SUMMARY_LOG for summary."
