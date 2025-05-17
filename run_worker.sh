#!/bin/bash
# Individual worker script for advanced optimization

# Get worker ID from command line argument
WORKER_ID=$1
if [ -z "$WORKER_ID" ]; then
    echo "Error: Worker ID not provided"
    exit 1
fi

# Set environment variables
export WORKER_ID=$WORKER_ID
export OPTUNA_STORAGE="mysql://optuna_user:anthem1234@localhost/optuna_db"

# Set worker-specific timeouts
TIMEOUT_SECONDS=$((MAX_TIME_MINUTES * 60))

echo "Starting worker $WORKER_ID with approach: $APPROACH, timeout: $TIMEOUT_SECONDS seconds"

# Run the optimization script with a timeout
timeout $TIMEOUT_SECONDS python -u src/modeling/distributed_optimization.py

exit_code=$?
if [ $exit_code -eq 124 ]; then
    echo "Worker $WORKER_ID timed out after $MAX_TIME_MINUTES minutes"
fi

echo "Worker $WORKER_ID completed with exit code $exit_code"
