#!/usr/bin/env bash
set -euo pipefail

# create_study.sh - (Re)create the Optuna study in MySQL

# Replace these with your actual MySQL credentials
DB_USER="optuna_user"
DB_PASS="anthem1234"
DB_NAME="optuna_db"
STUDY_NAME="co2_prediction"

# Ensure Optuna CLI and MySQL driver are installed
pip install --upgrade optuna mysqlclient

echo "🔄 Deleting existing study '$STUDY_NAME' (if any)…"
optuna delete-study \
  --study-name "$STUDY_NAME" \
  --storage "mysql://$DB_USER:$DB_PASS@localhost/$DB_NAME" \
  || true    # ignore “not found” errors

echo "✨ Creating fresh study '$STUDY_NAME'…"
optuna create-study \
  --study-name "$STUDY_NAME" \
  --storage "mysql://$DB_USER:$DB_PASS@localhost/$DB_NAME" \
  --direction minimize

echo "✅ Study '$STUDY_NAME' is ready with zero trials."
