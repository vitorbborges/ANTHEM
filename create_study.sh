#!/usr/bin/env bash
# create_study.sh - (Re)create the Optuna study in MySQL

set -euo pipefail

# Replace these with your actual MySQL credentials
DB_USER="optuna_user"
DB_PASS="anthem1234"
DB_NAME="optuna_db"
STUDY_NAME="co2_prediction"
STORAGE_URL="mysql://$DB_USER:$DB_PASS@localhost/$DB_NAME"

# Ensure Optuna CLI and MySQL driver are installed
echo "📦 Installing/upgrading required packages..."
pip install --upgrade optuna PyMySQL

echo "🔄 Deleting existing study '$STUDY_NAME' (if any)..."
optuna delete-study \
    --storage "$STORAGE_URL" \
    --study-name "$STUDY_NAME" \
    || true # ignore "not found" errors

echo "✨ Creating fresh study '$STUDY_NAME'..."
optuna create-study \
    --storage "$STORAGE_URL" \
    --study-name "$STUDY_NAME" \
    --direction minimize

echo "✅ Study '$STUDY_NAME' is ready with zero trials."
echo "🌐 You can view it at: optuna-dashboard $STORAGE_URL"

# Test the connection
echo "🧪 Testing study access..."
optuna studies --storage "$STORAGE_URL"

echo "🚀 Ready to run optimization with:"
echo "   python -m src.modeling.main --storage mysql --mysql-url \"$STORAGE_URL\""
