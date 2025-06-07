#!/bin/bash
set -euo pipefail

# This script is the entry point for the reimbursement calculation.
# It validates the inputs and then calls the python calculation script.

# Check for the correct number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <trip_duration_days> <miles_traveled> <total_receipts_amount>"
    exit 1
fi

# Assign arguments to variables for clarity
TRIP_DURATION_DAYS=$1
MILES_TRAVELED=$2
TOTAL_RECEIPTS_AMOUNT=$3

# Get the directory of the script to robustly locate other files
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Execute the python script with the project's virtual environment interpreter
PYTHON_EXEC="$SCRIPT_DIR/.venv/bin/python"
CALC_SCRIPT="$SCRIPT_DIR/calculate.py"

"$PYTHON_EXEC" "$CALC_SCRIPT" "$TRIP_DURATION_DAYS" "$MILES_TRAVELED" "$TOTAL_RECEIPTS_AMOUNT" 