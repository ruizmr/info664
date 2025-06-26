#!/bin/bash
# run_model.sh – evaluate using the surrogate ML model (calculate.py + model_state.pkl)
# Usage: ./run_model.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <trip_duration_days> <miles_traveled> <total_receipts_amount>" >&2
  exit 1
fi
python - "$@" <<'PY'
import sys, joblib
import calculate

model = joblib.load('model_state.pkl')

days = float(sys.argv[1])
miles = float(sys.argv[2])
receipts = float(sys.argv[3])
print(f"{calculate.calculate_reimbursement(days, miles, receipts, model):.2f}")
PY 