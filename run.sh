#!/bin/bash
set -euo pipefail

# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <trip_duration_days> <miles_traveled> <total_receipts_amount>" >&2
  exit 1
fi

python - "$@" <<'PY'
import sys
from legacy_reimburse import legacy_reimburse as lr

days = float(sys.argv[1])
miles = float(sys.argv[2])
receipts = float(sys.argv[3])
print(f"{lr(days, miles, receipts):.2f}")
PY