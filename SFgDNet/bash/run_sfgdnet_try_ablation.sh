#!/usr/bin/env bash
# Launch the sfgdnet_try architecture ablations for all standard Assembly actions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
ABLATION_SCRIPT="${SCRIPT_DIR}/ablation_architecture_simlpe.py"

DATASET="${DATASET:-assembly}"
WINDOW="${WINDOW:-70}"
HORIZON="${HORIZON:-30}"

# Default Assembly actions used across prior scripts.
ASSEMBLY_ACTION_FILTERS=("pick" "put" "remove" "unscrew" "rotate" "attempt")

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python binary '${PYTHON_BIN}' not found" >&2
  exit 1
fi

for action_filter in "${ASSEMBLY_ACTION_FILTERS[@]}"; do
  echo "[${DATASET}] ablation | action=${action_filter} window=${WINDOW} horizon=${HORIZON}"
  if ! "${PYTHON_BIN}" "${ABLATION_SCRIPT}" \
        --dataset "${DATASET}" \
        --action-filter "${action_filter}" \
        --window "${WINDOW}" \
        --horizon "${HORIZON}" \
        "$@"; then
    echo "Run FAILED: dataset=${DATASET} action=${action_filter} window=${WINDOW} horizon=${HORIZON}" >&2
  fi
done
