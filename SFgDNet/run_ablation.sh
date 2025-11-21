#!/usr/bin/env bash
set -u  # keep -u, but drop -e so we can continue on failures
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python '${PYTHON_BIN}' not found" >&2; exit 1; }

# Run across all datasets
DATASETS=("assembly")

# Use all known assembly action filters
ASSEMBLY_ACTION_FILTERS=("pick" "put" "remove" "unscrew" "rotate" "attempt")

# exact pairs requested
WINDOWS=(70)
HORIZONS=(30)

# if 50-50 should run, allow horizon == window
ALLOW_EQUAL_WINDOW_HORIZON=true

ABLATION_SCRIPT="${SCRIPT_DIR}/ablation_architecture_simlpe.py"

for dataset in "${DATASETS[@]}"; do
  for i in "${!WINDOWS[@]}"; do
    window="${WINDOWS[$i]}"
    horizon="${HORIZONS[$i]}"

    if [[ "${ALLOW_EQUAL_WINDOW_HORIZON}" != "true" && $horizon -ge $window ]]; then
      echo "Skipping horizon ${horizon} >= window ${window}" >&2
      continue
    fi

    if [[ "${dataset}" == "assembly" ]]; then
      for action_filter in "${ASSEMBLY_ACTION_FILTERS[@]}"; do
        echo "[${dataset}] action=${action_filter} window=${window} horizon=${horizon}"
        if ! "${PYTHON_BIN}" "${ABLATION_SCRIPT}" \
              --dataset "${dataset}" \
              --action-filter "${action_filter}" \
              --window "${window}" \
              --horizon "${horizon}" \
              --scenarios "component_transmlp_mixed_layers" \
              "$@"; then
          echo "Run FAILED: dataset=${dataset} action=${action_filter} window=${window} horizon=${horizon}" >&2
          # continue to next experiment
        fi
      done
    else
      echo "[${dataset}] window=${window} horizon=${horizon}"
      if ! "${PYTHON_BIN}" "${ABLATION_SCRIPT}" \
            --dataset "${dataset}" \
            --window "${window}" \
            --horizon "${horizon}" \
            "$@"; then
        echo "Run FAILED: dataset=${dataset} window=${window} horizon=${horizon}" >&2
        # continue to next experiment
      fi
    fi
  done
done
