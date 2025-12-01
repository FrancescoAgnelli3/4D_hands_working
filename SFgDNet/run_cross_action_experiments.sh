#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python '${PYTHON_BIN}' not found" >&2; exit 1; }

DATASETS=("assembly")
# add other datasets here if cross-action coverage is needed
ACTIONS=("pick" "put" "remove" "unscrew" "rotate" "attempt")
SAMPLES_PER_ACTION=300
ALLOW_EQUAL_WINDOW_HORIZON=true
SEED=0
declare -A seen_combinations=()

WINDOWS=(51 70 90)
HORIZONS=(49 30 10)

for dataset in "${DATASETS[@]}"; do
  action_list=$(IFS=,; echo "${ACTIONS[*]}")
  for i in "${!WINDOWS[@]}"; do
    window="${WINDOWS[$i]}"
    horizon="${HORIZONS[$i]}"

    if [[ "${ALLOW_EQUAL_WINDOW_HORIZON}" != "true" && $horizon -ge $window ]]; then
      echo "Skipping horizon ${horizon} >= window ${window}" >&2
      continue
    fi

    for withheld in "${ACTIONS[@]}"; do
      key="${dataset}:${withheld}:${window}:${horizon}"
      if [[ -n "${seen_combinations[$key]:-}" ]]; then
        echo "Skipping redundant run for ${key}" >&2
        continue
      fi
      seen_combinations[$key]=1
      echo "[${dataset}] test=${withheld} window=${window} horizon=${horizon}"
      results_csv_base="out/best_runs/results_cross_action_${withheld}.csv"
      if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/train_cross_action_models.py" \
            --dataset "${dataset}" \
            --actions "${action_list}" \
            --test-action "${withheld}" \
            --results-csv "${results_csv_base}" \
            --samples-per-action "${SAMPLES_PER_ACTION}" \
            --seed "${SEED}" \
            --window "${window}" \
            --horizon "${horizon}" \
            "$@"; then
        echo "Run FAILED: dataset=${dataset} test=${withheld} window=${window} horizon=${horizon}" >&2
      fi
    done
  done
done
