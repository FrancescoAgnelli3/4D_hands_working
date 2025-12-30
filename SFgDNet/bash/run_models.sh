#!/usr/bin/env bash
set -u  # keep -u, but drop -e so we can continue on failures
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python '${PYTHON_BIN}' not found" >&2; exit 1; }

# Run across all datasets
DATASETS=("assembly" "fpha" "h2o" "bighands")
DATASETS=("h2o" "bighands")

# Use all known assembly action filters
ASSEMBLY_ACTION_FILTERS=("pick" "put" "remove" "unscrew" "rotate" "attempt")

# exact pairs requested
WINDOWS=(51 60 70 80 90)
HORIZONS=(49 40 30 20 10)
# WINDOWS=(70)
# HORIZONS=(30)

# if 50-50 should run, allow horizon == window
ALLOW_EQUAL_WINDOW_HORIZON=true

# DEFAULT_MODEL_LIST="last,hisrep,simlpe_dct,SFgDNet,sfgdnet_try,mpt,estag,hstgcnn,trajdep,pgbig,motionmixer,eqmotion"
DEFAULT_MODEL_LIST="sfgdnet_try"
IFS=',' read -r -a MODELS <<< "${DEFAULT_MODEL_LIST}"

for dataset in "${DATASETS[@]}"; do
  for i in "${!WINDOWS[@]}"; do
    window="${WINDOWS[$i]}"
    horizon="${HORIZONS[$i]}"

    if [[ "${dataset}" == "assembly" ]] && { { [[ "${window}" -eq 60 && "${horizon}" -eq 40 ]]; } || { [[ "${window}" -eq 80 && "${horizon}" -eq 20 ]]; }; }; then
      echo "Skipping assembly run with window=${window} horizon=${horizon}"
      continue
    fi

    if [[ "${ALLOW_EQUAL_WINDOW_HORIZON}" != "true" && $horizon -ge $window ]]; then
      echo "Skipping horizon ${horizon} >= window ${window}" >&2
      continue
    fi

    for model in "${MODELS[@]}"; do
      if [[ "${dataset}" == "assembly" ]]; then
        for action_filter in "${ASSEMBLY_ACTION_FILTERS[@]}"; do
          echo "[${dataset}] model=${model} action=${action_filter} window=${window} horizon=${horizon}"
          if ! "${PYTHON_BIN}" "${REPO_ROOT}/train_best_models.py" \
                --dataset "${dataset}" \
                --action-filter "${action_filter}" \
                --model "${model}" \
                --window "${window}" \
                --horizon "${horizon}" \
                "$@"; then
            echo "Run FAILED: dataset=${dataset} model=${model} action=${action_filter} window=${window} horizon=${horizon}" >&2
            # continue to next experiment
          fi
        done
      else
        echo "[${dataset}] model=${model} window=${window} horizon=${horizon}"
        if ! "${PYTHON_BIN}" "${REPO_ROOT}/train_best_models.py" \
              --dataset "${dataset}" \
              --model "${model}" \
              --window "${window}" \
              --horizon "${horizon}" \
              "$@"; then
          echo "Run FAILED: dataset=${dataset} model=${model} window=${window} horizon=${horizon}" >&2
          # continue to next experiment
        fi
      fi
    done
  done
done