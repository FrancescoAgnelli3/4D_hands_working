#!/usr/bin/env bash
set -u  # keep strict undefined-variable checking
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python '${PYTHON_BIN}' not found" >&2; exit 1; }

DATASETS=("assembly")
TIME_SPLITS=("70 : 30")
ALLOW_EQUAL_WINDOW_HORIZON=false

ASSEMBLY_ACTION_FILTERS=("pick" "put" "remove" "unscrew" "rotate" "attempt")
ABLATION_SCRIPT="${SCRIPT_DIR}/ablation_study.py"
MODEL="sfgdnet_try"

for dataset in "${DATASETS[@]}"; do
  for split in "${TIME_SPLITS[@]}"; do
    window="${split%%:*}"
    horizon="${split##*:}"

    if [[ -z "${window}" || -z "${horizon}" ]]; then
      echo "Skipping malformed split '${split}'" >&2
      continue
    fi

    if [[ "${ALLOW_EQUAL_WINDOW_HORIZON}" != "true" && $horizon -ge $window ]]; then
      echo "Skipping horizon ${horizon} >= window ${window}" >&2
      continue
    fi

    if [[ "${dataset}" == "assembly" ]]; then
      for action_filter in "${ASSEMBLY_ACTION_FILTERS[@]}"; do
        echo "[${dataset}] action=${action_filter} input=${window} output=${horizon}"
        if ! "${PYTHON_BIN}" "${ABLATION_SCRIPT}" \
              --dataset "${dataset}" \
              --action-filter "${action_filter}" \
              --input-n "${window}" \
              --output-n "${horizon}" \
              --model "${MODEL}" \
              --preset "basic" \
              "$@"; then
          echo "Run FAILED: dataset=${dataset} action=${action_filter} input=${window} output=${horizon}" >&2
        fi
      done
    else
      echo "[${dataset}] input=${window} output=${horizon}"
      if ! "${PYTHON_BIN}" "${ABLATION_SCRIPT}" \
            --dataset "${dataset}" \
            --input-n "${window}" \
            --output-n "${horizon}" \
            --model "${MODEL}" \
            --preset full \
            "$@"; then
        echo "Run FAILED: dataset=${dataset} input=${window} output=${horizon}" >&2
      fi
    fi
  done
done
