#!/bin/bash
set -euo pipefail

# <exp_id>_index.npy compat: external scenario now requires --split and
# --exp_id (the underlying Python resolves the npy file path internally).
SCENARIOS="warm start"
SPLIT=""
EXP_ID=""
RESULT_FOLDER="./data/GDSC"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenarios)     SCENARIOS="$2"; shift 2 ;;
        --split)         SPLIT="$2"; shift 2 ;;
        --exp_id)        EXP_ID="$2"; shift 2 ;;
        --result_folder) RESULT_FOLDER="$2"; shift 2 ;;
        *) echo "[Step1] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

EXTRA_ARGS=()
if [[ "${SCENARIOS}" == "external" ]]; then
    if [[ -z "${SPLIT}" || -z "${EXP_ID}" ]]; then
        echo "[Step1] ERROR: --split and --exp_id required for external scenario" >&2
        exit 1
    fi
    EXTRA_ARGS+=(--split "${SPLIT}" --exp_id "${EXP_ID}")
fi

python -u Step1_Data_split.py \
    --model_type "regression" \
    --scenarios "${SCENARIOS}" \
    --n_clusters 0 \
    --n_sampling 0 \
    --result_folder "${RESULT_FOLDER}" \
    "${EXTRA_ARGS[@]}"
