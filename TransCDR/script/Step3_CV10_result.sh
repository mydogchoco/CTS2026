#!/bin/bash
set -euo pipefail

CV5_RESULT_PATH='./result/CV5'
OUTPUT=''

while [[ $# -gt 0 ]]; do
    case "$1" in
        --CV5_result_path) CV5_RESULT_PATH="$2"; shift 2 ;;
        --output)          OUTPUT="$2"; shift 2 ;;
        *) echo "[Step3] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${OUTPUT}" ]]; then
    echo "[Step3] ERROR: --output required" >&2
    exit 1
fi

python -u Step3_result.py \
    --CV5_result_path "${CV5_RESULT_PATH}" \
    --output "${OUTPUT}"