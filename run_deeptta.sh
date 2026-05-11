#!/bin/bash
# =============================================================================
# run_deeptta.sh - Unified entry point for DeepTTA model
#
# Usage:
#   bash run_deeptta.sh --split_file <path> [--mock]
#
# Modes:
#   --mock : run mock_run.py (no actual training, uses cached metrics)
#   (no flag): run real training pipeline
#                -> NOT YET IMPLEMENTED. Waiting for external split index.
#                   Will be filled in after Tuesday's index delivery.
#
# Examples:
#   bash run_deeptta.sh --split_file /tmp/fake.npy --mock
#   bash run_deeptta.sh --split_file /home/intern1_2026_1/Common/Input/SplitIndex/fold1.npy
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
COMMON_ROOT="/home/intern1_2026_1/Common"
CONDA_SH="${COMMON_ROOT}/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="deeptta"

CTS_ROOT="${COMMON_ROOT}/CTS2026"
MODEL_DIR="${CTS_ROOT}/DeepTTA"
MOCK_RUNNER="${CTS_ROOT}/mock_run.py"

MODEL_NAME="DeepTTA"

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
SPLIT_FILE=""
MOCK=false

usage() {
    cat <<EOF >&2
Usage: bash run_deeptta.sh --split_file <path> [--mock]

Required:
  --split_file <path>   Path to .npy file with train/val/test indices

Optional:
  --mock                Run mock pipeline (no training, cached metrics)
  -h, --help            Show this help message
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split_file)
            SPLIT_FILE="$2"
            shift 2
            ;;
        --mock)
            MOCK=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "[run_deeptta] Unknown argument: $1" >&2
            usage
            ;;
    esac
done

if [[ -z "${SPLIT_FILE}" ]]; then
    echo "[run_deeptta] ERROR: --split_file is required" >&2
    usage
fi

# -----------------------------------------------------------------------------
# Conda activation
# -----------------------------------------------------------------------------
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[run_deeptta] ERROR: conda.sh not found at ${CONDA_SH}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
if [[ "${MOCK}" == true ]]; then
    echo "[run_deeptta] Mode: MOCK"
    echo "[run_deeptta] Model: ${MODEL_NAME}"
    echo "[run_deeptta] Split file: ${SPLIT_FILE}"

    python -u "${MOCK_RUNNER}" \
        --model "${MODEL_NAME}" \
        --split_file "${SPLIT_FILE}"
else
    echo "[run_deeptta] Mode: REAL TRAINING" >&2
    echo "[run_deeptta] ERROR: Real training mode is not yet implemented." >&2
    echo "[run_deeptta]" >&2
    echo "[run_deeptta] TODO (after external index delivery on Tuesday):" >&2
    echo "[run_deeptta]   1. Add ByExternalIndex() to Step1_getData.py" >&2
    echo "[run_deeptta]   2. Add --split_file argparse to Step3_model.py" >&2
    echo "[run_deeptta]   3. Fix val=test leak (val_drug=valdata in net.train)" >&2
    echo "[run_deeptta]   4. Replace the block below with:" >&2
    echo "[run_deeptta]        cd \"\${MODEL_DIR}\"" >&2
    echo "[run_deeptta]        python -u Step3_model.py --split_file \"\${SPLIT_FILE}\"" >&2
    echo "[run_deeptta]" >&2
    echo "[run_deeptta] Hint: use --mock to test the runner interface in the meantime." >&2
    exit 2
fi
