#!/bin/bash
# =============================================================================
# run_deeptta.sh - Unified entry point for DeepTTA model
#
# Usage:
#   bash run_deeptta.sh --split_file <path> [--mock]
#
# Modes:
#   --mock      : run mock_run.py (no actual training, uses cached metrics)
#   (no flag)   : run real training pipeline
#
# Scenario is auto-inferred from the split_file basename:
#   mix_index.npy        -> mix
#   cellblind_index.npy  -> cellblind
#   drugblind_index.npy  -> drugblind
#   disjoint_index.npy   -> disjoint
#
# Examples:
#   bash run_deeptta.sh --split_file /tmp/fake.npy --mock
#   bash run_deeptta.sh --split_file /home/intern1_2026_1/Common/Input/SplitIndex/mix_index.npy
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
COMMON_ROOT="/home/intern1_2026_1/Common"
CONDA_SH="${COMMON_ROOT}/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="deeptta"

CTS_ROOT="${COMMON_ROOT}/CTS2026"
CODE_DIR="${CTS_ROOT}/DeepTTA"
MOCK_RUNNER="${CTS_ROOT}/mock_run.py"

OUTPUT_ROOT="${COMMON_ROOT}/Output/DeepTTA"

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

# Mock mode skips file existence check (lets us test with fake paths)
if [[ "${MOCK}" != true && ! -f "${SPLIT_FILE}" ]]; then
    echo "[run_deeptta] ERROR: split_file not found: ${SPLIT_FILE}" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Scenario inference from split_file basename
#   mix_index.npy -> mix, cellblind_index.npy -> cellblind, etc.
# -----------------------------------------------------------------------------
SPLIT_BASE="$(basename "${SPLIT_FILE}")"
SCENARIO="${SPLIT_BASE%_index.npy}"
if [[ "${SCENARIO}" == "${SPLIT_BASE}" ]]; then
    # filename didn't match *_index.npy pattern; fall back to literal basename
    SCENARIO="${SPLIT_BASE%.npy}"
fi
echo "[run_deeptta] Inferred scenario: ${SCENARIO}"

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
    echo "[run_deeptta] Mode: REAL TRAINING"
    echo "[run_deeptta] Model: ${MODEL_NAME}"
    echo "[run_deeptta] Split file: ${SPLIT_FILE}"
    echo "[run_deeptta] Scenario: ${SCENARIO}"

    MODELDIR="${CODE_DIR}/Model_unif/${SCENARIO}"
    OUTPUT="${OUTPUT_ROOT}/${SCENARIO}_metrics.csv"

    mkdir -p "${MODELDIR}" "$(dirname "${OUTPUT}")"

    cd "${CODE_DIR}"
    python -u Step3_model.py \
        --split_file "${SPLIT_FILE}" \
        --modeldir   "${MODELDIR}" \
        --output     "${OUTPUT}"

    echo "[run_deeptta] Done. Metrics: ${OUTPUT}"
fi