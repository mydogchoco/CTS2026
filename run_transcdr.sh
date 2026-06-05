#!/bin/bash
# =============================================================================
# run_transcdr.sh - Unified entry point for TransCDR model
#
# Usage:
#   bash run_transcdr.sh --split <scenario> --exp_id <id> [--gpu_id N] [--mock]
#
# Required:
#   --split <scenario>   mix | cellblind | drugblind | disjoint
#   --exp_id <id>        experiment identifier; resolves split file at
#                        ${COMMON_ROOT}/Input/SplitIndex/<exp_id>_index.npy
#
# Optional:
#   --gpu_id N           GPU device id (default: 0)
#   --mock               Run mock pipeline (no training, cached metrics)
#
# Output:
#   ${COMMON_ROOT}/Output/<exp_id>/TransCDR/<scenario>/metrics.csv
#
# Examples:
#   bash run_transcdr.sh --split mix --exp_id original
#   bash run_transcdr.sh --split mix --exp_id original --gpu_id 1
#   bash run_transcdr.sh --split mix --exp_id original --mock
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
COMMON_ROOT="/home/intern1_2026_1/Common"
CONDA_SH="${COMMON_ROOT}/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="deeptta"

CTS_ROOT="${COMMON_ROOT}/CTS2026"
MODEL_DIR="${CTS_ROOT}/TransCDR"
SCRIPT_DIR="${MODEL_DIR}/script"
MOCK_RUNNER="${CTS_ROOT}/mock_run.py"

MODEL_NAME="TransCDR"

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
SPLIT=""
EXP_ID=""
MOCK=false
GPU_ID=0

usage() {
    cat <<EOF >&2
Usage: bash run_transcdr.sh --split <scenario> --exp_id <id> [--gpu_id N] [--mock]

Required:
  --split <scenario>   mix | cellblind | drugblind | disjoint
  --exp_id <id>        experiment identifier (resolves split file path)

Optional:
  --gpu_id N           GPU device id (default: 0)
  --mock               Run mock pipeline (no training, cached metrics)
  -h, --help           Show this help message
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --exp_id)
            EXP_ID="$2"
            shift 2
            ;;
        --gpu_id)
            GPU_ID="$2"
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
            echo "[run_transcdr] Unknown argument: $1" >&2
            usage
            ;;
    esac
done

# Both --split and --exp_id are required (registry contract)
if [[ -z "${SPLIT}" || -z "${EXP_ID}" ]]; then
    echo "[run_transcdr] ERROR: both --split and --exp_id are required" >&2
    usage
fi

# <exp_id>_index.npy compat: resolve npy path from exp_id
SPLIT_FILE="${COMMON_ROOT}/Input/SplitIndex/${EXP_ID}_index.npy"

# Mock mode skips file existence check (lets registry smoke-test with fake exp_id)
if [[ "${MOCK}" != true && ! -f "${SPLIT_FILE}" ]]; then
    echo "[run_transcdr] ERROR: split_file not found: ${SPLIT_FILE}" >&2
    exit 1
fi

echo "[run_transcdr] exp_id=${EXP_ID} split=${SPLIT}"
echo "[run_transcdr] Split file: ${SPLIT_FILE}"

# -----------------------------------------------------------------------------
# Conda activation
# -----------------------------------------------------------------------------
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[run_transcdr] ERROR: conda.sh not found at ${CONDA_SH}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# -----------------------------------------------------------------------------
# GPU selection
# -----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[run_transcdr] GPU: CUDA_VISIBLE_DEVICES=${GPU_ID}"

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
# Output convention (registry): Output/<exp_id>/<MODEL>/<scenario>/metrics.csv
OUTPUT_ABS="${COMMON_ROOT}/Output/${EXP_ID}/${MODEL_NAME}/${SPLIT}/metrics.csv"

if [[ "${MOCK}" == true ]]; then
    echo "[run_transcdr] Mode: MOCK"

    python -u "${MOCK_RUNNER}" \
        --model "${MODEL_NAME}" \
        --split "${SPLIT}" \
        --exp_id "${EXP_ID}"
else
    echo "[run_transcdr] Mode: REAL TRAINING"

    # Per-exp_id workspace under TransCDR/ so concurrent exps don't clobber.
    DATA_PATH_REL="./data/GDSC/${EXP_ID}/${SPLIT}/external"
    MODELDIR_REL="./result/${EXP_ID}/external/${SPLIT}"

    cd "${MODEL_DIR}"

    echo "[run_transcdr] === Step1: data split ==="
    bash "${SCRIPT_DIR}/Step1_data_split.sh" \
        --scenarios external \
        --split "${SPLIT}" \
        --exp_id "${EXP_ID}" \
        --result_folder "./data/GDSC/${EXP_ID}/${SPLIT}"

    echo "[run_transcdr] === Step2: train 5 folds ==="
    bash "${SCRIPT_DIR}/Step2_TransCDR_CV10.sh" \
        --data_path "${DATA_PATH_REL}" \
        --modeldir "${MODELDIR_REL}"

    echo "[run_transcdr] === Step3: aggregate CV results ==="
    bash "${SCRIPT_DIR}/Step3_CV10_result.sh" \
        --CV5_result_path "${MODELDIR_REL}" \
        --output "${OUTPUT_ABS}"

    echo "[run_transcdr] Done. Metrics: ${OUTPUT_ABS}"
fi
