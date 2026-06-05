#!/bin/bash
set -euo pipefail

DATA_PATH='./data/GDSC/CV5'
MODELDIR='./result/CV5'

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_path) DATA_PATH="$2"; shift 2 ;;
        --modeldir)  MODELDIR="$2";  shift 2 ;;
        *) echo "[Step2] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

python -u Step2_train_model.py \
    --model_type 'regression' \
    --data_path "${DATA_PATH}" \
    --omics 'expr' \
    --input_dim_drug 2092 \
    --lr 1e-5 \
    --BATCH_SIZE 64 \
    --train_epoch 50 \
    --pre_train 'True' \
    --screening 'None' \
    --fusion_type 'encoder' \
    --drug_encoder 'None' \
    --drug_model 'sequence + graph + FP' \
    --modeldir "${MODELDIR}" \
    --seq_model 'seyonec/PubChem10M_SMILES_BPE_450k' \
    --graph_model 'gin_supervised_masking' \
    --external_dataset 'None'
