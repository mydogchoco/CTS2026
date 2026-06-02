# python3
# -*- coding: utf-8 -*-
"""
predict_smiles.py - TransCDR single-SMILES inference.

Given ONE SMILES string, load a trained TransCDR model and predict the cancer
drug response (lnIC50) of that drug against every cell line the model was
trained on (the multi-omics intersection, ~440 cells; expr-only universe ~729).

This is the inference entry point for the "researcher-facing" demo: a user
supplies a SMILES that was never seen during training, and the model returns a
predicted response profile across all known cell lines.

Two modes
---------
  predict  : pure inference. Dummy labels are used; only predictions are
             meaningful. Output is one predicted lnIC50 per cell line.
  validate : sanity check using a drug from a drugblind TEST split. The drug
             was, by construction, never in the training set, so its predicted
             profile vs. the ground-truth lnIC50 measures generalization to an
             unseen drug. Pearson / Spearman / RMSE are reported on the cells
             that have a ground-truth value.

Design notes
------------
* DataEncoding.encode() is intentionally BYPASSED. encode() maps a SMILES to a
  BPE encoding via a dict built from drug2smi.csv, which raises KeyError for a
  truly novel SMILES. With pre_train=True + encoder fusion the model only uses
  embedded_drug1/2/3 (ChemBERTa / GIN / ECFP), which are built inside
  data_process_loader.__init__ from df['smiles'].unique() and therefore accept
  any valid SMILES. We assemble the inference DataFrame directly instead.
* The training config is reloaded from modeldir/config.json so that all MLP
  input dimensions match the saved state_dict exactly (expr-only vs multi-omics).
  If config.json is missing/empty/corrupt, a fallback config is reconstructed
  from the Step2_train_model.py schema using CLI flags (--omics, --drug_model,
  --fusion_type, --pre_train, ...). See build_config() below.
* The cell universe is taken from the COSMIC_IDs present in the training split
  files, guaranteeing every cid resolves in the omics .loc[] lookups.

Repository layout (server)
--------------------------
  Trained weights + config live under:
      .../CTS2026/TransCDR/result/<scenario>/.../fold<k>/{model.pt,config.json}
        e.g. result/CV10/fold1            result/external/mix/fold1
  Split .txt files live in a DIFFERENT tree:
      .../CTS2026/TransCDR/data/GDSC/<scenario>/.../fold<k>/{train,test,val}.txt
        e.g. data/GDSC/CV10/fold1         data/GDSC/mix/external/fold1
  So --modeldir (weights) and --data_path (splits) point at separate trees.

Example
-------
  # pure inference over all cell lines of the mix model (skeleton smoke test)
  python predict_smiles.py \
      --mode predict \
      --smiles "CC(C)Cc1ccc(cc1)C(C)C(=O)O" \
      --modeldir /home/intern1_2026_1/Common/CTS2026/TransCDR/result/external/mix/fold1 \
      --data_path /home/intern1_2026_1/Common/CTS2026/TransCDR/data/GDSC/mix \
      --gpu_id 0 \
      --output /home/intern1_2026_1/Common/Output/transcdr_predict_demo.csv

  # validation: auto-pick an unseen drug from a drugblind test split
  # (requires a trained drugblind model; the split root holds fold*/test.txt)
  python predict_smiles.py \
      --mode validate \
      --modeldir /home/intern1_2026_1/Common/CTS2026/TransCDR/result/external/drugblind/fold1 \
      --data_path /home/intern1_2026_1/Common/CTS2026/TransCDR/data/GDSC/drugblind \
      --gpu_id 0 \
      --output /home/intern1_2026_1/Common/Output/transcdr_validate_demo.csv
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd

# --- CLI -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="TransCDR single-SMILES inference over all training cell lines.")
    p.add_argument('--mode', type=str, default='predict',
                   choices=['predict', 'validate'],
                   help="predict: pure inference; validate: compare an unseen "
                        "test-split drug against ground truth.")
    p.add_argument('--smiles', type=str, default=None,
                   help="SMILES string to predict. Required for --mode predict.")
    p.add_argument('--drug_name', type=str, default=None,
                   help="(validate) Pick this drug from the test split. "
                        "If omitted, the test-split drug with the most cell "
                        "lines is used.")
    p.add_argument('--modeldir', type=str, required=True,
                   help="Directory holding the trained model.pt and config.json "
                        "(weights tree, e.g. .../result/external/mix/fold1).")
    p.add_argument('--data_path', type=str, required=True,
                   help="Scenario split root (splits tree), holding "
                        "[.../]fold*/{train,test,val}.txt. Searched recursively, "
                        "so both .../data/GDSC/CV10 and .../data/GDSC/mix "
                        "(which nests external/fold*) work. Used to derive the "
                        "cell universe and, for validate, the ground truth.")
    p.add_argument('--gpu_id', type=int, default=0,
                   help="CUDA device id.")
    p.add_argument('--output', type=str, required=True,
                   help="Output CSV path for the predicted response profile.")

    # --- Fallback config flags (C) -------------------------------------
    # Only consulted when modeldir/config.json is missing/empty/unparseable.
    # Values mirror Step2_train_model.py exactly; dimension-critical ones
    # (omics, drug_model, fusion_type, pre_train) are REQUIRED in that case
    # because they decide the predictor's first-layer width (256 * (n_drug +
    # n_omics)) and therefore must match the saved state_dict.
    g = p.add_argument_group(
        'fallback config (used only if config.json is missing/empty/corrupt)')
    g.add_argument('--model_type', type=str, default='regression',
                   help="regression or classification (Step2 default: regression).")
    g.add_argument('--omics', type=str, default=None,
                   help="e.g. 'expr' | 'expr + mutation + methylation'. "
                        "REQUIRED if the fallback is triggered.")
    g.add_argument('--drug_model', type=str, default=None,
                   help="e.g. 'sequence + graph + FP'. REQUIRED if fallback.")
    g.add_argument('--fusion_type', type=str, default=None,
                   help="concat | decoder | encoder. REQUIRED if fallback.")
    g.add_argument('--pre_train', type=str, default=None,
                   help="'True' | 'False' (kept as a string, as in Step2). "
                        "REQUIRED if fallback.")
    g.add_argument('--drug_encoder', type=str, default='None',
                   help="None | CNN | RNN | Transformer | GCN | NeuralFP | "
                        "AttentiveFP (Step2 default for pre_train runs: None).")
    g.add_argument('--seq_model', type=str,
                   default='seyonec/ChemBERTa-zinc-base-v1',
                   help="Pretrained SMILES model name (must match training).")
    g.add_argument('--graph_model', type=str, default='gin_supervised_masking',
                   help="Pretrained GIN model name (must match training).")
    return p.parse_args()


# --- config ----------------------------------------------------------------

# Verbatim from Step2_train_model.py's config dict. These omics input dims are
# FIXED there regardless of --omics: Classifier always builds all three omics
# MLPs (model_rna/model_genetic/model_mrna), so the state_dict always contains
# them. Do NOT invent other values; only the keys below are constant.
_STEP2_FIXED = {
    'input_dim_rna': 3000,
    'input_dim_genetic': 735,
    'input_dim_mrna': 20617,
    'input_dim_drug': 1068,   # unused by Classifier; kept for schema parity
    'KG': '',
    'decay': 0,
    'lr': 0.0001,             # unused at inference; placeholder for parity
    'BATCH_SIZE': 16,         # unused at inference; placeholder for parity
    'train_epoch': 0,         # unused at inference; placeholder for parity
}


def _fallback_config_from_step2(args, reason):
    """Reconstruct the training config from the Step2_train_model.py schema.

    Used only when config.json cannot be read. The dimension-critical flags
    (omics, drug_model, fusion_type, pre_train) MUST be supplied because the
    predictor's input width = 256 * (n_drug + n_omics) is derived from them; a
    wrong value here surfaces as a load_state_dict size-mismatch error.
    """
    required = {
        'omics': args.omics,
        'drug_model': args.drug_model,
        'fusion_type': args.fusion_type,
        'pre_train': args.pre_train,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        sys.exit(
            f"[fatal] config.json is {reason} in --modeldir, so the fallback "
            f"config must be specified, but these required flags are absent: "
            f"{missing}. Re-run with e.g. --omics 'expr' "
            f"--drug_model 'sequence + graph + FP' --fusion_type encoder "
            f"--pre_train True (values MUST match how the model was trained).")

    print(f"[warn] config.json is {reason}; reconstructing config from the "
          f"Step2 schema using CLI flags. WARNING: if omics/drug_model/"
          f"fusion_type/pre_train do not match training, load_state_dict will "
          f"fail with a size mismatch or predictions will be invalid.")

    config = dict(_STEP2_FIXED)
    config.update({
        'model_type': args.model_type,
        'omics': args.omics,
        'pre_train': args.pre_train,          # string, as in Step2
        'fusion_type': args.fusion_type,
        'drug_encoder': args.drug_encoder,
        'drug_model': args.drug_model,
        'seq_model': args.seq_model,
        'graph_model': args.graph_model,
    })
    print(f"[warn] fallback config: omics='{config['omics']}', "
          f"drug_model='{config['drug_model']}', "
          f"fusion_type='{config['fusion_type']}', "
          f"pre_train='{config['pre_train']}', "
          f"drug_encoder='{config['drug_encoder']}', "
          f"seq_model='{config['seq_model']}', "
          f"graph_model='{config['graph_model']}'")
    return config


def build_config(modeldir, args):
    """Load the exact training config; fall back to the Step2 schema if needed.

    Priority: a valid modeldir/config.json wins. The MLP dimensions and omics
    mode are baked into config.json at train time (model.py dumps self.config
    after training), so reusing it guarantees the rebuilt Classifier matches the
    saved state_dict. Only when config.json is missing/empty/unparseable do we
    reconstruct it from CLI flags (see _fallback_config_from_step2).
    """
    cfg_path = os.path.join(modeldir, 'config.json')
    config = None

    # (B) Don't trust os.path.exists alone: a 0-byte or malformed config.json
    # would otherwise slip through and blow up later with an opaque
    # JSONDecodeError. Detect each failure mode explicitly.
    if not os.path.exists(cfg_path):
        reason = 'missing'
    elif os.path.getsize(cfg_path) == 0:
        reason = 'empty (0 bytes)'
    else:
        try:
            with open(cfg_path, 'r') as f:
                config = json.load(f)
            reason = None
        except (json.JSONDecodeError, ValueError) as e:
            reason = f'unparseable ({e})'

    if config is None:
        # (C) Reconstruct from the Step2 schema; exits with guidance if the
        # dimension-critical flags are absent.
        config = _fallback_config_from_step2(args, reason)
    else:
        print(f"[config] loaded {cfg_path}")

    # Inference must run against GDSC omics, never an external screening set.
    config['screening'] = 'None'
    config['external_dataset'] = 'None'
    # Point the model at THIS dir so load_pretrained / TransCDR.__init__ are happy.
    config['modeldir'] = modeldir
    return config


# --- split-file helpers ----------------------------------------------------

def collect_split_files(data_path):
    """Return all train/test/val .txt files under data_path.

    (A) Splits live under data/GDSC/<scenario>/.../fold*/, at varying depth:
    one level for CV10 (data/GDSC/CV10/fold1) and two for mix
    (data/GDSC/mix/external/fold1). A recursive '**' glob handles both, while
    the direct match also covers a flat data_path/{name}. Pass --data_path as
    the SCENARIO root so the recursion stays scoped to one scenario.
    """
    files = []
    for name in ('train.txt', 'test.txt', 'val.txt'):
        files += glob.glob(os.path.join(data_path, name))
        files += glob.glob(os.path.join(data_path, '**', name), recursive=True)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No train/test/val .txt files found under {data_path}. "
            f"Point --data_path at a scenario split root such as "
            f".../data/GDSC/CV10 or .../data/GDSC/mix.")
    return files


def get_cell_universe(data_path):
    """COSMIC_IDs the model was trained on = union over all split files.

    Every cid here passed Step1's omics filter, so the omics .loc[cid] lookups
    inside data_process_loader are guaranteed to resolve.
    """
    split_files = collect_split_files(data_path)
    cids = set()
    for fp in split_files:
        df = pd.read_csv(fp, sep='\t')
        if 'COSMIC_ID' not in df.columns:
            raise KeyError(f"'COSMIC_ID' column missing in {fp}.")
        cids.update(int(c) for c in df['COSMIC_ID'].values)
    cids = sorted(cids)
    print(f"[cells] universe size = {len(cids)} "
          f"(from {len(split_files)} split files)")
    return cids


def build_inference_df(smiles, cosmic_ids, labels=None):
    """Assemble the (1 drug x N cells) DataFrame consumed by net.predict().

    Required columns mirror what data_process_loader.__getitem__ reads in the
    external_dataset='None' branch: COSMIC_ID (int) and smiles, plus a Label
    column (net.predict() passes drug_data.Label.values to the loader).
    cell_type / assay_name / lnIC50 are filled defensively to match Step1's
    schema; the 'None' branch does not read them.

    labels: optional array of ground-truth lnIC50 aligned with cosmic_ids
            (validate mode). If None, dummy zeros are used.
    """
    n = len(cosmic_ids)
    if labels is None:
        labels = np.zeros(n, dtype=float)
    df = pd.DataFrame({
        'COSMIC_ID': [int(c) for c in cosmic_ids],
        'smiles': [smiles] * n,
        'lnIC50': labels,
        'Label': labels,
    })
    # Step1 fills these with COSMIC_ID; harmless for the 'None' branch but kept
    # so the schema is identical to the training frames.
    df['cell_type'] = df['COSMIC_ID']
    df['assay_name'] = df['COSMIC_ID']
    df = df.reset_index(drop=True)
    return df


def resolve_validate_drug(data_path, drug_name):
    """From the test split, pick a drug and return its
    (drug_name, smiles, DataFrame of [COSMIC_ID, true_lnIC50]) ground truth.

    For a drugblind split this drug never appears in training, so the comparison
    reflects generalization to an unseen compound.
    """
    # (A) Same recursive search as collect_split_files, restricted to test.txt.
    test_files = sorted(set(
        glob.glob(os.path.join(data_path, 'test.txt')) +
        glob.glob(os.path.join(data_path, '**', 'test.txt'), recursive=True)))
    if not test_files:
        raise FileNotFoundError(f"No test.txt under {data_path} for validate mode.")

    test = pd.concat([pd.read_csv(fp, sep='\t') for fp in test_files],
                     ignore_index=True)

    # lnIC50 may be named 'lnIC50' (renamed in Step1) or 'LN_IC50' (raw).
    ic50_col = 'lnIC50' if 'lnIC50' in test.columns else 'LN_IC50'
    if ic50_col not in test.columns:
        raise KeyError(f"Neither 'lnIC50' nor 'LN_IC50' in test columns: "
                       f"{list(test.columns)}")

    # Identify the drug column: prefer an explicit name, else fall back to id.
    name_col = next((c for c in ('drug_name', 'DRUG_NAME') if c in test.columns), None)
    id_col = next((c for c in ('drug_id', 'DRUG_ID') if c in test.columns), None)

    if drug_name is not None:
        if name_col is None:
            raise KeyError("--drug_name given but no drug-name column in test.txt.")
        sel = test[test[name_col] == drug_name]
        if sel.empty:
            raise ValueError(f"Drug '{drug_name}' not found in test split.")
    else:
        # Auto-pick the drug with the most test-split cell lines (most points
        # to correlate => most informative sanity check).
        group_col = name_col if name_col is not None else id_col
        if group_col is None:
            raise KeyError("No drug name/id column found in test.txt.")
        counts = test.groupby(group_col).size().sort_values(ascending=False)
        chosen = counts.index[0]
        sel = test[test[group_col] == chosen]
        drug_name = str(chosen)

    if 'smiles' not in sel.columns:
        raise KeyError("'smiles' column missing in test.txt; cannot run inference.")
    smiles = sel['smiles'].iloc[0]

    truth = (sel[['COSMIC_ID', ic50_col]]
             .rename(columns={ic50_col: 'true_lnIC50'})
             .drop_duplicates('COSMIC_ID')
             .reset_index(drop=True))
    truth['COSMIC_ID'] = truth['COSMIC_ID'].astype(int)
    print(f"[validate] drug='{drug_name}' | smiles={smiles} | "
          f"{len(truth)} ground-truth cell lines")
    return drug_name, smiles, truth


# --- main ------------------------------------------------------------------

def main():
    args = parse_args()

    # Pin the GPU before importing model.py (it sets `device` at import time).
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # model.py lives in the TransCDR package dir and uses relative ./data paths,
    # so run this script with TransCDR/ as the working directory (see notes).
    from model import TransCDR  # noqa: E402

    config = build_config(args.modeldir, args)
    if config.get('model_type', 'regression') != 'regression':
        print("[warn] config model_type is not 'regression'; predictions are "
              "binary-classification logits, not lnIC50.")

    cosmic_ids = get_cell_universe(args.data_path)

    if args.mode == 'validate':
        drug_name, smiles, truth = resolve_validate_drug(args.data_path, args.drug_name)
        # Predict over the FULL universe, then join ground truth where present.
        truth_map = dict(zip(truth['COSMIC_ID'], truth['true_lnIC50']))
        labels = np.array([truth_map.get(c, 0.0) for c in cosmic_ids], dtype=float)
        infer_df = build_inference_df(smiles, cosmic_ids, labels=labels)
    else:
        if not args.smiles:
            sys.exit("--smiles is required for --mode predict.")
        smiles = args.smiles
        drug_name = None
        infer_df = build_inference_df(smiles, cosmic_ids, labels=None)

    # Build the model from the (possibly reconstructed) config and load weights.
    net = TransCDR(**config)
    model_pt = os.path.join(args.modeldir, 'model.pt')
    if not os.path.exists(model_pt):
        sys.exit(f"Trained weights not found: {model_pt}")
    net.load_pretrained(path=model_pt)

    # predict() returns the regression tuple; y_pred is aligned with infer_df row
    # order (SequentialSampler, shuffle=False), hence with cosmic_ids.
    y_label, y_pred, mse, rmse, pearson, p_val, spearman, s_p_val, CI = \
        net.predict(infer_df)
    y_pred = np.asarray(y_pred, dtype=float)

    result = pd.DataFrame({
        'COSMIC_ID': cosmic_ids,
        'pred_lnIC50': y_pred,
    })

    if args.mode == 'validate':
        result = result.merge(truth, on='COSMIC_ID', how='left')
        result['drug_name'] = drug_name
        # Metrics only on cells that actually have a ground-truth value.
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import mean_squared_error
        ev = result.dropna(subset=['true_lnIC50'])
        if len(ev) >= 3:
            pcc = pearsonr(ev['true_lnIC50'], ev['pred_lnIC50'])[0]
            scc = spearmanr(ev['true_lnIC50'], ev['pred_lnIC50'])[0]
            rmse_v = float(np.sqrt(mean_squared_error(ev['true_lnIC50'], ev['pred_lnIC50'])))
            print(f"[validate] n={len(ev)} | PCC={pcc:.4f} | "
                  f"SCC={scc:.4f} | RMSE={rmse_v:.4f}")
        else:
            print(f"[validate] only {len(ev)} ground-truth cells; "
                  f"too few for correlation.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"[done] wrote {len(result)} rows -> {args.output}")
    if drug_name:
        print(f"[done] drug='{drug_name}', smiles={smiles}")
    else:
        print(f"[done] smiles={smiles}")


if __name__ == '__main__':
    main()
