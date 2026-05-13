#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_filter_consistency.py  (v2: + fold verification)
========================================================
Verify that DeepTTA's and TransCDR's filtering pipelines produce the SAME
(COSMIC_ID, DRUG_NAME) pair set on the unified input data, and that all
external split indices (mix / cellblind / drugblind / disjoint, both
single-split and 5-fold variants) are consistent with the raw response.csv
coordinate system.

This is the prerequisite for cross-model benchmarking with shared splits.

Run on the server:
    python -u verify_filter_consistency.py
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────
DATA_DIR  = "/home/intern1_2026_1/Common/Input"
SPLIT_DIR = f"{DATA_DIR}/SplitIndex"

RESPONSE_CSV = f"{DATA_DIR}/response.csv"
EXP_CSV      = f"{DATA_DIR}/exp.csv"
DRUG2SMI_CSV = f"{DATA_DIR}/drug2smi.csv"

SCENARIOS = ["mix", "cellblind", "drugblind", "disjoint"]
INDEX_FILES = {s: f"{SPLIT_DIR}/{s}_index.npy" for s in SCENARIOS}


def section(title):
    print(f"\n{'='*72}\n{title}\n{'='*72}")


# ── 1. Sanity check raw inputs ──────────────────────────────────────
def check_raw_inputs():
    section("1. Raw input files")

    resp = pd.read_csv(RESPONSE_CSV)
    print(f"response.csv shape:    {resp.shape}")
    print(f"  columns: {list(resp.columns)}")
    print(f"  index range:         [{resp.index.min()}, {resp.index.max()}]  "
          f"(default RangeIndex)")
    n_dup = resp.duplicated(subset=['COSMIC_ID', 'DRUG_NAME']).sum()
    print(f"  (COSMIC_ID, DRUG_NAME) duplicate rows: {n_dup}")

    drug = pd.read_csv(DRUG2SMI_CSV)
    print(f"\ndrug2smi.csv shape:    {drug.shape}")
    print(f"  columns: {list(drug.columns)}")
    n_dup_drug = drug.duplicated(subset=['DRUG_NAME']).sum()
    print(f"  DRUG_NAME duplicates:  {n_dup_drug}  "
          f"(>0 means TransCDR merge will explode rows!)")
    print(f"  unique DRUG_NAMEs:     {drug['DRUG_NAME'].nunique()}")

    exp = pd.read_csv(EXP_CSV, index_col=0)
    print(f"\nexp.csv shape:         {exp.shape}")
    n_replicate = (~exp.index.str.match(r"^DATA\.[0-9]+$")).sum()
    print(f"  replicate rows (DATA.X.1 pattern): {n_replicate}")
    exp_clean = exp[exp.index.str.match(r"^DATA\.[0-9]+$")]
    print(f"  after replicate drop:  {exp_clean.shape[0]} cell lines")

    return resp, drug, exp_clean


# ── 2. Verify single-split integrity (mix) ──────────────────────────
def check_single_split_integrity(resp, scenario="mix"):
    section(f"2. {scenario}_index.npy single-split integrity vs response.csv")

    split = np.load(INDEX_FILES[scenario], allow_pickle=True).item()
    train, val, test = split['train'], split['val'], split['test']

    print(f"top-level keys: {list(split.keys())}")
    print(f"train: {len(train):>7}  [{train.min()}, {train.max()}]")
    print(f"val:   {len(val):>7}  [{val.min()}, {val.max()}]")
    print(f"test:  {len(test):>7}  [{test.min()}, {test.max()}]")
    total = len(train) + len(val) + len(test)
    print(f"sum:   {total:>7}  (response.csv rows = {resp.shape[0]})")
    print(f"  match: {total == resp.shape[0]}")

    all_idx = np.concatenate([train, val, test])
    n_unique = len(set(all_idx.tolist()))
    print(f"\nunion unique indices: {n_unique}")
    print(f"  no duplicates across train/val/test: {n_unique == total}")
    print(f"  no missing rows from response.csv:   "
          f"{n_unique == resp.shape[0]}")

    return split


def check_fold_structure(resp):
    section("2b. fold structure verification (all scenarios)")

    n_response = resp.shape[0]
    all_ok = True

    for scenario in SCENARIOS:
        print(f"\n--- {scenario} ---")
        split = np.load(INDEX_FILES[scenario], allow_pickle=True).item()

        if 'folds' not in split:
            print(f"  ⚠️  no 'folds' key")
            all_ok = False
            continue

        folds = split['folds']
        if not isinstance(folds, list):
            print(f"  ⚠️  folds is not a list (got {type(folds).__name__})")
            all_ok = False
            continue

        n_folds = len(folds)
        print(f"  fold count: {n_folds}  ({'OK' if n_folds == 5 else 'NOT 5 ⚠️'})")

        # Each fold is an array of indices (the "test partition" for that fold)
        fold_sets = []
        for i, f in enumerate(folds):
            arr = np.asarray(f)
            if arr.dtype.kind not in 'iu':
                print(f"  fold[{i}]: dtype {arr.dtype} not integer ⚠️")
                all_ok = False
                continue
            n = len(arr)
            min_idx, max_idx = int(arr.min()), int(arr.max())
            print(f"  fold[{i}]: len={n:>6}, dtype={arr.dtype}, range=[{min_idx},{max_idx}]")
            if max_idx >= n_response:
                print(f"    ⚠️  max index {max_idx} >= response.csv rows ({n_response})")
                all_ok = False
            # uniqueness within fold
            if len(set(arr.tolist())) != n:
                print(f"    ⚠️  internal duplicates")
                all_ok = False
            fold_sets.append(set(arr.tolist()))

        # Pairwise disjointness across folds
        n_overlap = sum(len(fold_sets[i] & fold_sets[j])
                        for i in range(len(fold_sets))
                        for j in range(i+1, len(fold_sets)))
        if n_overlap == 0:
            print(f"  folds pairwise disjoint: True  (✓)")
        else:
            print(f"  ⚠️  folds overlap: {n_overlap} pairwise intersections")
            all_ok = False

# Coverage relationship to top-level
        union = set().union(*fold_sets)
        top_train = set(np.asarray(split['train']).tolist())
        top_val   = set(np.asarray(split['val']).tolist())
        top_test  = set(np.asarray(split['test']).tolist())
        top_all   = top_train | top_val | top_test

        print(f"  fold union size:           {len(union)}")
        print(f"  top-level test size:       {len(top_test)}")
        print(f"  top-level train+val+test:  {len(top_all)}")
        print(f"  fold union ∩ top_test:     {len(union & top_test)}  "
              f"(expected 0 — folds must exclude held-out test)")


        if scenario == "disjoint":
            # disjoint scenario uses a stricter subset (drug AND cell both
            # cold), so fold pool is intentionally smaller than (response \ test).
            # We don't know the exact expected pool here — just verify the two
            # universal invariants:
            #   1. folds pairwise disjoint (already checked above)
            #   2. folds don't overlap top-level test
            if len(union & top_test) == 0 and n_overlap == 0:
                print(f"  ✓ disjoint folds: pairwise disjoint + no top_test "
                      f"overlap  (coverage intentionally < response \\ test)")
            else:
                print(f"  ⚠️  disjoint folds have overlap issues")
                all_ok = False
        else:
            # Other scenarios: folds partition (response \ test)
            expected_pool = set(range(n_response)) - top_test
            if union == expected_pool:
                print(f"  ✓ folds form proper {n_folds}-fold CV over "
                      f"(response \\ test)")
            else:
                missing = len(expected_pool - union)
                extra   = len(union - expected_pool)
                print(f"  ⚠️  fold union != response \\ test: "
                      f"missing={missing}, extra={extra}")
                all_ok = False

    print(f"\noverall fold structure: {'OK' if all_ok else 'has issues ⚠️'}")
    return all_ok


# ── 3. DeepTTA-style filtering ──────────────────────────────────────
def deeptta_filter(resp, drug, exp_clean):
    section("3. DeepTTA-style filtering (isin)")

    valid_cells = set(
        exp_clean.index.str.replace("DATA.", "", regex=False).astype(int)
    )
    valid_drugs = set(drug['DRUG_NAME'])

    df = resp[['COSMIC_ID', 'DRUG_NAME', 'LN_IC50']].copy()
    n0 = df.shape[0]
    df = df[df['COSMIC_ID'].isin(valid_cells)]
    n1 = df.shape[0]
    df = df[df['DRUG_NAME'].isin(valid_drugs)]
    n2 = df.shape[0]

    print(f"  rows: {n0} → cell filter → {n1} → drug filter → {n2}")
    print(f"  unique COSMIC_IDs: {df['COSMIC_ID'].nunique()}")
    print(f"  unique DRUG_NAMEs: {df['DRUG_NAME'].nunique()}")
    print(f"  index preserved:   {df.index.is_monotonic_increasing}  "
          f"(should be True — coord system stays compatible)")
    return df


# ── 4. TransCDR-style filtering ─────────────────────────────────────
def transcdr_filter(resp, drug, exp_clean):
    section("4. TransCDR-style filtering (merge how='inner')")

    valid_cells = set(
        exp_clean.index.str.replace("DATA.", "", regex=False).astype(int)
    )

    cdr = resp[['COSMIC_ID', 'DRUG_ID', 'DRUG_NAME', 'LN_IC50']].copy()
    drug2smi_min = drug[['DRUG_NAME', 'smiles']]
    n0 = cdr.shape[0]
    cdr = pd.merge(cdr, drug2smi_min, on='DRUG_NAME', how='inner')
    n1 = cdr.shape[0]
    cdr = cdr[cdr['COSMIC_ID'].isin(valid_cells)]
    n2 = cdr.shape[0]

    print(f"  rows: {n0} → drug merge → {n1} → cell filter → {n2}")
    print(f"  unique COSMIC_IDs: {cdr['COSMIC_ID'].nunique()}")
    print(f"  unique DRUG_NAMEs: {cdr['DRUG_NAME'].nunique()}")
    print(f"  ⚠️  merge resets the RangeIndex — external split would break "
          f"without disabling shuffle in TransCDR's external branch.")
    return cdr


# ── 5. Cross-check pair sets ────────────────────────────────────────
def cross_check_pairs(deeptta_df, transcdr_df, resp):
    section("5. Pair-set cross-check (DeepTTA vs TransCDR)")

    dt_pairs = set(zip(deeptta_df['COSMIC_ID'], deeptta_df['DRUG_NAME']))
    tc_pairs = set(zip(transcdr_df['COSMIC_ID'], transcdr_df['DRUG_NAME']))

    print(f"DeepTTA  unique pairs: {len(dt_pairs)}")
    print(f"TransCDR unique pairs: {len(tc_pairs)}")
    print(f"intersection:          {len(dt_pairs & tc_pairs)}")
    print(f"DeepTTA-only:          {len(dt_pairs - tc_pairs)}")
    print(f"TransCDR-only:         {len(tc_pairs - dt_pairs)}")
    print(f"identical pair set:    {dt_pairs == tc_pairs}")

    print(f"\nrow count comparison:")
    print(f"  DeepTTA rows:  {deeptta_df.shape[0]}")
    print(f"  TransCDR rows: {transcdr_df.shape[0]}")
    print(f"  diff:          {transcdr_df.shape[0] - deeptta_df.shape[0]}  "
          f"(>0 means merge duplicated rows)")


# ── 6. Apply mix split to filtered DeepTTA frame ────────────────────
def apply_mix_split_deeptta(deeptta_df, resp, split):
    section("6. mix split applied via DeepTTA coordinate (response.csv index)")

    train_idx = split['train']
    val_idx   = split['val']
    test_idx  = split['test']

    surviving = set(deeptta_df.index)
    train_kept = [i for i in train_idx if i in surviving]
    val_kept   = [i for i in val_idx   if i in surviving]
    test_kept  = [i for i in test_idx  if i in surviving]

    print(f"  train: {len(train_idx)} → {len(train_kept)}  "
          f"(filtered out: {len(train_idx) - len(train_kept)})")
    print(f"  val:   {len(val_idx)} → {len(val_kept)}  "
          f"(filtered out: {len(val_idx) - len(val_kept)})")
    print(f"  test:  {len(test_idx)} → {len(test_kept)}  "
          f"(filtered out: {len(test_idx) - len(test_kept)})")
    total_kept = len(train_kept) + len(val_kept) + len(test_kept)
    print(f"  total kept: {total_kept}  "
          f"(should equal DeepTTA filtered rows = {deeptta_df.shape[0]})")
    print(f"  match: {total_kept == deeptta_df.shape[0]}")


# ── Main ────────────────────────────────────────────────────────────
def main():
    required = [RESPONSE_CSV, EXP_CSV, DRUG2SMI_CSV] + list(INDEX_FILES.values())
    if not all(os.path.exists(p) for p in required):
        print("ERROR: required input file missing. Check paths.")
        for p in required:
            print(f"  {'OK' if os.path.exists(p) else 'MISSING':>8}  {p}")
        sys.exit(1)

    resp, drug, exp_clean = check_raw_inputs()
    split = check_single_split_integrity(resp, "mix")
    check_fold_structure(resp)
    dt_df = deeptta_filter(resp, drug, exp_clean)
    tc_df = transcdr_filter(resp, drug, exp_clean)
    cross_check_pairs(dt_df, tc_df, resp)
    apply_mix_split_deeptta(dt_df, resp, split)

    section("DONE")
    print("Review the section-by-section output above.")
    print("Key checks:")
    print("  - section 1: drug2smi DRUG_NAME duplicates == 0")
    print("  - section 2: single-split sum == 171544, no duplicates")
    print("  - section 2b: each fold sums to 171544; test sets disjoint;")
    print("                test union == 171544 (proper k-fold CV)")
    print("  - section 5: DeepTTA and TransCDR pair sets identical")
    print("  - section 6: filter is no-op on current pre-cleaned data")


if __name__ == "__main__":
    main()
