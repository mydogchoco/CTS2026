# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-Step1_getData.py
@time:2021/8/12 15:48 
"""
import sys
import csv
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

# python3
# -*- coding:utf-8 -*-

import sys
import csv
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

# ── Unified data paths ──────────────────────────────────────────────
DATA_DIR = "/home/intern1_2026_1/Common/Input"


class GetData():
    def __init__(self):
        self.rnafile   = DATA_DIR + "/exp.csv"
        self.smilefile = DATA_DIR + "/drug2smi.csv"
        self.pairfile  = DATA_DIR + "/response.csv"

    def getDrug(self):
        # drug2smi.csv: ",DRUG_NAME,smiles" → index_col=0 으로 unnamed 인덱스 제거
        return pd.read_csv(self.smilefile, index_col=0)

    def _filter_pair(self, drug_cell_df):
        print("#" * 50)
        print("Step1: filter cells present in exp.csv")
        exp = pd.read_csv(self.rnafile, index_col=0)
        exp = exp[exp.index.str.match(r"^DATA\.[0-9]+$")]   # drop replicate rows (.1 suffix)
        exp.index = exp.index.str.replace("DATA.", "", regex=False).astype(int)
        valid_cells = set(exp.index)

        before = drug_cell_df.shape[0]
        drug_cell_df = drug_cell_df[drug_cell_df['COSMIC_ID'].isin(valid_cells)]
        print(f"  cells: {before} → {drug_cell_df.shape[0]} rows")

        print("Step2: filter drugs present in drug2smi.csv")
        drug2smi = self.getDrug()
        valid_drugs = set(drug2smi['DRUG_NAME'])

        before = drug_cell_df.shape[0]
        drug_cell_df = drug_cell_df[drug_cell_df['DRUG_NAME'].isin(valid_drugs)]
        print(f"  drugs: {before} → {drug_cell_df.shape[0]} rows")

        return drug_cell_df

    def ByRandom(self, random_seed=42, test_size=0.1):
        """Mixed split (random 80/20) — benchmark default split"""
        drug_cell_df = pd.read_csv(self.pairfile)
        drug_cell_df = drug_cell_df[['COSMIC_ID', 'DRUG_NAME', 'LN_IC50']]
        drug_cell_df = self._filter_pair(drug_cell_df)

        train_data, test_data = train_test_split(
            drug_cell_df, test_size=test_size, random_state=random_seed)
        train_data = train_data.reset_index(drop=True)
        test_data  = test_data.reset_index(drop=True)
        print(f"Train: {train_data.shape[0]}, Test: {test_data.shape[0]}")
        return train_data, test_data

    def getRna(self, traindata, testdata):
        """exp.csv: rows=cells (already cells×genes, no transpose needed)"""
        exp = pd.read_csv(self.rnafile, index_col=0)
        exp = exp[exp.index.str.match(r"^DATA\.[0-9]+$")]   # drop replicate rows (.1 suffix)
        exp.index = exp.index.str.replace("DATA.", "", regex=False).astype(int)

        # Preserve the row order of train/testdata (same cell may repeat)
        train_rna = exp.loc[list(traindata['COSMIC_ID'])].reset_index(drop=True)
        test_rna  = exp.loc[list(testdata['COSMIC_ID'])].reset_index(drop=True)

        print(f"RNA shape — train: {train_rna.shape}, test: {test_rna.shape}")
        return train_rna, test_rna

    def ByExternalIndex(self, split_file):
        """External index split (train/val/test from team-provided npy file).

        Coord system: original response.csv RangeIndex
        (verified by verify_filter_consistency.py — pre-filtered data,
        no rows lost during _filter_pair).
        """
        drug_cell_df = pd.read_csv(self.pairfile)
        drug_cell_df = drug_cell_df[['COSMIC_ID', 'DRUG_NAME', 'LN_IC50']]
        drug_cell_df = self._filter_pair(drug_cell_df)   # defensive (no-op on current data)

        split = np.load(split_file, allow_pickle=True).item()
        train_df = drug_cell_df.loc[split['train']].reset_index(drop=True)
        val_df   = drug_cell_df.loc[split['val']  ].reset_index(drop=True)
        test_df  = drug_cell_df.loc[split['test'] ].reset_index(drop=True)
        print(f"External split — train: {len(train_df)}, "
            f"val: {len(val_df)}, test: {len(test_df)}")
        return train_df, val_df, test_df

    def getRna_three_way(self, traindata, valdata, testdata):
        """Three-way version of getRna for train/val/test split."""
        exp = pd.read_csv(self.rnafile, index_col=0)
        exp = exp[exp.index.str.match(r"^DATA\.[0-9]+$")]
        exp.index = exp.index.str.replace("DATA.", "", regex=False).astype(int)

        train_rna = exp.loc[list(traindata['COSMIC_ID'])].reset_index(drop=True)
        val_rna   = exp.loc[list(valdata['COSMIC_ID'])  ].reset_index(drop=True)
        test_rna  = exp.loc[list(testdata['COSMIC_ID']) ].reset_index(drop=True)
        print(f"RNA shape — train: {train_rna.shape}, "
            f"val: {val_rna.shape}, test: {test_rna.shape}")
        return train_rna, val_rna, test_rna



if __name__ == '__main__':
    obj = GetData()
    train, test = obj.ByRandom(random_seed=42)
    train_rna, test_rna = obj.getRna(train, test)
    print("Sanity check passed.")
