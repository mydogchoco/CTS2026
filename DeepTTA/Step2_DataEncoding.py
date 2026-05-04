# python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE
from Step1_getData import GetData


class DataEncoding:
    def __init__(self, vocab_dir):
        self.vocab_dir = vocab_dir
        self.Getdata = GetData()

    def _drug2emb_encoder(self, smile):
        vocab_path = "{}/ESPF/drug_codes_chembl_freq_1500.txt".format(self.vocab_dir)
        sub_csv = pd.read_csv("{}/ESPF/subword_units_map_chembl_freq_1500.csv".format(self.vocab_dir))

        bpe_codes_drug = codecs.open(vocab_path)
        dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        idx2word_d = sub_csv['index'].values
        words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

        max_d = 50
        t1 = dbpe.process_line(smile).split()
        try:
            i1 = np.asarray([words2idx_d[i] for i in t1])
        except:
            i1 = np.array([0])

        l = len(i1)
        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l))
        else:
            i = i1[:max_d]
            input_mask = [1] * max_d

        return i, np.asarray(input_mask)

    def encode(self, traindata, testdata):
        drug_smiles = self.Getdata.getDrug()
        # ── DRUG_ID → DRUG_NAME 으로 키 변경 ──
        drugname2smile = dict(zip(drug_smiles['DRUG_NAME'], drug_smiles['smiles']))
        smile_encode = pd.Series(drug_smiles['smiles'].unique()).apply(self._drug2emb_encoder)
        uniq_smile_dict = dict(zip(drug_smiles['smiles'].unique(), smile_encode))

        traindata['smiles'] = [drugname2smile[i] for i in traindata['DRUG_NAME']]
        testdata['smiles']  = [drugname2smile[i] for i in testdata['DRUG_NAME']]
        traindata['drug_encoding'] = [uniq_smile_dict[i] for i in traindata['smiles']]
        testdata['drug_encoding']  = [uniq_smile_dict[i] for i in testdata['smiles']]

        traindata = traindata.reset_index(drop=True)
        traindata['Label'] = traindata['LN_IC50']
        testdata = testdata.reset_index(drop=True)
        testdata['Label'] = testdata['LN_IC50']

        train_rnadata, test_rnadata = self.Getdata.getRna(
            traindata=traindata,
            testdata=testdata)
        # ── .T 삭제: exp.csv 이미 cells×genes ──
        train_rnadata.index = range(train_rnadata.shape[0])
        test_rnadata.index  = range(test_rnadata.shape[0])

        return traindata, train_rnadata, testdata, test_rnadata


if __name__ == '__main__':
    vocab_dir = '/home/intern3_2026_1/DeepTTC'
    obj = DataEncoding(vocab_dir=vocab_dir)
    traindata, testdata = obj.Getdata.ByRandom(random_seed=1)
    traindata, train_rnadata, testdata, test_rnadata = obj.encode(
        traindata=traindata,
        testdata=testdata)
    print("traindata:", traindata.shape, "| train_rna:", train_rnadata.shape)