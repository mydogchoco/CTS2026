import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='CV5 results aggregation')
parser.add_argument('--CV5_result_path', type=str, required=True,
                    help='the CV5 result path (contains fold1..fold5)')
parser.add_argument('--output', type=str, required=True,
                    help='output metrics.csv path (columns: fold,pcc,spearman,rmse,mse)')
args = parser.parse_args()

# fold별 메트릭 수집 (fold 순서 1..5 보존)
folds = []
MSE, RMSE, Pearson, Pearson_pval = [], [], [], []
Spearman, Spearman_pval, Concordance_Index = [], [], []

for i in range(1, 6):
    res = pd.read_csv(args.CV5_result_path + '/fold' + str(i) + '/test_markdowntable.txt')
    folds.append(i)
    MSE.append(float(res.iloc[2, 0].split('|')[1]))
    RMSE.append(float(res.iloc[2, 0].split('|')[2]))
    Pearson.append(float(res.iloc[2, 0].split('|')[3]))
    Pearson_pval.append(float(res.iloc[2, 0].split('|')[4]))
    Spearman.append(float(res.iloc[2, 0].split('|')[5]))
    Spearman_pval.append(float(res.iloc[2, 0].split('|')[6]))
    Concordance_Index.append(float(res.iloc[2, 0].split('|')[7]))

# 내부 보존용 (전체 메트릭 + p-value + CI) — RMSE sort, 디버깅/검증에 유용
result = pd.DataFrame({
    'fold': folds,
    'MSE': MSE, 'RMSE': RMSE,
    'Pearson': Pearson, 'Pearson_pval': Pearson_pval,
    'Spearman': Spearman, 'Spearman_pval': Spearman_pval,
    'Concordance_Index': Concordance_Index,
}).sort_values(by='RMSE')
# fold는 식별자라 mean/std 계산에서 제외
result.drop(columns=['fold']).describe().to_csv(args.CV5_result_path + '/res_mean.csv')
result.to_csv(args.CV5_result_path + '/res.csv', index=False)

# 팀 표준 출력: fold,pcc,spearman,rmse,mse (fold 1..5 순서)
standard = pd.DataFrame({
    'fold': folds,
    'pcc': Pearson,
    'spearman': Spearman,
    'rmse': RMSE,
    'mse': MSE,
})
os.makedirs(os.path.dirname(args.output), exist_ok=True)
standard.to_csv(args.output, index=False)

print(f"[Step3] Results saved to: {args.output}")
print(standard.to_string(index=False))
print()
print(f"  Pearson  = {standard['pcc'].mean():.4f} ± {standard['pcc'].std():.4f}")
print(f"  Spearman = {standard['spearman'].mean():.4f} ± {standard['spearman'].std():.4f}")
print(f"  RMSE     = {standard['rmse'].mean():.4f} ± {standard['rmse'].std():.4f}")
print(f"  MSE      = {standard['mse'].mean():.4f} ± {standard['mse'].std():.4f}")