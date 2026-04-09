# CTS2026 - CDR 모델 구현

AIGENDRUG 인턴 프로젝트 - 암 약물 반응 예측 모델 구현

## 모델 목록
- **DeepTTA**: Transformer 기반 CDR 예측 모델 (jianglikun/DeepTTC 기반)
- **TransCDR**: 멀티모달 CDR 예측 모델 (XiaoqiongXia/TransCDR 기반)

## 환경 재현
```bash
conda env create -f environment_deeptta.yml
conda activate deeptta
```

## 주요 수정 사항
- DeepTTA: pickle5 호환성 수정, pandas append 수정, 데이터 경로 수정
- TransCDR: dgl → PyTorch Geometric 교체 (Blackwell GPU 호환성)
