# Computational Requirements

## Kaggle Notebook Environment (Primary)

| Resource | Limit | Our Usage |
|----------|-------|-----------|
| GPU | P100 16GB or T4 16GB | DL models only |
| GPU Quota | 30 hrs/week | ~8-10 hrs estimated |
| Session Length | 12 hrs max | Longest run ~4 hrs |
| RAM | 16 GB | ~4-6 GB peak |
| Disk | 73 GB | ~2 GB total |
| CPU Cores | 4 | GBM training |
| Internet | Available during edit | Data download, pip |

## Estimated Training Times (Kaggle GPU)

| Model | Time | GPU Required |
|-------|------|-------------|
| Data loading + preprocessing | 1-2 min | No |
| Elo rating computation | 30 sec | No |
| Feature engineering | 1-2 min | No |
| Logistic Regression | < 10 sec | No |
| XGBoost (300 trees) | 1-3 min | No |
| LightGBM (300 trees) | 1-2 min | No |
| CatBoost (300 iters) | 2-4 min | No |
| Optuna tuning (100 trials) | 15-30 min | No |
| LSTM (50 epochs) | 10-30 min | Yes |
| Transformer (50 epochs) | 15-45 min | Yes |
| TFT (100 epochs) | 30-90 min | Yes |
| Ensemble optimization | < 1 min | No |
| Submission generation | < 30 sec | No |
| **Total estimated** | **~2-4 hrs** | |

## Local Environment (EDA Only)

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk | 1 GB | 2 GB |
| GPU | Not needed | Not needed |

## Package Versions (Kaggle Pre-installed as of Mar 2026)

| Package | Kaggle Version | Our Requirement |
|---------|---------------|----------------|
| Python | 3.10.x | >=3.10 |
| numpy | 1.26.x | >=1.24 |
| pandas | 2.1.x | >=2.0 |
| scikit-learn | 1.4.x | >=1.3 |
| xgboost | 2.0.x | >=2.0 |
| lightgbm | 4.2.x | >=4.1 |
| catboost | 1.2.x | >=1.2 |
| torch | 2.1.x | >=2.1 |
| matplotlib | 3.8.x | >=3.7 |
| seaborn | 0.13.x | >=0.12 |

## GPU Quota Budget Plan

| Day | Task | GPU Hours | Cumulative |
|-----|------|-----------|------------|
| 1-2 | EDA (CPU only) | 0 | 0 |
| 3 | GBM tuning (CPU) | 0 | 0 |
| 3 | LSTM training | 1 | 1 |
| 4 | Transformer training | 1.5 | 2.5 |
| 4 | TFT training | 2 | 4.5 |
| 4-5 | Retraining + ensemble | 2 | 6.5 |
| **Buffer** | | **23.5** | **30** |
