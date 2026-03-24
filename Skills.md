# Skills & Capabilities - Chai and Samosa

## What We Built (7 Notebooks, 40+ Models)

### Skill 1: Data Acquisition & Preprocessing [COMPLETED]
- **What**: Downloaded and parsed all 35 Kaggle CSVs, cleaned missing values, merged datasets
- **Tools**: pandas, numpy, kaggle API
- **Data location**: `data/raw/` (35 CSV files)
- **Output**: Clean merged dataframes used across all notebooks

### Skill 2: Extensive EDA [COMPLETED]
- **What**: Deep exploratory analysis covering seeds, upsets, conferences, coaches, momentum
- **Notebook**: `notebooks/eda_kaggle.ipynb`
- **Coverage**: 12 analysis areas including seed vs outcome, upset patterns, conference strength, geographic advantage, Massey ordinals comparison
- **Output**: 30+ visualizations with written insights

### Skill 3: Feature Engineering [COMPLETED]
- **Features built** (50+ per matchup):
  - Dual Elo ratings (regular season + tournament-only)
  - Elo momentum (late season form trend)
  - Four Factors (eFG%, TO%, ORB%, FT rate) — offensive & defensive
  - Massey ordinals PCA (top 3 principal components from 100+ ranking systems)
  - GLM team quality scores (Bradley-Terry latent strengths)
  - Seed-based features (seed diff, seed product, historical seed win%)
  - Offensive/defensive efficiency (points per 100 possessions)
  - Rolling momentum features (last N games performance)
  - Strength of schedule
  - Conference tournament performance
  - Coach experience features
  - Point differential and win percentage
  - Symmetric matchup features (TeamA - TeamB for each metric)

### Skill 4: Core Models (train_kaggle) [COMPLETED]
- **CV Brier**: 0.1687
- **Models**: LR (3 variants), XGBoost, XGBoost-Tuned, LightGBM, CatBoost, Elo
- **Ensemble**: Optimized weights + Ridge stacking meta-learner
- **Calibration**: Isotonic regression
- **Submissions**: Conservative (30% seed prior blend) + Aggressive

### Skill 5: Deep Learning Models (advanced_models_kaggle) [COMPLETED]
- **CV Brier**: 0.1660
- **Models**:
  - Team Autoencoder (unsupervised feature learning)
  - MatchupMLP (3-layer feedforward)
  - TransformerMatchup (multi-head attention, d=64)
  - LSTMMatchup (2-layer, hidden=128)
  - Hybrid GBMs (DL embeddings fed into XGBoost/LightGBM)
- **Training**: PyTorch on GTX 1650 Ti (4GB VRAM)

### Skill 6: Expert Models (expert_models_kaggle) [PARTIALLY COMPLETE]
- **Models** (13 specialized):
  - Bradley-Terry, Random Forest, Extra Trees, SVM-RBF, KNN
  - GradientBoosting, BayesianRidge, TabNet, Wide&Deep
  - Sklearn-MLP, ElasticNet, Gaussian Process, Historical Seed Prior
- **Status**: Notebook ready, local training timed out

### Skill 7: Temporal/Foundation Models (foundation_models_kaggle) [COMPLETED]
- **CV Brier**: 0.1632
- **Models**:
  - BiLSTM-Attention (44.6% weight) — best single temporal model
  - TFT-Lite (26.8%) — custom GRN + VSN + LSTM + attention
  - TS-Embedder (24.3%) — time series embedding network
  - Conv1D (4.3%) — 1D convolutional temporal model
  - LR-Temporal — logistic regression on temporal features
- **Features**: 5-season team trajectories, rolling 3-year averages, trend signals

### Skill 8: Mega Ensemble (mega_ensemble_kaggle) [COMPLETED]
- **CV Brier**: ~0.17
- **What**: Self-contained CPU-only notebook with 13+ models
- **Models**: All sklearn + GBM models trained from scratch
- **Meta-learner**: Ridge stacking + optimized weights
- **Advantage**: No GPU needed, fully reproducible

### Skill 9: Netflix-Inspired Creative Models (creative_models_kaggle) [PARTIALLY COMPLETE]
- **Models** (8 novel approaches):
  - SVD++ for Sports (matrix factorization with implicit feedback)
  - Neural Collaborative Filtering (NeuMF: GMF + MLP paths)
  - Siamese Network (twin encoders, symmetric comparison)
  - Mixture of Experts (4 expert nets + gating network)
  - Contrastive Learning (pull winners closer, push losers apart)
  - Ordinal Margin Net (predict margin, convert to P(win))
  - NMF Game Model (non-negative matrix factorization)
  - Margin Regression -> Sigmoid
- **Status**: Notebook ready, local training timed out

### Skill 10: Ultimate Model (ultimate_model) [COMPLETED - BEST]
- **CV Brier**: Men's 0.0902, Women's 0.1373
- **Combines ALL winning techniques**:
  1. XGBoost regression on point differential (1st place winner technique)
  2. Spline calibration (non-parametric margin -> probability)
  3. LightGBM + CatBoost classifiers
  4. Logistic Regression (calibration anchor)
  5. Logit-space Ridge meta-learner (MetaStack)
  6. Isotonic calibration
  7. Separate Men's/Women's models
  8. Symmetric data augmentation (2x data)
  9. Dual Elo (regular + tournament-only)
  10. Elo momentum
  11. Massey ordinals PCA
  12. GLM team quality (Bradley-Terry)
  13. Four Factors of basketball
  14. Inverse-Brier weighted ensemble
  15. Historical seed prior
  16. goto_conversion (favourite-longshot bias)
  17. Tight clipping [0.025, 0.975]

### Skill 11: Post-Processing & Calibration [COMPLETED]
- **goto_conversion**: Applied to all submissions (alpha=1.15)
- **Isotonic regression**: Applied within each model pipeline
- **Spline calibration**: Used in ultimate model for margin->probability
- **Probability clipping**: [0.05, 0.95] standard, [0.025, 0.975] for ultimate

### Skill 12: Submission Strategy [IN PROGRESS]
- **7 submissions on Kaggle** — waiting for tournament scores
- **Strategy**: Once scores appear (March 17-18), compare all submissions and select best 2 before March 19 deadline
- **Approach**: Diverse submissions (conservative to aggressive, different model families)

## Computational Setup
- **Local GPU**: NVIDIA GeForce GTX 1650 Ti (4GB VRAM)
- **Local Python**: Python 3.13 (miniconda + pip)
- **Kaggle**: GPU quota shared with another competition
- **Key packages**: PyTorch, XGBoost, LightGBM, CatBoost, scikit-learn, pandas, numpy, scipy

## Key Learnings
1. **Point differential regression > classification** for predicting game outcomes
2. **Spline calibration** outperforms isotonic for margin-to-probability conversion
3. **Logit-space meta-learning** (MetaStack) better than raw probability stacking
4. **Separate Men/Women models** improves over combined model
5. **goto_conversion** is a simple but powerful post-processing step
6. **Symmetric augmentation** doubles data without introducing bias
7. **Dual Elo** (regular + tournament) captures different team dynamics
8. **Massey PCA** compresses 100+ ranking systems into 3 powerful features
9. **Simple models with great features** beat complex models with poor features
10. **Calibration is everything** with Brier score metric
