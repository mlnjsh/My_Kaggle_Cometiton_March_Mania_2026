# March Machine Learning Mania 2026 - Project Guide

## Competition Overview
- **Competition**: March Machine Learning Mania 2026 (12th annual)
- **Team Name**: Chai and Samosa
- **Kaggle Username**: MilanJoshi
- **URL**: https://www.kaggle.com/competitions/march-machine-learning-mania-2026
- **Goal**: Forecast the 2026 NCAA Division 1 Men's & Women's Basketball Tournaments
- **Prize Pool**: $50,000
- **Stage 1 Deadline**: ~March 15, 2026 (historical validation - MUST submit to qualify for Stage 2)
- **Stage 2 Deadline**: March 19th, 2026 (actual 2026 bracket predictions)
- **Evaluation Metric**: **Brier Score / MSE** (MSE between predicted probabilities and actual 0/1 outcomes). Lower = better.
  - Note: Some years used Log Loss. 2025/2026 uses MSE (Brier). Predictions clipped to [0.05, 0.95]
- **Submission Format**: CSV with columns `ID` and `Pred`
  - `ID` = `2026_TeamIdLow_TeamIdHigh`
  - `Pred` = P(lower TeamId beats higher TeamId)
  - Men's TeamIDs: 1000-1999, Women's TeamIDs: 3000-3999

## Current Status (as of March 17, 2026)
- **7 submissions uploaded to Kaggle** (all showing 0.00000 - tournament starts today)
- **Best CV Brier**: Men's 0.0902, Women's 0.1373 (Ultimate model)
- **14 submission CSVs generated locally** in `.tmp/`
- **Tournament starts**: March 17, 2026 (First Four games)
- **Deadline**: March 19, 2026 — must select best 2 submissions

## Submissions on Kaggle
| # | File | CV Brier | Description |
|---|---|---|---|
| 1 | submission_stage2_ultimate_goto.csv | 0.09/0.14 | Ultimate ensemble + goto bias correction |
| 2 | submission_stage2_ultimate.csv | 0.09/0.14 | Ultimate ensemble (no goto) |
| 3 | submission_stage2_mega_goto.csv | ~0.17 | Mega ensemble + goto |
| 4 | submission_stage2_mega.csv | ~0.17 | Mega ensemble (13+ models) |
| 5 | submission_stage2_temporal_goto.csv | 0.1632 | Temporal ensemble + goto |
| 6 | submission_stage2_temporal.csv | 0.1632 | Temporal ensemble |
| 7 | submission_stage2_conservative.csv | 0.1687 | Original ensemble (conservative blend) |

## Project Structure
```
March_ML_Mania_2026/
  data/raw/              # Competition CSV files (35 files)
  data/processed/        # Processed/intermediate data
  data/external/         # External data sources
  notebooks/             # Kaggle-ready .ipynb notebooks
    eda_kaggle.ipynb             # Exploratory data analysis
    train_kaggle.ipynb           # Original training (LR, XGB, LGBM, CatBoost, Elo)
    advanced_models_kaggle.ipynb # DL models (MLP, Transformer, LSTM, Autoencoder)
    expert_models_kaggle.ipynb   # 13 specialized models (RF, SVM, KNN, TabNet, etc.)
    foundation_models_kaggle.ipynb # Temporal models (BiLSTM, TFT, Conv1D, TS-Embedder)
    mega_ensemble_kaggle.ipynb   # Mega ensemble (13+ models, CPU-only)
    creative_models_kaggle.ipynb # Netflix-inspired (SVD++, NeuMF, Siamese, MoE)
  .tmp/                  # Local execution scripts + generated CSVs (gitignored)
  configs/               # Configuration files
  src/                   # Source code modules
  catboost_info/         # CatBoost training logs
  Chai_and_Samosa_Playbook.md  # Detailed playbook document
  Chai_and_Samosa_Playbook.pdf # PDF version of playbook
```

## Models Built

### Notebook 1: train_kaggle (Original Ensemble)
- **CV Brier**: 0.1687
- **Models**: Logistic Regression (3 variants), XGBoost, XGBoost-Tuned, LightGBM, CatBoost, Elo
- **Ensemble**: Optimized weights via scipy.optimize + Ridge stacking
- **Output**: submission_stage2.csv, submission_stage2_conservative.csv

### Notebook 2: advanced_models_kaggle (Deep Learning)
- **CV Brier**: 0.1660
- **Models**: Team Autoencoder, MatchupMLP, TransformerMatchup, LSTMMatchup, Hybrid GBMs
- **Output**: submission_stage2_advanced.csv

### Notebook 3: expert_models_kaggle (13 Specialized Models)
- **Models**: Bradley-Terry, Random Forest, Extra Trees, SVM-RBF, KNN, GradientBoosting, BayesianRidge, TabNet, Wide&Deep, Sklearn-MLP, ElasticNet, Gaussian Process, Historical Seed Prior
- **Status**: Training incomplete (timed out locally)

### Notebook 4: foundation_models_kaggle (Temporal Models)
- **CV Brier**: 0.1632
- **Models**: BiLSTM-Attention (44.6%), TFT-Lite (26.8%), TS-Embedder (24.3%), Conv1D (4.3%)
- **Output**: submission_stage2_temporal.csv

### Notebook 5: mega_ensemble_kaggle (CPU-Only Mega Ensemble)
- **CV Brier**: ~0.17
- **Models**: 13+ models trained from scratch (all sklearn/GBM, no GPU needed)
- **Output**: submission_stage2_mega.csv

### Notebook 6: creative_models_kaggle (Netflix-Inspired)
- **Models**: SVD++ for Sports, Neural Collaborative Filtering (NeuMF), Siamese Network, Mixture of Experts, Contrastive Learning, Ordinal Margin Net, NMF Game Model
- **Status**: Training incomplete (timed out locally)

### Notebook 7: ultimate_model (Best Model)
- **CV Brier**: Men's 0.0902, Women's 0.1373
- **Techniques**:
  - XGBoost regression on point differential (1st place winner technique)
  - Spline calibration (non-parametric margin -> probability)
  - LightGBM + CatBoost classifiers
  - Logistic Regression (calibration anchor)
  - Logit-space Ridge meta-learner (MetaStack approach)
  - Isotonic calibration
  - Separate Men's/Women's models (different decay rates)
  - Symmetric data augmentation (2x training data)
  - Dual Elo (regular + tournament-only)
  - Elo momentum (late season form)
  - Massey ordinals PCA (top 3 principal components)
  - GLM team quality (Bradley-Terry)
  - Four Factors of basketball
  - Inverse-Brier weighted ensemble
  - Historical seed prior
  - goto_conversion (favourite-longshot bias correction, alpha=1.15)
  - Tight clipping [0.025, 0.975]
- **Output**: submission_stage2_ultimate.csv, submission_stage2_ultimate_goto.csv

## Key Techniques

### goto_conversion (Favourite-Longshot Bias Correction)
- Pushes confident predictions further from 0.5
- Formula: `corrected_odds = odds ^ alpha` where alpha=1.15
- Won $47K+ and 10+ gold medals for its creator
- Applied as post-processing on all submission CSVs

### MetaStack (Logit-Space Blending)
- Transform predictions to logit space before meta-learning
- Ridge regression in logit space, then inverse-logit back
- Better than raw probability blending for calibration

### Separate Men's/Women's Models
- Different recency decay rates (Men's data more stable)
- Women's models exclude Massey PCA (no ordinals data)
- Improves over single combined model

### Symmetric Data Augmentation
- For each A-vs-B game, also create B-vs-A with flipped features and target
- Doubles training data, ensures model is symmetric

## Data Provided (35 CSVs) - M/W prefix for Men's/Women's
| File | Description |
|------|------------|
| MTeams / WTeams | Team IDs, names, conferences |
| MSeasons / WSeasons | Season info, region assignments, key dates |
| MRegularSeasonCompactResults | Game results (winner, loser, score, location, OT) |
| MRegularSeasonDetailedResults | + box scores: FGM/A, FG3M/A, FTM/A, OR, DR, Ast, TO, Stl, Blk, PF |
| MNCAATourneyCompactResults | Historical tournament results (compact) |
| MNCAATourneyDetailedResults | Tournament results with full box scores |
| MNCAATourneySeeds | Seedings per year (e.g., W01, X02, Y16) |
| MNCAATourneySlots | Bracket structure (which slot feeds where) |
| MConferenceTourneyGames | Conference tournament results |
| MGameCities | City/venue for each game |
| MCities | City metadata (name, state) |
| MTeamCoaches | Coach assignments by season (with date ranges) |
| MTeamConferences | Conference affiliations by season |
| MMasseyOrdinals | 100+ ranking systems (KenPom, Sagarin, RPI, etc.) |
| MSecondaryTourney* | NIT, CBI, etc. results |
| SampleSubmission* | Template submission files for Stage 1 & 2 |

## Key Rules & Constraints
- Predict probability for **every possible matchup** in both Men's & Women's tournaments
- Submissions evaluated only AFTER tournaments conclude and Kaggle rescores
- **Two submissions** must be manually selected for final scoring
- 2-5 submissions per day limit
- External data is allowed (any publicly available data)
- Predictions clipped to [0.05, 0.95] to avoid infinite loss
- Local training on laptop (GTX 1650 Ti 4GB VRAM) + Kaggle notebooks

## Critical Pitfalls to Avoid
1. **Overfitting** - Only ~63 tournament games/year; complex models fail. Keep it simple.
2. **Predicting 0 or 1** - Brier → catastrophic penalty. Always clip to [0.05, 0.95].
3. **Data leakage** - Use pre-tournament snapshots only. AP/Coaches polls, RPI may include postseason.
4. **Using team IDs as features** - Teams change completely year to year.
5. **Head-to-head records** - Tiny sample sizes, unreliable signal.
6. **Wrong team ordering** - Submission requires TeamID_low < TeamID_high.
7. **Not handling play-in games** correctly.
8. **Conference tournament games** leaking into "regular season" averages.

## Workflow Rules
- **Notebooks in notebooks/ folder** — .ipynb only (no .py)
- **Local execution copies** in `.tmp/` folder (gitignored)
- **All Unicode characters** replaced with ASCII in .tmp/ scripts (Windows cp1252 compatibility)
- **Kaggle API token** saved at `~/.kaggle/kaggle.json`
- Use `.tmp/` directory for intermediate files and generated CSVs

## Kaggle API
- **Token**: Saved at `~/.kaggle/kaggle.json`
- **Check leaderboard**: `kaggle competitions leaderboard march-machine-learning-mania-2026 -s`
- **Check submissions**: `kaggle competitions submissions march-machine-learning-mania-2026`
- **List notebooks**: `kaggle kernels list --competition march-machine-learning-mania-2026`

## References
- [Top 1% Gold 2023 Writeup](https://medium.com/@maze508/top-1-gold-kaggle-march-machine-learning-mania-2023-solution-writeup-2c0273a62a78)
- [goto_conversion winning solution](https://www.kaggle.com/code/kaito510/goto-conversion-winning-solution) - 264 votes
- [2025 1st Place Solution (modeh7)](https://www.kaggle.com/code/kacchanwriting/2025-1st-place-solution-modeh7-2026-adaptation) - XGBoost margin regression
- [MetaStack Madness Engine](https://www.kaggle.com/code/furqonaryadana/0-1471-stage-2-metastack-madness-engine-eda) - Brier 0.1471
- [Elo + Massey + Four Factors](https://www.kaggle.com/code/imaadmahmood/elo-massey-ordinals-four-factors-ensemble)
- [Starter Notebook 2026](https://www.kaggle.com/code/martynaplomecka/march-machine-learning-mania-2026-starter) - 550 votes
- [Calculate ELO-Ratings](https://www.kaggle.com/code/lennarthaupts/calculate-elo-ratings) - 277 votes
