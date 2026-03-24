# March Machine Learning Mania 2026

**Team Chai and Samosa** | Kaggle Competition | $50,000 Prize Pool

---

Forecasting the 2026 NCAA Division 1 Men's & Women's Basketball Tournaments using an ensemble of 30+ models spanning gradient boosting, deep learning, temporal architectures, and collaborative filtering.

## Results

| Metric | Men's | Women's |
|--------|-------|---------|
| Best CV Brier | **0.0902** | **0.1373** |

> Lower Brier score = better. Predictions clipped to [0.025, 0.975].

## Model Pipeline

```
Raw Data (35 CSVs)
    |
    v
Feature Engineering
  - Dual Elo (regular + tournament-only)
  - Elo Momentum (late-season form)
  - Four Factors of Basketball
  - Massey Ordinals PCA (top 3 components)
  - GLM Team Quality (Bradley-Terry)
  - Symmetric Data Augmentation (2x training data)
    |
    v
+---------------------------+---------------------------+---------------------------+
|   Base Ensemble           |   Deep Learning           |   Temporal Models         |
|   - Logistic Regression   |   - MatchupMLP            |   - BiLSTM-Attention      |
|   - XGBoost (tuned)       |   - TransformerMatchup    |   - TFT-Lite              |
|   - LightGBM              |   - LSTMMatchup           |   - TS-Embedder           |
|   - CatBoost              |   - Hybrid GBMs           |   - Conv1D Temporal       |
|   - Elo Rating            |   - Team Autoencoder      |                           |
+---------------------------+---------------------------+---------------------------+
|   Expert Models           |   Netflix-Inspired        |   Ultimate Model          |
|   - Bradley-Terry         |   - SVD++ for Sports      |   - XGBoost Margin Reg    |
|   - Random Forest         |   - NeuMF                 |   - Spline Calibration    |
|   - Extra Trees           |   - Siamese Network       |   - Logit-Space Ridge     |
|   - SVM-RBF               |   - Mixture of Experts    |     Meta-Learner          |
|   - TabNet                |   - Contrastive Learning  |   - Isotonic Calibration  |
|   - Wide & Deep           |   - Ordinal Margin Net    |   - Inverse-Brier         |
|   - Gaussian Process      |   - NMF Game Model        |     Weighted Ensemble     |
|   + 6 more                |                           |                           |
+---------------------------+---------------------------+---------------------------+
    |
    v
Meta-Ensemble (Logit-Space Ridge Stacking)
    |
    v
Post-Processing
  - goto_conversion (favourite-longshot bias, alpha=1.15)
  - Historical seed prior blending
  - Tight clipping [0.025, 0.975]
    |
    v
Final Submission (2026 bracket probabilities)
```

## Key Techniques

| Technique | Description |
|-----------|-------------|
| **goto_conversion** | Favourite-longshot bias correction (`odds^1.15`). Won $47K+ and 10+ gold medals for its creator |
| **MetaStack** | Logit-space Ridge regression for ensemble blending -- better calibration than raw probability averaging |
| **Separate M/W Models** | Different recency decay rates; women's models exclude Massey PCA (no ordinals data) |
| **Symmetric Augmentation** | Each A-vs-B game mirrored as B-vs-A with flipped features, doubling training data |
| **XGBoost Margin Regression** | Predict point differential first, then convert to probability via spline calibration (2025 1st place technique) |
| **Dual Elo** | Separate Elo ratings for regular season and tournament performance |

## Project Structure

```
.
|-- notebooks/                 # Kaggle-ready .ipynb notebooks (7 notebooks)
|   |-- eda_kaggle.ipynb              # Exploratory data analysis
|   |-- train_kaggle.ipynb            # Base ensemble (LR, XGB, LGBM, CatBoost, Elo)
|   |-- advanced_models_kaggle.ipynb  # Deep learning (MLP, Transformer, LSTM, AE)
|   |-- expert_models_kaggle.ipynb    # 13 specialized models
|   |-- foundation_models_kaggle.ipynb # Temporal models (BiLSTM, TFT, Conv1D)
|   |-- mega_ensemble_kaggle.ipynb    # CPU-only mega ensemble (13+ models)
|   |-- creative_models_kaggle.ipynb  # Netflix-inspired collaborative filtering
|-- src/                       # Reusable source modules
|-- configs/                   # Configuration files
|-- data/                      # Raw + processed data (gitignored)
|-- requirements.txt
```

## Submissions

7 submissions uploaded to Kaggle with varying ensemble strategies:

| # | Submission | CV Brier | Strategy |
|---|-----------|----------|----------|
| 1 | Ultimate + goto | **0.09 / 0.14** | Best ensemble + bias correction |
| 2 | Ultimate | **0.09 / 0.14** | Best ensemble (no goto) |
| 3 | Mega + goto | ~0.17 | 13+ model ensemble + bias correction |
| 4 | Mega | ~0.17 | 13+ model ensemble |
| 5 | Temporal + goto | 0.1632 | BiLSTM/TFT ensemble + bias correction |
| 6 | Temporal | 0.1632 | BiLSTM/TFT ensemble |
| 7 | Conservative | 0.1687 | Original weighted blend |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download competition data
python download_data.py

# Run notebooks in order via Jupyter or upload to Kaggle
```

## References

- [goto_conversion winning solution](https://www.kaggle.com/code/kaito510/goto-conversion-winning-solution) -- 264 votes
- [2025 1st Place Solution](https://www.kaggle.com/code/kacchanwriting/2025-1st-place-solution-modeh7-2026-adaptation) -- XGBoost margin regression
- [MetaStack Madness Engine](https://www.kaggle.com/code/furqonaryadana/0-1471-stage-2-metastack-madness-engine-eda) -- Brier 0.1471
- [Elo + Massey + Four Factors](https://www.kaggle.com/code/imaadmahmood/elo-massey-ordinals-four-factors-ensemble)

## Competition

- **Competition**: [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
- **Evaluation**: Brier Score (MSE between predicted probabilities and actual 0/1 outcomes)
- **Format**: Predict P(lower TeamID wins) for every possible tournament matchup

---

*Built with lots of chai and samosas*
