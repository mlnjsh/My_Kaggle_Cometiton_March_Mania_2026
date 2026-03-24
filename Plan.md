# Execution Plan - March ML Mania 2026

## Timeline: March 14 - March 19, 2026 (5 days)
**Deadline: March 19, 2026**

---

## Phase 1: Data Setup & Understanding (Day 1 - March 14)
**Status**: [ ] Not Started

### Step 1.1: Download Competition Data
- [ ] Download all 35 CSVs from Kaggle competition page
- [ ] Organize data files, read documentation
- [ ] Understand file relationships and join keys

### Step 1.2: Data Inventory
- [ ] Catalog all files: rows, columns, date ranges, missing values
- [ ] Identify Men's vs Women's data separation
- [ ] Map TeamIDs across files
- [ ] Understand Massey ordinals structure (which rating systems available)
- [ ] Check sample submission format

### Step 1.3: Quick Baseline
- [ ] Create seed-only baseline (historical seed win probabilities)
- [ ] Submit to Kaggle to establish baseline Brier score
- [ ] This gives us a score to beat immediately

---

## Phase 2: Extensive EDA (Days 1-2 - March 14-15)
**Status**: [ ] Not Started
**Location**: Local (then transfer best visualizations to Kaggle notebook)

### Step 2.1: Data Quality EDA
- [ ] Missing value heatmap across all datasets
- [ ] Data type validation
- [ ] Duplicate detection
- [ ] Temporal coverage analysis (which seasons have which data)
- [ ] Join integrity checks

### Step 2.2: Target Analysis
- [ ] Tournament outcome distribution (upset rates by round)
- [ ] Seed matchup win probability matrix (1v16, 2v15, ... 8v9)
- [ ] Score distributions in tournament vs regular season
- [ ] Historical Brier score analysis of seed-only predictions

### Step 2.3: Feature Exploration
- [ ] Win% distribution by season and conference
- [ ] Point differential distributions and trends
- [ ] Offensive vs Defensive efficiency scatter plots
- [ ] Pace of play analysis
- [ ] Free throw rate, turnover rate, rebound rate distributions
- [ ] Four Factors analysis (eFG%, TO%, ORB%, FTR)

### Step 2.4: Temporal Patterns
- [ ] Year-over-year stability of team performance
- [ ] Conference strength trends (which conferences dominate when)
- [ ] Upset frequency trends over time
- [ ] Regular season vs tournament performance correlation
- [ ] Late season momentum analysis (last 10 games vs season)

### Step 2.5: Advanced Insights
- [ ] Massey ordinals comparison: which systems predict best?
- [ ] Geographic/travel distance impact on outcomes
- [ ] Coach experience vs tournament success
- [ ] Men's vs Women's tournament pattern differences
- [ ] Correlation matrix of top 30 features vs tournament wins
- [ ] Feature stability analysis (which features are consistent predictors)

### Step 2.6: Publishable EDA Notebook
- [ ] Compile all insights into a clean Kaggle notebook
- [ ] 30+ publication-quality visualizations
- [ ] Written narrative explaining each finding
- [ ] Feature recommendation summary for modeling

---

## Phase 3: Feature Engineering (Day 2-3 - March 15-16)
**Status**: [ ] Not Started

### Step 3.1: Core Features
- [ ] Seed difference (TeamA_seed - TeamB_seed)
- [ ] Win percentage (overall, conference, last 10)
- [ ] Point differential (season avg margin of victory)
- [ ] Offensive/Defensive rating (points per 100 possessions)
- [ ] Net rating

### Step 3.2: Elo Rating System
- [ ] Build custom Elo from scratch using all historical games
- [ ] Tune K-factor (try 20, 32, 40)
- [ ] Add margin-of-victory adjustment
- [ ] Compute end-of-regular-season Elo for each team
- [ ] Elo difference as feature

### Step 3.3: External Ratings (Massey Ordinals)
- [ ] Extract KenPom ratings (if available in ordinals)
- [ ] Extract Sagarin ratings
- [ ] Extract BPI, RPI, NET rankings
- [ ] Compute rating differences for each matchup
- [ ] Consensus ranking (average across systems)

### Step 3.4: Advanced Features
- [ ] Four Factors: eFG%, TOV%, ORB%, FT/FGA (and opponent versions)
- [ ] Strength of schedule (avg opponent win%, avg opponent Elo)
- [ ] Consistency metrics (std dev of point differential)
- [ ] Road/away performance split
- [ ] Performance vs Top-25 teams
- [ ] Conference tournament result (won conf tourney = momentum?)
- [ ] Rolling features: last 5, 10, 14 game averages

### Step 3.5: Interaction Features
- [ ] seed_diff * elo_diff
- [ ] seed_diff * off_rating_diff
- [ ] Historical seed matchup win probability
- [ ] Conference matchup history

### Step 3.6: Feature Selection
- [ ] Recursive Feature Elimination (RFE) with cross-validation
- [ ] SHAP-based importance ranking
- [ ] Correlation-based redundancy removal
- [ ] Select top 30-50 features for final models

---

## Phase 4: Modeling - Traditional ML (Day 3 - March 16)
**Status**: [ ] Not Started
**Location**: Kaggle Notebook

### Step 4.1: Cross-Validation Strategy
- [ ] Implement leave-one-season-out CV (2015-2025)
- [ ] Alternatively: expanding window CV (train 2003-2014, val 2015, etc.)
- [ ] Track both Brier score and log loss per fold

### Step 4.2: Logistic Regression
- [ ] Train LR on seed_diff only → baseline
- [ ] Train LR on top 10 features → improved baseline
- [ ] Train LR on full feature set with L1/L2 regularization
- [ ] Record CV Brier scores

### Step 4.3: XGBoost
- [ ] Baseline: n_estimators=100, lr=0.05, max_depth=4
- [ ] Hyperparameter tuning with Optuna (50-100 trials)
  - n_estimators: [100, 300, 500]
  - learning_rate: [0.01, 0.05, 0.1]
  - max_depth: [3, 4, 5, 6]
  - subsample: [0.7, 0.8, 0.9]
  - colsample_bytree: [0.7, 0.8, 0.9]
- [ ] SHAP analysis on best model
- [ ] Calibrate with isotonic regression

### Step 4.4: LightGBM
- [ ] Same tuning approach as XGBoost
- [ ] Compare with XGBoost results
- [ ] Calibrate probabilities

### Step 4.5: CatBoost
- [ ] Train with default parameters (handles categoricals natively)
- [ ] Tune iterations, depth, learning_rate
- [ ] Calibrate probabilities

---

## Phase 5: Modeling - Deep Learning (Days 3-4 - March 16-17)
**Status**: [ ] Not Started
**Location**: Kaggle Notebook (GPU)

### Step 5.1: LSTM Model
- [ ] Architecture: 2-layer LSTM, hidden_dim=64-128
- [ ] Input: sequence of season game features per team
- [ ] Loss: Brier loss (MSE on probabilities)
- [ ] Training: 50 epochs, AdamW, lr=1e-3, batch=64
- [ ] Early stopping patience=15
- [ ] Target: Brier < 0.16

### Step 5.2: Transformer Model
- [ ] Architecture: 2-4 attention heads, 2 layers, d_model=64
- [ ] Input: team season feature sequences
- [ ] Loss: BCE (for discrimination) + try Brier
- [ ] Training: 50 epochs, AdamW, lr=1e-4, cosine schedule
- [ ] Target: AUC > 0.84

### Step 5.3: Temporal Fusion Transformer
- [ ] Use pytorch-forecasting library
- [ ] Static features: seed, conference, coach experience
- [ ] Time-varying known: game schedule, opponent seed
- [ ] Time-varying unknown: scores, stats
- [ ] Training: 100 epochs, lr=1e-3
- [ ] Interpret attention weights for insights

### Step 5.4: Foundation Models (If Time Permits)
- [ ] Chronos-2: frame team performance as time series
- [ ] TimesFM: zero-shot forecasting experiment
- [ ] Use as additional ensemble component

---

## Phase 6: Ensemble & Calibration (Day 4 - March 17-18)
**Status**: [ ] Not Started

### Step 6.1: Model Diversity Check
- [ ] Correlation matrix of model predictions
- [ ] Ensure models disagree on enough games to benefit from ensembling
- [ ] Identify which models are strong on upsets vs chalk

### Step 6.2: Ensemble Methods
- [ ] Simple average of top 3-5 models
- [ ] Weighted average (optimize weights via scipy.minimize on CV Brier)
- [ ] Stacking: LogisticRegression meta-learner on OOF predictions
- [ ] Rank averaging

### Step 6.3: Final Calibration
- [ ] Plot calibration curves for ensemble
- [ ] Apply isotonic regression if needed
- [ ] Verify Brier score improvement
- [ ] Reliability diagram check

### Step 6.4: Separate Men's & Women's Treatment
- [ ] Evaluate if separate models per gender improve score
- [ ] Women's tournament may have different upset patterns
- [ ] Consider separate Elo systems for Men's and Women's

---

## Phase 7: Submission Strategy (Day 5 - March 18-19)
**Status**: [ ] Not Started

### Step 7.1: Backtesting
- [ ] Simulate predictions on 2015-2025 tournaments
- [ ] Compute Brier score for each year
- [ ] Identify which model/ensemble was best in each year
- [ ] Check variance across years

### Step 7.2: Create Submissions
- [ ] Submission 1 (Conservative): Blend of seed-prior + best ensemble
  - Weighted 30% seed-based + 70% model-based
  - Lower variance, good floor
- [ ] Submission 2 (Aggressive): Pure model ensemble
  - Best performing ensemble from backtesting
  - Higher variance, higher ceiling

### Step 7.3: Validation Checks
- [ ] All probabilities between 0.05 and 0.95 (avoid extreme predictions)
- [ ] Clip predictions: max(0.05, min(0.95, pred))
- [ ] Verify all required matchups present in submission
- [ ] Verify CSV format matches sample submission exactly
- [ ] Cross-check: do predictions align with seed expectations?

### Step 7.4: Submit
- [ ] Submit both CSVs to Kaggle
- [ ] Select 2 submissions for final scoring
- [ ] Document which models/weights went into each submission

---

## Notebook Structure (for Kaggle)

### Notebook 1: EDA (Public, Medal-worthy)
```
1. Data Loading & Overview
2. Data Quality Assessment
3. Tournament Seed Analysis
4. Upset Pattern Analysis
5. Four Factors Analysis
6. Temporal Trends
7. Conference Analysis
8. Geographic Analysis
9. Massey Ordinals Comparison
10. Men's vs Women's Comparison
11. Feature Correlation Study
12. Key Insights & Recommendations
```

### Notebook 2: Feature Engineering + Baseline Models
```
1. Feature Engineering Pipeline
2. Elo Rating System
3. External Ratings Integration
4. Feature Selection
5. Logistic Regression Baseline
6. XGBoost/LightGBM Training
7. Cross-Validation Results
```

### Notebook 3: Advanced Models + Ensemble
```
1. LSTM/Transformer Training
2. TFT Training
3. Ensemble Construction
4. Probability Calibration
5. Backtesting
6. Final Submission Generation
```

---

## Risk Mitigation
| Risk | Mitigation |
|------|-----------|
| Kaggle GPU quota exhausted | Prioritize GBM models (CPU), limit DL experiments |
| Overfitting to historical data | Expanding window CV, ensemble diversity |
| Poor calibration | Isotonic regression, clip predictions [0.05, 0.95] |
| Women's data too sparse | Use Men's model with transfer, or combine datasets |
| Time pressure (5 days) | Skip Phase 5.4 (foundation models) if behind schedule |
| Data leakage | Only use regular season data for features, never tournament |

## Daily Goals
| Day | Date | Focus | Deliverable |
|-----|------|-------|------------|
| 1 | Mar 14 | Data + Start EDA | Data inventory, baseline submission |
| 2 | Mar 15 | EDA + Features | Complete EDA notebook, feature pipeline |
| 3 | Mar 16 | Modeling (ML) | Tuned XGBoost/LightGBM, CV scores |
| 4 | Mar 17 | Modeling (DL) + Ensemble | DL models, ensemble, calibration |
| 5 | Mar 18-19 | Submit | Final submissions, backtest validation |
