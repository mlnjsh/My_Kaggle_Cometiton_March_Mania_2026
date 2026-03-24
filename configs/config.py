"""
March ML Mania 2026 - Central Configuration
============================================
All paths, hyperparameters, feature lists, and constants in one place.
"""

import os
from pathlib import Path

# =============================================================
# ENVIRONMENT DETECTION
# =============================================================
IS_KAGGLE = os.path.exists("/kaggle/input")

# =============================================================
# PATHS
# =============================================================
if IS_KAGGLE:
    # Kaggle notebook paths
    DATA_DIR = Path("/kaggle/input/march-machine-learning-mania-2026")
    OUTPUT_DIR = Path("/kaggle/working")
    MODEL_DIR = Path("/kaggle/working/models")
    SUBMISSION_DIR = Path("/kaggle/working")
else:
    # Local paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    OUTPUT_DIR = PROJECT_ROOT / ".tmp"
    MODEL_DIR = PROJECT_ROOT / "models"
    SUBMISSION_DIR = PROJECT_ROOT / "submissions"

# =============================================================
# COMPETITION CONSTANTS
# =============================================================
COMPETITION_NAME = "march-machine-learning-mania-2026"
CURRENT_SEASON = 2026
PREDICTION_YEAR = 2026

# Team ID ranges
MENS_TEAM_ID_MIN = 1000
MENS_TEAM_ID_MAX = 1999
WOMENS_TEAM_ID_MIN = 3000
WOMENS_TEAM_ID_MAX = 3999

# Prediction clipping (avoid extreme probabilities)
PRED_CLIP_MIN = 0.05
PRED_CLIP_MAX = 0.95

# =============================================================
# DATA FILES (provided by competition)
# =============================================================
DATA_FILES = {
    # Men's data
    "m_teams": "MTeams.csv",
    "m_seasons": "MSeasons.csv",
    "m_regular_compact": "MRegularSeasonCompactResults.csv",
    "m_regular_detailed": "MRegularSeasonDetailedResults.csv",
    "m_tourney_compact": "MNCAATourneyCompactResults.csv",
    "m_tourney_detailed": "MNCAATourneyDetailedResults.csv",
    "m_tourney_seeds": "MNCAATourneySeeds.csv",
    "m_tourney_slots": "MNCAATourneySlots.csv",
    "m_conf_tourney": "MConferenceTourneyGames.csv",
    "m_game_cities": "MGameCities.csv",
    "m_team_coaches": "MTeamCoaches.csv",
    "m_team_conferences": "MTeamConferences.csv",
    "m_massey_ordinals": "MMasseyOrdinals.csv",
    "m_secondary_compact": "MSecondaryTourneyCompactResults.csv",
    "m_secondary_detailed": "MSecondaryTourneyDetailedResults.csv",
    # Women's data
    "w_teams": "WTeams.csv",
    "w_seasons": "WSeasons.csv",
    "w_regular_compact": "WRegularSeasonCompactResults.csv",
    "w_regular_detailed": "WRegularSeasonDetailedResults.csv",
    "w_tourney_compact": "WNCAATourneyCompactResults.csv",
    "w_tourney_detailed": "WNCAATourneyDetailedResults.csv",
    "w_tourney_seeds": "WNCAATourneySeeds.csv",
    "w_tourney_slots": "WNCAATourneySlots.csv",
    "w_conf_tourney": "WConferenceTourneyGames.csv",
    "w_game_cities": "WGameCities.csv",
    "w_team_coaches": "WTeamCoaches.csv",
    "w_team_conferences": "WTeamConferences.csv",
    "w_massey_ordinals": "WMasseyOrdinals.csv",
    "w_secondary_compact": "WSecondaryTourneyCompactResults.csv",
    "w_secondary_detailed": "WSecondaryTourneyDetailedResults.csv",
    # Shared
    "cities": "Cities.csv",
    "sample_sub_stage1": "SampleSubmissionStage1.csv",
    "sample_sub_stage2": "SampleSubmissionStage2.csv",
}

# =============================================================
# FEATURE CONFIGURATION
# =============================================================

# Core features (always included)
CORE_FEATURES = [
    "seed_diff",
    "elo_diff",
    "win_pct_diff",
    "point_diff_avg_diff",
    "off_rating_diff",
    "def_rating_diff",
    "net_rating_diff",
]

# Four Factors (Dean Oliver)
FOUR_FACTORS = [
    "efg_pct_diff",       # Effective FG%
    "tov_pct_diff",       # Turnover rate
    "orb_pct_diff",       # Offensive rebound %
    "ft_rate_diff",       # Free throw rate (FTA/FGA)
]

# Advanced features
ADVANCED_FEATURES = [
    "pace_diff",               # Possessions per game
    "sos_diff",                # Strength of schedule
    "consistency_diff",        # Std dev of point differential
    "last10_win_pct_diff",     # Last 10 games momentum
    "road_win_pct_diff",       # Away performance
    "vs_top25_win_pct_diff",   # Performance vs ranked teams
    "conf_tourney_wins_diff",  # Conference tournament performance
    "coach_tourney_exp_diff",  # Coach tournament experience
]

# External rating features (from Massey Ordinals)
EXTERNAL_RATINGS = [
    "pom_rating_diff",  # KenPom (Pomeroy)
    "sag_rating_diff",  # Sagarin
    "rpi_rating_diff",  # RPI
    "mor_rating_diff",  # Moore
    "consensus_rank_diff",  # Average across all systems
]

# Interaction features
INTERACTION_FEATURES = [
    "seed_x_elo_diff",
    "seed_x_off_rating_diff",
    "historical_seed_matchup_win_pct",
]

# All features combined
ALL_FEATURES = (
    CORE_FEATURES
    + FOUR_FACTORS
    + ADVANCED_FEATURES
    + EXTERNAL_RATINGS
    + INTERACTION_FEATURES
)

# =============================================================
# ELO CONFIGURATION
# =============================================================
ELO_CONFIG = {
    "initial_rating": 1500,
    "k_factor": 32,             # Base K-factor
    "k_factor_tournament": 40,  # Higher K for tournament games
    "home_advantage": 100,      # Elo points for home court
    "margin_multiplier": 0.006, # Margin-of-victory adjustment
    "season_reversion": 0.25,   # Revert 25% to mean between seasons
    "start_season": 2003,       # First season with detailed data
}

# =============================================================
# MODEL HYPERPARAMETERS
# =============================================================

# XGBoost defaults (to be tuned with Optuna)
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
}

# LightGBM defaults
LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary",
    "metric": "binary_logloss",
    "verbose": -1,
    "random_state": 42,
}

# CatBoost defaults
CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 5,
    "l2_leaf_reg": 3.0,
    "border_count": 254,
    "loss_function": "Logloss",
    "eval_metric": "BrierScore",
    "random_seed": 42,
    "verbose": 0,
}

# Logistic Regression
LOGREG_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
}

# Optuna tuning
OPTUNA_CONFIG = {
    "n_trials": 100,
    "timeout": 1800,  # 30 minutes max
    "metric": "neg_brier_score",
}

# =============================================================
# DEEP LEARNING HYPERPARAMETERS (Kaggle GPU)
# =============================================================

LSTM_CONFIG = {
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 50,
    "patience": 15,       # Early stopping
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "loss": "brier",      # MSE on probabilities
}

TRANSFORMER_CONFIG = {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.2,
    "learning_rate": 1e-4,
    "batch_size": 64,
    "epochs": 50,
    "patience": 15,
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "loss": "bce",
}

TFT_CONFIG = {
    "hidden_size": 64,
    "attention_head_size": 4,
    "dropout": 0.2,
    "hidden_continuous_size": 32,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 100,
    "patience": 20,
    "gradient_clip_val": 0.1,
    "reduce_on_plateau_patience": 5,
    # TFT-specific feature types
    "static_covariates": ["conference", "coach_tourney_exp", "program_strength"],
    "known_future_inputs": ["seed", "opponent_seed", "round_num", "location"],
    "past_observed_inputs": ["score", "opp_score", "fg_pct", "reb", "ast", "to",
                             "rolling_elo", "rolling_off_rating", "rolling_def_rating"],
}

# =============================================================
# FOUNDATION MODEL CONFIGS (Tier 3 - Experimental)
# =============================================================

CHRONOS_CONFIG = {
    "model_id": "amazon/chronos-t5-base",  # 200M params
    "mode": "zero_shot",          # or "fine_tune"
    "fine_tune_epochs": 15,
    "fine_tune_lr": 5e-5,
    "prediction_length": 1,
    "context_length": 30,         # ~30 games of history
    "batch_size": 32,
    "device": "cuda",             # T4 sufficient for Base
}

TIMESFM_CONFIG = {
    "model_id": "google/timesfm-1.0-200m",
    "mode": "zero_shot",
    "context_length": 32,
    "prediction_length": 1,
    "batch_size": 32,
}

MOIRAI_CONFIG = {
    "model_id": "salesforce/moirai-1.0-R-base",  # 91M params
    "mode": "zero_shot",
    "context_length": 30,
    "prediction_length": 1,
    "batch_size": 32,
}

# =============================================================
# CROSS-VALIDATION
# =============================================================
CV_CONFIG = {
    "strategy": "leave_one_season_out",
    "train_start_season": 2003,
    "train_end_season": 2025,
    "val_seasons": list(range(2015, 2026)),  # Validate on 2015-2025
    "n_splits": 11,  # One per validation season
}

# =============================================================
# ENSEMBLE CONFIGURATION
# =============================================================
ENSEMBLE_CONFIG = {
    # Three-tier ensemble architecture
    "tier1_core": {
        "models": ["lightgbm", "catboost", "xgboost", "logreg", "elo"],
        "weight": 0.60,
        "description": "GBM + statistical models on engineered features",
    },
    "tier2_deep_learning": {
        "models": ["tft", "lstm", "transformer"],
        "weight": 0.25,
        "description": "Temporal models capturing season trajectories",
    },
    "tier3_foundation": {
        "models": ["chronos", "timesfm"],
        "weight": 0.15,
        "description": "Zero-shot foundation model signals",
    },
    "meta_learner": "logreg",     # Combines all tiers
    "calibration": "isotonic",     # or "platt", "none"
    "optimize_weights": True,
    "weight_bounds": (0.0, 1.0),
}

# Submission strategy
SUBMISSION_STRATEGY = {
    "conservative": {
        "description": "Blend of seed-prior + model ensemble",
        "seed_weight": 0.30,
        "model_weight": 0.70,
    },
    "aggressive": {
        "description": "Pure model ensemble, higher ceiling",
        "seed_weight": 0.0,
        "model_weight": 1.0,
    },
}

# =============================================================
# KAGGLE COMPUTE LIMITS
# =============================================================
KAGGLE_LIMITS = {
    "gpu_quota_hours_per_week": 30,
    "session_max_hours": 12,
    "gpu_type": "P100 (16GB) or T4 (16GB)",
    "ram_gb": 16,
    "disk_gb": 73,
    "cpu_cores": 4,
    "internet_access": True,  # During editing, disabled during commit
}

# =============================================================
# RANDOM SEEDS
# =============================================================
SEED = 42
