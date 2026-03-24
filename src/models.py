"""
Model Training & Evaluation
============================
Train, tune, calibrate, and ensemble models for tournament prediction.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from configs.config import (
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS, LOGREG_PARAMS,
    CV_CONFIG, PRED_CLIP_MIN, PRED_CLIP_MAX, SEED
)


def brier_score(y_true, y_pred):
    """Compute Brier score (lower is better)."""
    return np.mean((y_true - y_pred) ** 2)


def clip_predictions(preds):
    """Clip predictions to avoid extreme probabilities."""
    return np.clip(preds, PRED_CLIP_MIN, PRED_CLIP_MAX)


def leave_one_season_out_cv(X, y, seasons, model_fn, feature_cols):
    """
    Leave-one-season-out cross-validation.

    Parameters
    ----------
    X : pd.DataFrame - features + Season column
    y : array-like - target
    seasons : array-like - season for each row
    model_fn : callable - returns (model, name) tuple
    feature_cols : list - feature column names

    Returns
    -------
    dict with per-fold and overall scores
    """
    val_seasons = CV_CONFIG["val_seasons"]
    results = []
    oof_preds = np.zeros(len(y))
    oof_mask = np.zeros(len(y), dtype=bool)

    for val_season in val_seasons:
        train_mask = seasons < val_season
        val_mask = seasons == val_season

        if val_mask.sum() == 0:
            continue

        X_train = X.loc[train_mask, feature_cols]
        X_val = X.loc[val_mask, feature_cols]
        y_train = y[train_mask]
        y_val = y[val_mask]

        model, name = model_fn()
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_val)[:, 1]
        preds = clip_predictions(preds)

        bs = brier_score(y_val, preds)
        ll = log_loss(y_val, preds)

        results.append({
            "season": val_season,
            "n_games": val_mask.sum(),
            "brier_score": bs,
            "log_loss": ll,
        })

        oof_preds[val_mask] = preds
        oof_mask |= val_mask

    overall_brier = brier_score(y[oof_mask], oof_preds[oof_mask])
    overall_logloss = log_loss(y[oof_mask], oof_preds[oof_mask])

    return {
        "per_fold": pd.DataFrame(results),
        "overall_brier": overall_brier,
        "overall_logloss": overall_logloss,
        "oof_preds": oof_preds,
        "oof_mask": oof_mask,
    }


def get_xgboost_model():
    """Create XGBoost classifier with default params."""
    from xgboost import XGBClassifier
    params = XGBOOST_PARAMS.copy()
    return XGBClassifier(**params), "XGBoost"


def get_lightgbm_model():
    """Create LightGBM classifier with default params."""
    from lightgbm import LGBMClassifier
    params = LIGHTGBM_PARAMS.copy()
    return LGBMClassifier(**params), "LightGBM"


def get_catboost_model():
    """Create CatBoost classifier with default params."""
    from catboost import CatBoostClassifier
    params = CATBOOST_PARAMS.copy()
    return CatBoostClassifier(**params), "CatBoost"


def get_logreg_model():
    """Create Logistic Regression with default params."""
    params = LOGREG_PARAMS.copy()
    return LogisticRegression(**params), "LogisticRegression"


def ensemble_predictions(model_preds, weights=None):
    """
    Ensemble multiple model predictions via weighted average.

    Parameters
    ----------
    model_preds : dict of {name: array} - predictions from each model
    weights : dict of {name: float}, optional - model weights (normalized internally)

    Returns
    -------
    array of ensemble predictions
    """
    names = list(model_preds.keys())
    preds_matrix = np.column_stack([model_preds[n] for n in names])

    if weights is None:
        w = np.ones(len(names)) / len(names)
    else:
        w = np.array([weights.get(n, 1.0) for n in names])
        w = w / w.sum()

    ensemble = preds_matrix @ w
    return clip_predictions(ensemble)


def optimize_ensemble_weights(model_preds, y_true, metric="brier"):
    """
    Optimize ensemble weights using scipy.minimize.

    Parameters
    ----------
    model_preds : dict of {name: array}
    y_true : array
    metric : str - "brier" or "logloss"
    """
    from scipy.optimize import minimize

    names = list(model_preds.keys())
    preds_matrix = np.column_stack([model_preds[n] for n in names])
    n_models = len(names)

    def objective(weights):
        w = weights / weights.sum()
        ensemble = clip_predictions(preds_matrix @ w)
        if metric == "brier":
            return brier_score(y_true, ensemble)
        return log_loss(y_true, ensemble)

    # Start with equal weights
    x0 = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method="SLSQP")
    optimal_weights = dict(zip(names, result.x))

    return optimal_weights, result.fun
