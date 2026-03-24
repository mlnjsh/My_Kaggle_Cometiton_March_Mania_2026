"""
Submission Generator
====================
Generate properly formatted Kaggle submission files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from configs.config import (
    DATA_DIR, SUBMISSION_DIR, PREDICTION_YEAR,
    PRED_CLIP_MIN, PRED_CLIP_MAX, DATA_FILES
)


def load_sample_submission(stage=2):
    """Load the sample submission file for the given stage."""
    key = f"sample_sub_stage{stage}"
    filepath = DATA_DIR / DATA_FILES[key]
    return pd.read_csv(filepath)


def parse_submission_id(id_str):
    """Parse '2026_1101_1437' into (year, team_a, team_b)."""
    parts = id_str.split("_")
    return int(parts[0]), int(parts[1]), int(parts[2])


def generate_submission(predict_fn, stage=2, filename=None):
    """
    Generate a submission CSV.

    Parameters
    ----------
    predict_fn : callable
        Function that takes (team_a_id, team_b_id, season) and returns P(team_a wins).
    stage : int
        1 or 2.
    filename : str, optional
        Output filename. Defaults to 'submission_stage{stage}.csv'.
    """
    sample = load_sample_submission(stage)
    predictions = []

    for _, row in sample.iterrows():
        year, team_a, team_b = parse_submission_id(row["ID"])
        pred = predict_fn(team_a, team_b, year)
        pred = np.clip(pred, PRED_CLIP_MIN, PRED_CLIP_MAX)
        predictions.append(pred)

    sample["Pred"] = predictions

    if filename is None:
        filename = f"submission_stage{stage}.csv"

    output_path = SUBMISSION_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path} ({len(sample)} matchups)")

    # Validation
    validate_submission(sample)
    return sample


def validate_submission(sub_df):
    """Validate submission format and values."""
    errors = []

    if "ID" not in sub_df.columns or "Pred" not in sub_df.columns:
        errors.append("Missing required columns: ID, Pred")

    if sub_df["Pred"].isnull().any():
        errors.append(f"Found {sub_df['Pred'].isnull().sum()} null predictions")

    if (sub_df["Pred"] < 0).any() or (sub_df["Pred"] > 1).any():
        errors.append("Predictions outside [0, 1] range")

    extreme = ((sub_df["Pred"] < PRED_CLIP_MIN) | (sub_df["Pred"] > PRED_CLIP_MAX)).sum()
    if extreme > 0:
        errors.append(f"WARNING: {extreme} predictions outside [{PRED_CLIP_MIN}, {PRED_CLIP_MAX}]")

    if errors:
        for e in errors:
            print(f"  VALIDATION: {e}")
    else:
        print("  Submission validation passed!")
        print(f"  Pred stats: mean={sub_df['Pred'].mean():.4f}, "
              f"std={sub_df['Pred'].std():.4f}, "
              f"min={sub_df['Pred'].min():.4f}, "
              f"max={sub_df['Pred'].max():.4f}")
