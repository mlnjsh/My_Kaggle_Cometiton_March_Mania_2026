"""
Data Loading Utilities
======================
Load all competition CSVs into a dictionary of DataFrames.
Works on both local and Kaggle environments.
"""

import pandas as pd
from pathlib import Path
from configs.config import DATA_DIR, DATA_FILES, IS_KAGGLE


def load_all_data(verbose=True):
    """Load all competition CSV files into a dict of DataFrames."""
    data = {}
    missing = []

    for key, filename in DATA_FILES.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
            if verbose:
                print(f"  Loaded {key}: {data[key].shape}")
        else:
            missing.append(filename)
            if verbose:
                print(f"  MISSING: {filename}")

    if verbose:
        print(f"\nLoaded {len(data)}/{len(DATA_FILES)} files")
        if missing:
            print(f"Missing files: {missing}")

    return data


def load_single(key):
    """Load a single CSV by its config key."""
    filename = DATA_FILES[key]
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found. Download data first.")
    return pd.read_csv(filepath)


def get_data_summary(data):
    """Print summary statistics for all loaded datasets."""
    summary = []
    for key, df in data.items():
        summary.append({
            "dataset": key,
            "rows": len(df),
            "cols": len(df.columns),
            "columns": ", ".join(df.columns[:5]) + ("..." if len(df.columns) > 5 else ""),
            "missing_pct": f"{df.isnull().sum().sum() / df.size * 100:.1f}%",
        })
    return pd.DataFrame(summary)
