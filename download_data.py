"""
Download Competition Data
=========================
Run this script to download data from Kaggle.
Requires: kaggle CLI installed + API token configured.

Setup:
  1. pip install kaggle
  2. Place kaggle.json in ~/.kaggle/ (or %USERPROFILE%\.kaggle\ on Windows)
  3. python download_data.py
"""

import os
import subprocess
from pathlib import Path

COMPETITION = "march-machine-learning-mania-2026"
DATA_DIR = Path(__file__).parent / "data" / "raw"


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {COMPETITION} data to {DATA_DIR}...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA_DIR)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        print("\nMake sure:")
        print("  1. kaggle is installed: pip install kaggle")
        print("  2. API token is at ~/.kaggle/kaggle.json")
        print("  3. You've accepted the competition rules on kaggle.com")
        return False

    print(result.stdout)

    # Unzip if needed
    zip_files = list(DATA_DIR.glob("*.zip"))
    if zip_files:
        import zipfile
        for zf in zip_files:
            print(f"Extracting {zf.name}...")
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(DATA_DIR)
            zf.unlink()  # Remove zip after extraction

    # List downloaded files
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    print(f"\nDownloaded {len(csv_files)} CSV files:")
    for f in csv_files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")

    return True


if __name__ == "__main__":
    download()
