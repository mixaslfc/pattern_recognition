
"""
- Loading the .npy (user, item, rating, timestamp)
- Computes unique users (U) and items (I)
- Saves counts to tables/summary_counts.csv
"""

import argparse
import pandas as pd

import pandas as pd
import os
from pathlib import Path
import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CH1 = Path(__file__).resolve().parent
TABLES_DIR = CH1 / "tables"
FIGS_DIR = CH1 / "figs"
ARTIFACTS_DIR = CH1 / "artifacts"
for d in (TABLES_DIR, FIGS_DIR, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    # print(path)
    # print(CH1)
    # print(os.path.exists(path))
    path = CH1.parent / "data" / "Dataset.npy"
    print(f"Loading data from: {path}")
    print(f"Exists: {os.path.exists(path)}")
    
    if os.path.exists(path):
        arr = np.load(path, allow_pickle=True)
        df = pd.DataFrame([entry.split(',') for entry in arr], columns=['user', 'item', 'rating', 'timestamp'])
    else:
        data = []
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["user", "item", "rating"]).reset_index(drop=True)
    df["user"] = df["user"].astype(str)
    df["item"] = df["item"].astype(str)
    return df
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="Dataset.npy")
    args = parser.parse_args()

    df = load_dataset(args.data)

    users = df["user"].unique()
    items = df["item"].unique()
    u_count = len(users)
    i_count = len(items)

    # out_csv = TABLES_DIR / "summary_counts.csv"
    # pd.DataFrame([{"unique_users": u_count, "unique_items": i_count}]).to_csv(out_csv, index=False)

    print(f"Unique users: {u_count} | Unique items: {i_count}")
    # print(f"Save location: {out_csv}")

if __name__ == "__main__":
    main()
