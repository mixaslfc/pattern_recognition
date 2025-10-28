"""
- Filter users/items based on number of ratings [m, M]
- Remap user/item to 0..U'-1 and 0..I'-1
- Build sparse CSR interaction matrix (values = rating)
- Save:
  - artifacts/user2idx.json, item2idx.json
  - artifacts/R_csr.npz (CSR)
  - tables/filtered_preview.csv (first 10 rows)
- Print a brief summary
"""

import argparse
import json
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

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

    path = CH1.parent / "data" / "Dataset.npy"
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

def filter_users_items(df: pd.DataFrame, m: int, M: int):
    counts = df.groupby("user")["item"].count()
    keep_users = counts[(counts >= m) & (counts <= M)].index
    print(f"Users before filtering: {df['user'].nunique()} | After: {len(keep_users)}")

    dff = df[df["user"].isin(keep_users)].copy()
    dff = dff.sort_values("timestamp")
    dff = dff.drop_duplicates(["user","item"], keep="last")  
    keep_items = dff["item"].unique()
    user2idx = {u: i for i, u in enumerate(sorted(keep_users))}
    item2idx = {it: j for j, it in enumerate(sorted(keep_items))}
    dff["u_idx"] = dff["user"].map(user2idx)
    dff["i_idx"] = dff["item"].map(item2idx)

    rows = dff["u_idx"].to_numpy()
    cols = dff["i_idx"].to_numpy()
    vals = dff["rating"].astype(int).to_numpy()
    R = csr_matrix((vals, (rows, cols)), shape=(len(user2idx), len(item2idx)))

    return dff, R, user2idx, item2idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/Dataset.npy")
    parser.add_argument("--min_ratings", type=int, default=5, help="Lowest ratings per user")
    parser.add_argument("--max_ratings", type=int, default=50, help="Highest ratings per user")
    args = parser.parse_args()

    df = load_dataset(args.data)
    dff, R, user2idx, item2idx = filter_users_items(df, args.min_ratings, args.max_ratings)

    with open(ARTIFACTS_DIR / "user2idx.json", "w", encoding="utf-8") as f:
        json.dump(user2idx, f, ensure_ascii=False, indent=2)
    with open(ARTIFACTS_DIR / "item2idx.json", "w", encoding="utf-8") as f:
        json.dump(item2idx, f, ensure_ascii=False, indent=2)

    save_npz(ARTIFACTS_DIR / "R_csr.npz", R)

    preview = dff[["user", "item", "rating", "timestamp", "u_idx", "i_idx"]].head(10)
    preview.to_csv(TABLES_DIR / "filtered_preview.csv", index=False)

    dff.to_csv(ARTIFACTS_DIR / "filtered_interactions.csv", index=False)

    print(f"Filtered users U': {R.shape[0]} | Filtered items I': {R.shape[1]} | Ratings: {R.nnz}")
    print(f"CSR: {ARTIFACTS_DIR / 'R_csr.npz'} | Maps: user2idx.json, item2idx.json")

# To use in histograms.py
def make_csr_from_df(df: pd.DataFrame, m: int, M: int):
    return filter_users_items(df, m, M)

if __name__ == "__main__":
    main()
