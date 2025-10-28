"""
- For each user U', it computes:
  (i) number of ratings
  (ii) timespan (days) between first and last rating
- Create 2 histograms and saves them as PNG
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from filter_users_items import make_csr_from_df

import pandas as pd
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

def load_or_filter(data_path: str, m: int, M: int) -> pd.DataFrame:
    filt_csv = ARTIFACTS_DIR / "filtered_interactions.csv"
    if filt_csv.exists():
        dff = pd.read_csv(filt_csv)
        dff["timestamp"] = pd.to_datetime(dff["timestamp"], utc=True, errors="coerce")
    else:
        data_path = CH1.parent / "data" / "Dataset.npy"
        df= load_dataset(data_path)
        dff, _, _, _ = make_csr_from_df(df, m, M)
    return dff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/Dataset.npy")
    parser.add_argument("--min_ratings", type=int, default=1)
    parser.add_argument("--max_ratings", type=int, default=10)
    args = parser.parse_args()

    dff = load_or_filter(args.data, args.min_ratings, args.max_ratings)

    counts = dff.groupby("user")["item"].count()

    grp = dff.groupby("user")["timestamp"]
    delta = grp.max() - grp.min()     
    span_days = delta.dt.total_seconds().fillna(0) / (24 * 3600)

    plt.figure()
    plt.hist(counts.values, bins=20)
    plt.title("Histogram: Count of Ratings per User")
    plt.xlabel("Count of Ratings")
    plt.ylabel("Frequency")
    out1 = FIGS_DIR / "hist_ratings_per_user.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close()


    plt.figure()
    plt.hist(span_days.values, bins=20)
    plt.title("Histogram: Timespan (days) per User")
    plt.xlabel("Days between first and last rating")
    plt.ylabel("Frequency")
    out2 = FIGS_DIR / "hist_timespan_per_user.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()

    print(f"Save locations: {out1} , {out2}")

if __name__ == "__main__":
    main()
