"""
- 2D visualization of users with TruncatedSVD
- Scatter plot of clusters from q1_kmeans_metrics.py
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CH2 = Path(__file__).resolve().parent
FIGS_DIR = CH2 / "figs"
ARTIFACTS_DIR = CH2 / "artifacts"
for d in (FIGS_DIR, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def load_R():
    ch1 = CH2.parent / "preprocessing" / "artifacts"
    R = load_npz(ch1 / "R_csr.npz")
    return R

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", default=["cosine", "pearson", "jaccard"])
    parser.add_argument("--k_values", nargs="+", type=int, default=[3, 4, 5])
    args = parser.parse_args()

    R = load_R()
    # 2D SVD
    Rn = normalize(R, norm="l2", axis=1, copy=True)
    svd2 = TruncatedSVD(n_components=2, random_state=RANDOM_SEED)
    X2 = svd2.fit_transform(Rn)

    for metric in args.metrics:
        for k in args.k_values:
            labels_path = ARTIFACTS_DIR / f"labels_{metric}_k{k}.npy"
            if not labels_path.exists():
                print(f"Missing  {metric}, k={k}. RUN q1_kmeans_metrics.py first.")
                continue
            labels = np.load(labels_path)

            plt.figure()
            # Scatter plot: 2D, without special colors (default)
            for cl in sorted(np.unique(labels)):
                idx = (labels == cl)
                plt.scatter(X2[idx, 0], X2[idx, 1], label=f"Cluster {cl}", s=12)
            plt.title(f"User Clusters — Metric: {metric}, k={k}")
            plt.xlabel("Component 1 (SVD)")
            plt.ylabel("Component 2 (SVD)")
            plt.legend()
            out = FIGS_DIR / f"clusters_{metric}_k{k}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"Saved image: {out}")

if __name__ == "__main__":
    main()
