# clustering/kmeans_metrics.py
# -*- coding: utf-8 -*-
"""
- Grouping users (clustering) based on their rating behavior.
- Two approaches:
  (A) K-Means over Cosine similarity (using L2-normalization + TruncatedSVD as a proxy)
  (B) K-Medoids for non-Euclidean metrics (pearson/jaccard/euclidean) using a CLARA-like approach:
      * Take a random sample S of users (e.g., 2000)
      * Run PAM only on the sample (S×S distances)
      * Assign ALL users to the closest of the k medoids (N×k distances)
      * Iterations and keeping the best
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import random
import time

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

THIS_DIR = Path(__file__).resolve().parent
TABLES_DIR = THIS_DIR / "tables"
FIGS_DIR = THIS_DIR / "figs"
ARTIFACTS_DIR = THIS_DIR / "artifacts"
for d in (TABLES_DIR, FIGS_DIR, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def find_preproc_artifacts_dir():
    candidates = [
        THIS_DIR.parent / "preprocessing" / "artifacts",           
        THIS_DIR.parent / "chapter1_preprocessing" / "artifacts",   
    ]
    for c in candidates:
        if (c / "R_csr.npz").exists():
            return c
    return candidates[0]

def load_R_and_maps():
    ch1_art = find_preproc_artifacts_dir()
    R = load_npz(ch1_art / "R_csr.npz")
    with open(ch1_art / "user2idx.json", "r", encoding="utf-8") as f:
        user2idx = json.load(f)
    with open(ch1_art / "item2idx.json", "r", encoding="utf-8") as f:
        item2idx = json.load(f)
    return R, user2idx, item2idx

def run_kmeans_cosine(R: csr_matrix, k: int, n_components: int = 50):
    Rn = normalize(R, norm="l2", axis=1, copy=True)
    if Rn.shape[1] <= 2:
        n_components = 2
    else:
        n_components = min(n_components, max(2, Rn.shape[1]-1))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    X = svd.fit_transform(Rn)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED, max_iter=300, verbose=0)
    labels = km.fit_predict(X)
    inertia = float(km.inertia_)
    return labels, inertia

def pearson_distance_rowpair(a: csr_matrix, b: csr_matrix) -> float:
    ai = set(a.indices); bi = set(b.indices)
    inter = ai & bi
    if len(inter) < 2:
        return 1.0
    idx = np.fromiter(inter, dtype=np.int32)
    av = a[:, idx].toarray().ravel()
    bv = b[:, idx].toarray().ravel()
    if av.std() == 0 or bv.std() == 0:
        return 1.0
    corr = np.corrcoef(av, bv)[0, 1]
    if np.isnan(corr):
        return 1.0
    return 1.0 - float(corr)

def jaccard_distance_rowpair(a: csr_matrix, b: csr_matrix) -> float:
    ai = set(a.indices); bi = set(b.indices)
    union = ai | bi
    if len(union) == 0:
        return 1.0
    inter = ai & bi
    jacc = len(inter) / len(union)
    return 1.0 - float(jacc)

def euclidean_distance_rowpair(a: csr_matrix, b: csr_matrix) -> float:
    ai = set(a.indices); bi = set(b.indices)
    inter = ai & bi
    if not inter:
        return 1.0
    idx = np.fromiter(inter, dtype=np.int32)
    av = a[:, idx].toarray().ravel()
    bv = b[:, idx].toarray().ravel()
    return float(np.linalg.norm(av - bv))

def distance_rowpair(a: csr_matrix, b: csr_matrix, metric: str) -> float:
    if metric == "pearson":
        return pearson_distance_rowpair(a, b)
    elif metric == "jaccard":
        return jaccard_distance_rowpair(a, b)
    elif metric == "euclidean":
        return euclidean_distance_rowpair(a, b)
    else:
        raise ValueError("Unsupported metric for K-Medoids/CLARA.")

def k_medoids_pam(D: np.ndarray, k: int, max_iter: int = 100, seed: int = RANDOM_SEED):

    rng = np.random.default_rng(seed)
    n = D.shape[0]
    medoids = rng.choice(n, size=k, replace=False).tolist()

    def assign(meds):
        return np.argmin(D[:, meds], axis=1)

    def total_cost(meds, labels):
        return float(np.sum(D[np.arange(n), np.array(meds)[labels]]))

    labels = assign(medoids)
    best_cost = total_cost(medoids, labels)

    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for m_pos in range(len(medoids)):
            mi = medoids[m_pos]
            for h in range(n):
                if h in medoids:
                    continue
                trial = medoids.copy()
                trial[m_pos] = h
                lab = assign(trial)
                cost = total_cost(trial, lab)
                if cost + 1e-12 < best_cost:
                    medoids = trial
                    labels = lab
                    best_cost = cost
                    improved = True
    return labels, medoids, best_cost

def clara_kmedoids(R: csr_matrix, metric: str, k: int, sample_size: int = 2000, restarts: int = 1,
                   pam_max_iter: int = 100, seed: int = RANDOM_SEED):

    rng = np.random.default_rng(seed)
    n = R.shape[0]
    if sample_size >= n:
        sample_size = n

    best_labels_full = None
    best_medoids_full = None
    best_cost_full = np.inf

    for rep in range(restarts):
        t0 = time.time()
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        Rs = R[sample_idx, :]

        S = Rs.shape[0]
        D = np.zeros((S, S), dtype=np.float32)
        for i in range(S):
            ri = Rs.getrow(i)
            for j in range(i+1, S):
                rj = Rs.getrow(j)
                d = distance_rowpair(ri, rj, metric=metric)
                D[i, j] = D[j, i] = d

        _, medoids_s, _ = k_medoids_pam(D, k=k, max_iter=pam_max_iter, seed=seed)

        medoid_indices_full = sample_idx[np.array(medoids_s, dtype=int)]

        labels_full, cost_full = assign_all_to_medoids(R, medoid_indices_full, metric)

        if cost_full < best_cost_full:
            best_cost_full = cost_full
            best_labels_full = labels_full
            best_medoids_full = medoid_indices_full

        dt = time.time() - t0
        print(f"CLARA reps {rep+1}/{restarts}: cost={cost_full:.3f} (time {dt:.1f}s)")

    return best_labels_full, best_medoids_full, float(best_cost_full)

def assign_all_to_medoids(R: csr_matrix, medoid_indices: np.ndarray, metric: str):
  
    k = len(medoid_indices)
    n = R.shape[0]

    medoid_rows = [R.getrow(int(m)) for m in medoid_indices]

    labels = np.empty(n, dtype=np.int32)
    min_d = np.empty(n, dtype=np.float32)

    for i in range(n):
        ri = R.getrow(i)
        best_j = 0
        best_val = np.float32(1e9)
        for j in range(k):
            d = distance_rowpair(ri, medoid_rows[j], metric=metric)
            if d < best_val:
                best_val = d
                best_j = j
        labels[i] = best_j
        min_d[i] = best_val
        if (i+1) % 100000 == 0:
            print(f"      ... assignment {i+1}/{n}")

    total_cost = float(np.sum(min_d))
    return labels, total_cost

# -------------------------------------------------------
# main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", default=["cosine", "pearson", "jaccard", "euclidean"],
        help="Metrics: cosine(KMeans), pearson/jaccard/euclidean (KMedoids-CLARA)")
    parser.add_argument("--k_values", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--svd_components", type=int, default=50)
    parser.add_argument("--sample_size", type=int, default=2000, help="Sample size for CLARA")
    parser.add_argument("--clara_restarts", type=int, default=1, help="CLARA restarts with new samples")
    parser.add_argument("--pam_max_iter", type=int, default=100, help="Max iterations for PAM on the sample")
    args = parser.parse_args()

    R, user2idx, item2idx = load_R_and_maps()
    print(f"Φορτώθηκε R: {R.shape}, nnz={R.nnz}")

    for metric in args.metrics:
        results_rows = []
        for k in args.k_values:
            if metric == "cosine":
                labels, inertia = run_kmeans_cosine(R, k, n_components=args.svd_components)
                np.save(ARTIFACTS_DIR / f"labels_{metric}_k{k}.npy", labels)
                results_rows.append({"k": k, "inertia": inertia})
                sizes = pd.Series(labels).value_counts().sort_index()
                sizes.to_csv(TABLES_DIR / f"cluster_sizes_{metric}_k{k}.csv", header=["size"])
                print(f"[cosine] k={k}: inertia={inertia:.3f} | sizes={list(sizes.values)}")
            else:
                print(f"[{metric}] k={k}: running CLARA (sample={args.sample_size}, restarts={args.clara_restarts}) ...")
                labels, medoids, cost = clara_kmedoids(
                    R, metric=metric, k=k,
                    sample_size=args.sample_size,
                    restarts=args.clara_restarts,
                    pam_max_iter=args.pam_max_iter,
                    seed=RANDOM_SEED
                )
                np.save(ARTIFACTS_DIR / f"labels_{metric}_k{k}.npy", labels)
                results_rows.append({"k": k, "within_sum": float(cost)})
                sizes = pd.Series(labels).value_counts().sort_index()
                sizes.to_csv(TABLES_DIR / f"cluster_sizes_{metric}_k{k}.csv", header=["size"])
                print(f"[{metric}] k={k}: within_sum={cost:.3f} | sizes={list(sizes.values)}")

        out_csv = TABLES_DIR / (("inertia_" + metric + ".csv") if metric == "cosine" else ("within_sum_" + metric + ".csv"))
        if results_rows:
            pd.DataFrame(results_rows).to_csv(out_csv, index=False)
            print(f"Saved: {out_csv}")
        else:
            print(f"No results for metric={metric} — skipping CSV creation.")

if __name__ == "__main__":
    main()
