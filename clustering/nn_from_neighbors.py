# clustering/nn_from_neighbors.py
# -*- coding: utf-8 -*-
"""
NN from neighbors (per cluster) with Jaccard set distance:
- Select top-M items per cluster (Î)
- For each user: create features from k neighbors (ratings on Î)
- Model: MLPRegressor (multi-output), MAE only on true values (not on zeros)
Note: We use speed-up tricks (precompute sets, Rc_items) and safe paths.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

THIS = Path(__file__).resolve().parent
ART = THIS / "artifacts"
TAB = THIS / "tables"
ART.mkdir(exist_ok=True, parents=True)
TAB.mkdir(exist_ok=True, parents=True)

def find_preproc_artifacts_dir(base: Path) -> Path:
    candidates = [
        base.parent / "preprocessing" / "artifacts",
        base.parent / "chapter1_preprocessing" / "artifacts",
    ]
    for c in candidates:
        if (c / "R_csr.npz").exists():
            return c
    return candidates[0]

def load_R_and_labels(labels_path: Path):
    ch1 = find_preproc_artifacts_dir(THIS)
    R = load_npz(ch1 / "R_csr.npz").tocsr()
    labels = np.load(labels_path)
    return R, labels

def jaccard_set_distance_from_sets(ai: set, bi: set) -> float:
    union = ai | bi
    if not union:
        return 1.0
    inter = ai & bi
    return 1.0 - (len(inter) / len(union))

def topM_items_in_cluster(Rc: csr_matrix, M: int) -> np.ndarray:
    counts = np.diff(Rc.tocsc().indptr)  
    M_eff = min(M, Rc.shape[1])
    top = np.argsort(-counts)[:M_eff]
    return np.sort(top.astype(np.int32))

def build_xy_for_user(Rc_items: csr_matrix, u: int, neigh: list, M: int, k: int):
    feats = []
    for j in neigh:
        rj = Rc_items.getrow(j).toarray().ravel()
        feats.append(rj)
    if feats:
        X_u = np.concatenate(feats, axis=0)
    else:
        X_u = np.zeros(M * k, dtype=np.float32)
    y_u = Rc_items.getrow(u).toarray().ravel()
    return X_u.astype(np.float32, copy=False), y_u.astype(np.float32, copy=False)

def mae_on_observed(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (y_true != 0)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def k_neighbors_in_cluster_sets(user_sets: list[set], u_idx: int, k: int) -> list[int]:
    ai = user_sets[u_idx]
    dists = []
    for j, bj in enumerate(user_sets):
        if j == u_idx:
            continue
        d = jaccard_set_distance_from_sets(ai, bj)
        dists.append((d, j))
    dists.sort(key=lambda x: x[0])
    return [j for _, j in dists[:k]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="π.χ. clustering/artifacts/labels_jaccard_k3.npy")
    ap.add_argument("--k", type=int, default=5, help="")
    ap.add_argument("--M", type=int, default=300, help="")
    ap.add_argument("--hidden", type=int, default=256, help="")
    ap.add_argument("--max_users_per_cluster", type=int, default=4000, help="")
    ap.add_argument("--test_size", type=float, default=0.2, help="")
    args = ap.parse_args()

    R, labels = load_R_and_labels(Path(args.labels))
    n_clusters = int(labels.max()) + 1
    rows = []

    print(f"Loading R: {R.shape}, nnz={R.nnz}")
    print(f"Loaded labels: {n_clusters} clusters")

    for cl in range(n_clusters):
        idx = np.where(labels == cl)[0]
        users_in_cluster = len(idx)

        if users_in_cluster < max(args.k + 5, 10):
            rows.append({"cluster": cl, "users": users_in_cluster, "train_MAE": np.nan, "test_MAE": np.nan})
            print(f"[cl {cl}] very few users ({users_in_cluster}), skipping.")
            continue

        if users_in_cluster > args.max_users_per_cluster:
            rng = np.random.default_rng(RANDOM_SEED)
            idx = rng.choice(idx, size=args.max_users_per_cluster, replace=False)
            users_in_cluster = len(idx)

        Rc = R[idx, :].tocsr()

        items = topM_items_in_cluster(Rc, args.M)
        Rc_items = Rc[:, items].tocsr()
        M_eff = Rc_items.shape[1]

        user_sets = [set(Rc_items.getrow(i).indices.tolist()) for i in range(users_in_cluster)]
        k_eff = min(args.k, max(1, users_in_cluster - 1))

        X_list, Y_list = [], []
        for u_local in range(users_in_cluster):
            neigh = k_neighbors_in_cluster_sets(user_sets, u_local, k_eff)
            X_u, y_u = build_xy_for_user(Rc_items, u_local, neigh, M_eff, k_eff)
            if np.any(y_u != 0):
                X_list.append(X_u)
                Y_list.append(y_u)

        if len(X_list) < 20:
            rows.append({"cluster": cl, "users": users_in_cluster, "train_MAE": np.nan, "test_MAE": np.nan})
            print(f"[cl {cl}] very few samples with observations in Î, skipping.")
            continue

        X = np.vstack(X_list).astype(np.float32, copy=False)
        Y = np.vstack(Y_list).astype(np.float32, copy=False)

        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=args.test_size, random_state=RANDOM_SEED)

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        mlp = MLPRegressor(
            hidden_layer_sizes=(args.hidden,),
            random_state=42,
            max_iter=150,
            learning_rate_init=5e-4,
            early_stopping=True,
            n_iter_no_change=5,
            validation_fraction=0.1,
            verbose=False,
        )
        mlp.fit(Xtr, Ytr)

        Ytr_hat = mlp.predict(Xtr)
        Yte_hat = mlp.predict(Xte)

        tr_mae = float(np.nanmean([mae_on_observed(t, p) for t, p in zip(Ytr, Ytr_hat)]))
        te_mae = float(np.nanmean([mae_on_observed(t, p) for t, p in zip(Yte, Yte_hat)]))

        rows.append({"cluster": cl, "users": int(users_in_cluster), "train_MAE": tr_mae, "test_MAE": te_mae})
        print(f"[cl {cl}] users={users_in_cluster}  train_MAE={tr_mae:.4f}  test_MAE={te_mae:.4f}")

    df = pd.DataFrame(rows)
    out = TAB / f"nn_mae_{Path(args.labels).stem}.csv"
    df.to_csv(out, index=False)
    print(f"Saving: {out}")

if __name__ == "__main__":
    main()
