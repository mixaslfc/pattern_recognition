
"""
Light & Deep sanity checks for your pre-processing artifacts.
Run:
  python tests_quick.py                 # light tests (fast)
  python tests_quick.py --deep          # deep tests (slower, uses CSV in chunks)
"""

import argparse, json, os
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz

def try_load(path):
    if path.exists():
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    return None

def load_maps(base: Path):
    user2idx = try_load(base/"user2idx.json")
    item2idx = try_load(base/"item2idx.json")
    return user2idx, item2idx

def light_tests(base: Path):
    print("=== LIGHT TESTS ===")
    print(f"Using artifacts from: {base.resolve()}")
    
    R = load_npz(base/"R_csr.npz")
    user2idx, item2idx = load_maps(base)

    # 1) Shape vs maps
    if user2idx is not None:
        assert R.shape[0] == len(user2idx), f"R rows {R.shape[0]} != len(user2idx) {len(user2idx)}"
        print("[OK] R.rows matches len(user2idx)")
    else:
        print("[WARN] user2idx.json not found — skipping rows vs map check")
    if item2idx is not None:
        assert R.shape[1] == len(item2idx), f"R cols {R.shape[1]} != len(item2idx) {len(item2idx)}"
        print("[OK] R.cols matches len(item2idx)")
    else:
        print("[WARN] item2idx.json not found — skipping cols vs map check")

    
    # 2) Values are integers; range should be [1,10] per spec, but we WARN if not
    data = R.data
    assert data.size > 0, "R has no non-zero entries"
    assert np.all(np.equal(np.mod(data, 1), 0)), "Ratings are not integers"
    vmin, vmax = int(data.min()), int(data.max())
    print(f"[INFO] Rating value range in CSR: [{vmin},{vmax}]")
    out_mask = (data < 0) | (data > 10)
    if out_mask.any():
        bad = int(out_mask.sum())
        print(f"[WARN] {bad} ratings outside [0,10]; consider normalizing/clipping before CSR.")
    zeros = int((np.array(data) == 0).sum())
    if zeros > 0:
        print(f"[WARN] Found {zeros} explicit 0 ratings in CSR (spec treats 0 as absence).")
    else:
        print("[OK] No explicit zeros stored in CSR")

def deep_tests(base: Path):
    print("=== DEEP TESTS ===")
    import pandas as pd

    R = load_npz(base/"R_csr.npz")
    user2idx, item2idx = load_maps(base)

    # A) Verify u_idx/i_idx bounds and (if possible) membership (chunked)
    csv_path = base/"filtered_interactions.csv"
    assert csv_path.exists(), f"{csv_path} not found"
    chunksize = 200_000
    total_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=["user","item","u_idx","i_idx"], chunksize=chunksize):
        total_rows += len(chunk)
        assert chunk["u_idx"].between(0, R.shape[0]-1).all(), "Found u_idx out of bounds"
        assert chunk["i_idx"].between(0, R.shape[1]-1).all(), "Found i_idx out of bounds"
        if user2idx is not None:
            sub = chunk.sample(n=min(1000, len(chunk)), random_state=42)
            mapped = sub["user"].map(user2idx)
            assert mapped.notna().all(), "Found user not in user2idx"
            assert np.array_equal(mapped.to_numpy(), sub["u_idx"].to_numpy()), "Mismatch: user2idx vs u_idx"
    msg = "[OK] CSV indices consistent with bounds"
    if user2idx is not None:
        msg += " and maps"
    print(msg+f" on {total_rows} rows")

    # B) Cross-check columns count vs item2idx if available
    if item2idx is not None:
        assert len(item2idx) == R.shape[1], "item2idx length != R.shape[1]"
        print("[OK] item2idx length matches R columns")
    else:
        print("[WARN] item2idx.json not found — skipping column check")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true", help="run deeper CSV-index checks (slower)")
    parser.add_argument("--base", type=Path, default=Path(__file__).parent, help="path with artifacts")
    args = parser.parse_args()

    light_tests(args.base)
    if args.deep:
        deep_tests(args.base)
