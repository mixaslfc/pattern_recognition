
"""
- Vector representation of user preferences (CSR)
- Provides get_user_vector(u_id) function for original user id
- Uses artifacts produced in q2_filter_users_items.py
"""

import json
from pathlib import Path
from scipy.sparse import load_npz

CH1 = Path(__file__).resolve().parent
ARTIFACTS_DIR = CH1 / "artifacts"

def load_R_and_maps():
    R = load_npz(ARTIFACTS_DIR / "R_csr.npz")
    with open(ARTIFACTS_DIR / "user2idx.json", "r", encoding="utf-8") as f:
        user2idx = json.load(f)
    with open(ARTIFACTS_DIR / "item2idx.json", "r", encoding="utf-8") as f:
        item2idx = json.load(f)
    return R, user2idx, item2idx

def get_user_vector(u_id: str):
    """
    Returns sparse vector (CSR row) for the original user id.
    If the user is not in the filtered set, returns None.
    """
    R, user2idx, _ = load_R_and_maps()
    key = str(u_id)
    if key not in user2idx:
        return None
    return R.getrow(user2idx[key])

def main():
    try:
        R, user2idx, item2idx = load_R_and_maps()
        print(f"Loaded R with shape {R.shape} and nnz={R.nnz}")
        if len(user2idx) > 0:
            any_uid = list(user2idx.keys())[0]
            vec = get_user_vector(str(any_uid))
            print(f"Sample user vector {any_uid}: non-zeros={vec.nnz}")
        else:
            print("No users found in artifacts.")
    except FileNotFoundError:
        print("No artifacts found. Please run q2_filter_users_items.py first.")

if __name__ == "__main__":
    main()
