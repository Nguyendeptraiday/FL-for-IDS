"""
task.py -- Random Forest cho Hoc Lien Ket (Federated Forest)
Moi client trong N cay tren du lieu cuc bo, server gop cay cua cac client thanh 1 rung toan cuc.
"""
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# 1. MO HINH
# ============================================================
def create_model(n_estimators: int = 10, n_jobs: int = 2):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=n_jobs)

# ============================================================
# 2. DOC DU LIEU (file .npz da chia san)
# ============================================================
def load_client_data(client_id: int, data_dir: str = "client_data"):
    d = np.load(f"./{data_dir}/client_{client_id}.npz")
    return d["X"], d["Y"]


def load_server_test_data(data_dir: str = "client_data"):
    d = np.load(f"./{data_dir}/server_test.npz")
    return d["X"], d["Y"]

# ============================================================
# 3. DONG GOI / MO GOI MO HINH (pickle <-> mang uint8)
# ============================================================
def model_to_array(model) -> np.ndarray:
    return np.frombuffer(pickle.dumps(model), dtype=np.uint8).copy()


def array_to_model(arr: np.ndarray):
    return pickle.loads(arr.tobytes())
