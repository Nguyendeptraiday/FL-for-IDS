"""
task_xgb.py -- XGBoost cho Hoc Lien Ket (Federated Bagging)
Moi client tu huan luyen 1 mo hinh XGBoost tren du lieu cuc bo, gui len server.
Server giu ca 20 mo hinh lam thanh 1 "ensemble" toan cuc: du doan = trung binh
xac suat cua 20 mo hinh (tuong tu cach Random Forest lay trung binh cac cay).
XGBoost la mo hinh dua tren cay nen KHONG can chuan hoa du lieu.
"""
import numpy as np
import xgboost as xgb

# ============================================================
# 1. DOC DU LIEU (cung file .npz da chia san)
# ============================================================
def load_client_data(client_id: int, data_dir: str = "client_data"):
    d = np.load(f"./{data_dir}/client_{client_id}.npz")
    return d["X"].astype(np.float32), d["Y"].astype(np.int64)


def load_server_test_data(data_dir: str = "client_data"):
    d = np.load(f"./{data_dir}/server_test.npz")
    return d["X"].astype(np.float32), d["Y"].astype(np.int64)

# ============================================================
# 2. HUAN LUYEN CUC BO
# ============================================================
def train_booster(X, y, num_trees: int, max_depth: int, eta: float, seed: int = 42):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": max_depth,
        "eta": eta,
        "tree_method": "hist",
        "nthread": 2,
        "seed": seed,
    }
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(params, dtrain, num_boost_round=num_trees)

# ============================================================
# 3. DONG GOI / MO GOI MO HINH (booster <-> mang uint8)
# ============================================================
def booster_to_array(booster) -> np.ndarray:
    return np.frombuffer(bytes(booster.save_raw("ubj")), dtype=np.uint8).copy()


def array_to_booster(arr: np.ndarray):
    booster = xgb.Booster()
    booster.load_model(bytearray(arr.tobytes()))
    return booster

# ============================================================
# 4. DU DOAN CUA ENSEMBLE TOAN CUC
# ============================================================
def ensemble_predict_proba(boosters, X) -> np.ndarray:
    """Trung binh xac suat 'tan cong' cua tat ca mo hinh client"""
    dtest = xgb.DMatrix(X)
    proba = np.zeros(len(X), dtype=np.float64)
    for b in boosters:
        proba += b.predict(dtest)
    return proba / len(boosters)
