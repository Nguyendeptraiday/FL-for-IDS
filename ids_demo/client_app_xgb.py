"""
client_app_xgb.py -- Client Flower cho XGBoost
"""
import time

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from ids_demo.task_xgb import booster_to_array, load_client_data, train_booster


class XGBClient(NumPyClient):
    def __init__(self, X, y, num_trees, max_depth, eta):
        self.X, self.y = X, y
        self.num_trees, self.max_depth, self.eta = num_trees, max_depth, eta

    def get_parameters(self, config):
        return [np.zeros(1, dtype=np.uint8)]      # chua co mo hinh

    def fit(self, parameters, config):
        t0 = time.time()
        print(f"[Client] Dang huan luyen XGBoost ({self.num_trees} cay, "
              f"do sau {self.max_depth}) tren {len(self.y)} mau...")
        booster = train_booster(self.X, self.y, self.num_trees, self.max_depth, self.eta)
        arr = booster_to_array(booster)
        dt = time.time() - t0
        print(f"[Client] Xong sau {dt:.1f}s, kich thuoc mo hinh {arr.nbytes / 1024:.1f} KB")
        return [arr], len(self.y), {"train_time": float(dt), "model_kb": float(arr.nbytes / 1024)}

    def evaluate(self, parameters, config):
        return 0.0, len(self.y), {}


def client_fn(context: Context):
    try:
        partition_id = int(context.node_config["partition-id"])
    except Exception:
        partition_id = int(context.node_id) % 20

    cfg = context.run_config
    data_dir = str(cfg.get("data-dir", "client_data"))
    num_trees = int(cfg.get("xgb-trees", 50))
    max_depth = int(cfg.get("xgb-depth", 6))
    eta = float(cfg.get("xgb-eta", 0.1))

    X, y = load_client_data(partition_id, data_dir)
    return XGBClient(X, y, num_trees, max_depth, eta).to_client()


app = ClientApp(client_fn=client_fn)
