"""
client_app.py -- Client Flower cho Random Forest
"""
import time

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from ids_demo.task import create_model, load_client_data, model_to_array


class RFClient(NumPyClient):
    def __init__(self, model, X, y):
        self.model, self.X, self.y = model, X, y

    def get_parameters(self, config):
        return [np.zeros(1, dtype=np.uint8)]      # chua co mo hinh

    def fit(self, parameters, config):
        t0 = time.time()
        print(f"[Client] Dang trong {self.model.n_estimators} cay tren {len(self.y)} mau...")
        self.model.fit(self.X, self.y)
        arr = model_to_array(self.model)
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
    n_trees = int(cfg.get("rf-trees", 10))

    X, y = load_client_data(partition_id, data_dir)
    return RFClient(create_model(n_trees), X, y).to_client()


app = ClientApp(client_fn=client_fn)
