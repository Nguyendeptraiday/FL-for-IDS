"""
server_app_mlp.py -- Server Flower cho MLP (FedAvg co san + cham diem tren server_test.npz)
"""
import csv
import os
import time

import numpy as np
import torch
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from ids_demo.task_mlp import (
    create_model, get_weights, load_scaler, load_server_test_data, predict, set_weights,
)

NUM_CLIENTS = 20


class FedAvgMLP(FedAvg):
    """FedAvg (trung binh co trong so theo so mau) + danh gia toan cuc moi vong"""

    def __init__(self, data_dir, n_features, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.tag = os.path.basename(data_dir.rstrip("/\\"))
        self.global_model = create_model(n_features)
        self._test = None
        self.t0 = time.time()
        self.cum_comm_mb = 0.0

        os.makedirs("results", exist_ok=True)
        self.csv_path = f"results/mlp_{self.tag}.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["round", "loss", "accuracy", "precision", "recall",
                                    "f1", "tn", "fp", "fn", "tp", "time_s", "comm_mb_cum"])

    def evaluate(self, server_round, parameters):
        if server_round == 0:           # chi la tham so khoi tao
            return None

        weights = parameters_to_ndarrays(parameters)
        set_weights(self.global_model, weights)

        if self._test is None:
            self._test = load_server_test_data(self.data_dir)
        X_test, y_test = self._test

        loss, y_pred = predict(self.global_model, X_test, y_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        # Chi phi truyen thong: moi vong 20 client tai xuong + tai len 1 bo trong so
        model_mb = sum(w.nbytes for w in weights) / (1024 * 1024)
        self.cum_comm_mb += 2 * NUM_CLIENTS * model_mb
        elapsed = time.time() - self.t0

        print(f"\n{'=' * 50}")
        print(f"[SERVER - MLP] KET QUA DANH GIA VONG {server_round}  ({self.tag})")
        print(f"{'=' * 50}")
        print(f"Loss            : {loss:.4f}")
        print(f"Do chinh xac    : {acc * 100:.4f}%")
        print(f"Do chuan (Prec) : {prec * 100:.4f}%")
        print(f"Do nhay (Recall): {rec * 100:.4f}%")
        print(f"Diem F1-Score   : {f1 * 100:.4f}%")
        print(f"Chi phi mang    : {self.cum_comm_mb:.2f} MB (tich luy)")
        print(f"Thoi gian       : {elapsed:.1f} s")
        print(f"{'=' * 50}\n")

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([server_round, loss, acc, prec, rec, f1,
                                    tn, fp, fn, tp, round(elapsed, 1),
                                    round(self.cum_comm_mb, 3)])

        torch.save(self.global_model.state_dict(), f"ids_global_mlp_{self.tag}.pt")

        return float(loss), {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


def server_fn(context: Context):
    cfg = context.run_config
    num_rounds = int(cfg.get("num-server-rounds", 10))
    data_dir = str(cfg.get("data-dir", "client_data"))

    n_features = len(load_scaler()[0])
    init_params = ndarrays_to_parameters(get_weights(create_model(n_features)))

    strategy = FedAvgMLP(
        data_dir=data_dir,
        n_features=n_features,
        fraction_fit=1.0,
        fraction_evaluate=0.0,          # khong can vong evaluate phia client
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=init_params,
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))


app = ServerApp(server_fn=server_fn)
