"""
server_app_xgb.py -- Server Flower cho XGBoost (Federated Bagging, 1 vong)
Thu thap 20 mo hinh XGBoost tu client, gop thanh ensemble toan cuc,
cham diem tren server_test.npz.
"""
import csv
import os
import pickle
import time

import numpy as np
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

from ids_demo.task_xgb import array_to_booster, ensemble_predict_proba, load_server_test_data

NUM_CLIENTS = 20


class FedXgbBagging(FedAvg):
    def __init__(self, data_dir, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.tag = os.path.basename(data_dir.rstrip("/\\"))
        self.t0 = time.time()
        self.upload_mb = 0.0
        self.download_mb = 0.0
        self.comm_mb = 0.0

        os.makedirs("results", exist_ok=True)
        self.csv_path = f"results/xgb_{self.tag}.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["round", "loss", "accuracy", "precision", "recall",
                                    "f1", "tn", "fp", "fn", "tp", "time_s", "comm_mb_cum"])

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if failures:
            print(f"[SERVER] CANH BAO: {len(failures)} client bi loi")

        # Giu nguyen moi mo hinh client (moi mo hinh la 1 mang uint8)
        arrays = [parameters_to_ndarrays(fit_res.parameters)[0] for _, fit_res in results]
        self.upload_mb = sum(a.nbytes for a in arrays) / (1024 * 1024)
        # Download: ensemble toan cuc (= tong cac mo hinh) gui xuong NUM_CLIENTS client
        global_mb = sum(a.nbytes for a in arrays) / (1024 * 1024)
        self.download_mb = global_mb * NUM_CLIENTS
        self.comm_mb = self.upload_mb + self.download_mb

        print(f"[SERVER] Da nhan {len(arrays)} mo hinh XGBoost | "
              f"upload {self.upload_mb:.2f} MB, download {self.download_mb:.2f} MB")
        return ndarrays_to_parameters(arrays), {}

    def evaluate(self, server_round, parameters):
        if server_round == 0:
            return None

        arrays = parameters_to_ndarrays(parameters)
        boosters = [array_to_booster(a) for a in arrays]

        X_test, y_test = load_server_test_data(self.data_dir)
        proba = ensemble_predict_proba(boosters, X_test)
        y_pred = (proba >= 0.5).astype(np.int64)

        loss = log_loss(y_test, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1])
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        elapsed = time.time() - self.t0

        model_path = f"ids_global_xgb_{self.tag}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump([a.tobytes() for a in arrays], f)
        file_mb = os.path.getsize(model_path) / (1024 * 1024)

        print(f"\n{'=' * 50}")
        print(f"[SERVER - XGBOOST] KET QUA DANH GIA VONG {server_round}  ({self.tag})")
        print(f"{'=' * 50}")
        print(f"So mo hinh      : {len(boosters)} (1 mo hinh / client)")
        print(f"Loss (logloss)  : {loss:.4f}")
        print(f"Do chinh xac    : {acc * 100:.4f}%")
        print(f"Do chuan (Prec) : {prec * 100:.4f}%")
        print(f"Do nhay (Recall): {rec * 100:.4f}%")
        print(f"Diem F1-Score   : {f1 * 100:.4f}%")
        print(f"Chi phi mang    : {self.comm_mb:.2f} MB "
              f"(upload {self.upload_mb:.2f} + download {self.download_mb:.2f})")
        print(f"Thoi gian       : {elapsed:.1f} s")
        print(f"Da luu mo hinh  : {model_path} ({file_mb:.2f} MB)")
        print(f"{'=' * 50}\n")

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([server_round, loss, acc, prec, rec, f1, tn, fp, fn, tp,
                                    round(elapsed, 1), round(self.comm_mb, 3)])

        return float(loss), {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


def server_fn(context: Context):
    data_dir = str(context.run_config.get("data-dir", "client_data"))

    strategy = FedXgbBagging(
        data_dir=data_dir,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=ndarrays_to_parameters([np.zeros(1, dtype=np.uint8)]),
    )
    # Giong RF: chi 1 vong, vi cac mo hinh duoc gop (bagging) chu khong cap nhat lap lai
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=1))


app = ServerApp(server_fn=server_fn)
