"""
server_app.py -- Server Flower cho Random Forest (Federated Forest, 1 vong)
Gop cay cua 20 client thanh rung toan cuc, cham diem tren server_test.npz.
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

from ids_demo.task import array_to_model, create_model, load_server_test_data, model_to_array

NUM_CLIENTS = 20


class FedForestStrategy(FedAvg):
    def __init__(self, data_dir, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.tag = os.path.basename(data_dir.rstrip("/\\"))
        self.t0 = time.time()
        self.upload_mb = 0.0
        self.download_mb = 0.0
        self.comm_mb = 0.0

        os.makedirs("results", exist_ok=True)
        self.csv_path = f"results/rf_{self.tag}.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["round", "loss", "accuracy", "precision", "recall",
                                    "f1", "tn", "fp", "fn", "tp", "time_s", "comm_mb_cum"])

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if failures:
            print(f"[SERVER] CANH BAO: {len(failures)} client bi loi")

        global_model = create_model(n_jobs=-1)
        global_model.estimators_ = []
        upload_bytes = 0

        for i, (_, fit_res) in enumerate(results):
            arr = parameters_to_ndarrays(fit_res.parameters)[0]
            upload_bytes += arr.nbytes
            client_model = array_to_model(arr)
            global_model.estimators_.extend(client_model.estimators_)
            if i == 0:      # muon cac thuoc tinh noi bo tu client dau tien
                global_model.classes_ = client_model.classes_
                global_model.n_classes_ = client_model.n_classes_
                global_model.n_outputs_ = client_model.n_outputs_
                global_model.n_features_in_ = client_model.n_features_in_

        global_model.n_estimators = len(global_model.estimators_)

        global_arr = model_to_array(global_model)
        self.upload_mb = upload_bytes / (1024 * 1024)
        # Download: rung toan cuc gui xuong NUM_CLIENTS client
        self.download_mb = global_arr.nbytes * NUM_CLIENTS / (1024 * 1024)
        self.comm_mb = self.upload_mb + self.download_mb

        print(f"[SERVER] Da gop {len(results)} client -> {global_model.n_estimators} cay | "
              f"upload {self.upload_mb:.2f} MB, download {self.download_mb:.2f} MB")
        return ndarrays_to_parameters([global_arr]), {}

    def evaluate(self, server_round, parameters):
        if server_round == 0:
            return None

        global_model = array_to_model(parameters_to_ndarrays(parameters)[0])
        global_model.n_jobs = -1

        X_test, y_test = load_server_test_data(self.data_dir)
        proba = global_model.predict_proba(X_test)
        y_pred = global_model.classes_.take(proba.argmax(axis=1))

        loss = log_loss(y_test, np.clip(proba, 1e-7, 1 - 1e-7), labels=global_model.classes_)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        elapsed = time.time() - self.t0

        model_path = f"ids_global_rf_{self.tag}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(global_model, f)
        file_mb = os.path.getsize(model_path) / (1024 * 1024)

        print(f"\n{'=' * 50}")
        print(f"[SERVER - RANDOM FOREST] KET QUA DANH GIA VONG {server_round}  ({self.tag})")
        print(f"{'=' * 50}")
        print(f"Kich thuoc rung : {global_model.n_estimators} cay")
        print(f"Loss (logloss)  : {loss:.4f}")
        print(f"Do chinh xac    : {acc * 100:.4f}%")
        print(f"Do chuan (Prec) : {prec * 100:.4f}%")
        print(f"Do nhay (Recall): {rec * 100:.4f}%")
        print(f"Diem F1-Score   : {f1 * 100:.4f}%")
                # Thay dong print "Chi phi mang":
        print(f"Chi phi mang    : {self.comm_mb:.2f} MB "
              f"(upload {self.upload_mb:.2f} + download {self.download_mb:.2f})")
        print(f"Thoi gian       : {elapsed:.1f} s")
        print(f"Da luu mo hinh  : {model_path} ({file_mb:.2f} MB)")
        print(f"{'=' * 50}\n")

                # Thay doan ghi CSV:
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([server_round, loss, acc, prec, rec, f1, tn, fp, fn, tp,
                                    round(elapsed, 1), round(self.comm_mb, 3)])

        return float(loss), {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


def server_fn(context: Context):
    data_dir = str(context.run_config.get("data-dir", "client_data"))

    strategy = FedForestStrategy(
        data_dir=data_dir,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=ndarrays_to_parameters([np.zeros(1, dtype=np.uint8)]),
    )
    # RF chi can 1 vong: cay duoc gop lai, khong cap nhat lap di lap lai
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=1))


app = ServerApp(server_fn=server_fn)
