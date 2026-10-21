"""
client_app_mlp.py -- Client Flower cho MLP
"""
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from ids_demo.task_mlp import (
    create_model, load_client_data, load_scaler, get_weights, set_weights, train,
)


class MLPClient(NumPyClient):
    def __init__(self, model, X, y, epochs, lr):
        self.model, self.X, self.y = model, X, y
        self.epochs, self.lr = epochs, lr

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)                     # nhan mo hinh toan cuc
        loss = train(self.model, self.X, self.y, self.epochs, self.lr)
        print(f"[Client] train xong {len(self.X)} mau, loss={loss:.4f}")
        return get_weights(self.model), len(self.X), {"train_loss": float(loss)}

    def evaluate(self, parameters, config):
        return 0.0, len(self.X), {}


def client_fn(context: Context):
    try:
        partition_id = int(context.node_config["partition-id"])
    except Exception:
        partition_id = int(context.node_id) % 20

    cfg = context.run_config
    data_dir = str(cfg.get("data-dir", "client_data"))
    epochs = int(cfg.get("local-epochs", 2))
    lr = float(cfg.get("lr", 0.001))

    n_features = len(load_scaler()[0])
    model = create_model(n_features)
    X, y = load_client_data(partition_id, data_dir)
    return MLPClient(model, X, y, epochs, lr).to_client()


app = ClientApp(client_fn=client_fn)
