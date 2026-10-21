"""
task_mlp.py -- Mo hinh MLP (PyTorch) + doc du lieu + huan luyen + danh gia
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCALER_PATH = "./scaler_stats.npz"
CLIP = 10.0   # cat gia tri ngoai le sau khi chuan hoa

# ============================================================
# 1. KIEN TRUC MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, n_features: int, h1: int = 128, h2: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 2),            # 2 lop: 0 = BENIGN, 1 = tan cong
        )

    def forward(self, x):
        return self.net(x)


def create_model(n_features: int) -> MLP:
    torch.manual_seed(42)               # moi noi deu khoi tao giong nhau
    return MLP(n_features)

# ============================================================
# 2. DOC + CHUAN HOA DU LIEU
# ============================================================
def load_scaler():
    s = np.load(SCALER_PATH)
    return s["mean"].astype(np.float32), s["std"].astype(np.float32)


def preprocess(X):
    mean, std = load_scaler()
    X = (X.astype(np.float32) - mean) / std
    return np.clip(X, -CLIP, CLIP).astype(np.float32)


def load_client_data(client_id: int, data_dir: str = "client_data"):
    d = np.load(f"./{data_dir}/client_{client_id}.npz")
    return preprocess(d["X"]), d["Y"].astype(np.int64)


def load_server_test_data(data_dir: str = "client_data"):
    d = np.load(f"./{data_dir}/server_test.npz")
    return preprocess(d["X"]), d["Y"].astype(np.int64)

# ============================================================
# 3. LAY / GAN TRONG SO (FedAvg chi can list numpy array)
# ============================================================
def get_weights(model):
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_weights(model, weights):
    keys = list(model.state_dict().keys())
    model.load_state_dict({k: torch.tensor(w) for k, w in zip(keys, weights)}, strict=True)

# ============================================================
# 4. HUAN LUYEN CUC BO + DANH GIA
# ============================================================
def train(model, X, y, epochs: int, lr: float, batch_size: int = 512):
    torch.set_num_threads(2)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=batch_size, shuffle=True)
    last_loss = 0.0
    for _ in range(epochs):
        total, count = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            count += len(xb)
        last_loss = total / count
    return last_loss


@torch.no_grad()
def predict(model, X, y, batch_size: int = 8192):
    """Tra ve (loss trung binh, mang du doan)"""
    model.eval()
    preds, loss_sum = [], 0.0
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size])
        yb = torch.from_numpy(y[i:i + batch_size])
        out = model(xb)
        loss_sum += F.cross_entropy(out, yb, reduction="sum").item()
        preds.append(out.argmax(1).numpy())
    return loss_sum / len(X), np.concatenate(preds)
