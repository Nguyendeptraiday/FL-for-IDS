"""
make_scaler.py  -- CHAY 1 LAN DUY NHAT (tai thu muc FL_Project)
Tinh mean/std toan cuc cho viec chuan hoa du lieu cua MLP.

Moi client chi can gui (n, mean, M2) cua rieng minh len server, server gop lai
(cong thuc Chan) -> tuong duong "thong ke lien ket", khong can gom du lieu tho.
IID va non-IID dung chung X_train nen chi can chay 1 lan tren client_data.
"""
import glob
import numpy as np

files = sorted(glob.glob("./client_data/client_*.npz"))
assert files, "Khong thay ./client_data/client_*.npz - hay chay lenh nay tai thu muc FL_Project"

n_tot, mean_tot, M2_tot = 0, None, None
for f in files:
    X = np.load(f)["X"].astype(np.float64)
    n_k = len(X)
    mean_k = X.mean(axis=0)
    M2_k = ((X - mean_k) ** 2).sum(axis=0)

    if mean_tot is None:
        mean_tot, M2_tot = np.zeros_like(mean_k), np.zeros_like(mean_k)

    delta = mean_k - mean_tot
    new_n = n_tot + n_k
    mean_tot = mean_tot + delta * n_k / new_n
    M2_tot = M2_tot + M2_k + delta ** 2 * n_tot * n_k / new_n
    n_tot = new_n
    print(f"Da gop {f} (n={n_k})")

std = np.sqrt(M2_tot / n_tot)
std[std < 1e-8] = 1.0          # cot hang so -> tranh chia cho 0
np.savez("scaler_stats.npz", mean=mean_tot, std=std)
print(f"Xong! {len(mean_tot)} dac trung, {n_tot} mau -> scaler_stats.npz")
