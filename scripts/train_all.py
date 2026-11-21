"""Train temporal-only + STGNN (fixed / adaptive / hybrid) on LargeST-SD."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import STGNN, TemporalOnly, normalize_adj  # noqa: E402
from src.pngutil import heatmap_png, line_plot_png  # noqa: E402

DATA = ROOT / "data" / "sd_processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
WTS = ROOT / "weights"
for p in (RES, FIG, WTS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
IN_STEPS = 8
OUT_STEPS = 4
HIDDEN = 24
BATCH = 16
EPOCHS = 8
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WindowDataset(Dataset):
    def __init__(self, X, starts):
        self.X = X  # T,N
        self.starts = starts  # list of t where window ends at t (inclusive)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = self.starts[i]
        x = self.X[t - IN_STEPS + 1 : t + 1]  # T,N
        y = self.X[t + 1 : t + 1 + OUT_STEPS]
        return (
            torch.from_numpy(x[:, :, None]),  # T,N,1
            torch.from_numpy(y[:, :, None]),
        )


def split_indices(timestamps: np.ndarray):
    """Return valid prediction end indices t for train/val/test."""
    ts = timestamps.astype("datetime64[ns]")
    train_end = np.datetime64("2020-12-31T23:59:59")
    val_end = np.datetime64("2021-03-31T23:59:59")
    # windows must stay inside each split
    def collect(lo, hi):
        # lo/hi are inclusive datetime bounds for the split's timestamps
        idx = np.where((ts >= lo) & (ts <= hi))[0]
        if len(idx) == 0:
            return []
        i0, i1 = int(idx[0]), int(idx[-1])
        # need t-7 >= i0 and t+4 <= i1
        starts = list(range(i0 + IN_STEPS - 1, i1 - OUT_STEPS + 1))
        return starts

    train = collect(np.datetime64("2020-01-01"), train_end)
    val = collect(np.datetime64("2021-01-01"), val_end)
    test = collect(np.datetime64("2021-04-01"), np.datetime64("2021-12-31T23:59:59"))
    return train, val, test


@torch.no_grad()
def evaluate(model, loader, mean, std, A_hat):
    model.eval()
    se = 0.0
    ae = 0.0
    n = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # normalize input
        xn = (x - mean) / std
        pred_n = model(xn, A_hat)
        pred = pred_n * std + mean
        err = pred - y
        se += float(torch.sum(err**2))
        ae += float(torch.sum(torch.abs(err)))
        n += err.numel()
    mae = ae / n
    rmse = (se / n) ** 0.5
    return mae, rmse


def train_one(name, model, loaders, mean, std, A_hat, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()
    train_loader, val_loader, test_loader = loaders
    best_val = 1e9
    best_state = None
    t0 = time.time()
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            xn = (x - mean) / std
            yn = (y - mean) / std
            pred = model(xn, A_hat)
            loss = loss_fn(pred, yn)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.size(0)
            count += x.size(0)
        val_mae, val_rmse = evaluate(model, val_loader, mean, std, A_hat)
        history.append(
            {"epoch": ep, "train_l1_norm": total / max(count, 1), "val_mae": val_mae, "val_rmse": val_rmse}
        )
        print(f"{name} ep{ep}: train={total/max(count,1):.4f} val_mae={val_mae:.4f}", flush=True)
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(best_state, WTS / f"{name}.pt")
    val_mae, val_rmse = evaluate(model, val_loader, mean, std, A_hat)
    test_mae, test_rmse = evaluate(model, test_loader, mean, std, A_hat)
    return {
        "model": name,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "train_time_sec": train_time,
        "history": history,
        "device": str(DEVICE),
    }


def forecast_curves(model, X, mean, std, A_hat, sensor_ids, t_end):
    model.eval()
    x = torch.from_numpy(X[t_end - IN_STEPS + 1 : t_end + 1, :, None]).unsqueeze(0).to(DEVICE)
    y = X[t_end + 1 : t_end + 1 + OUT_STEPS]
    xn = (x - mean) / std
    with torch.no_grad():
        pred = (model(xn, A_hat) * std + mean).cpu().numpy()[0, :, :, 0]  # out,N
    curves = {}
    for s in sensor_ids:
        curves[str(s)] = {
            "actual": y[:, s].tolist(),
            "pred": pred[:, s].tolist(),
            "history": X[t_end - IN_STEPS + 1 : t_end + 1, s].tolist(),
        }
    return curves


def main():
    set_seed()
    X = np.load(DATA / "flow_15min.npy")  # T,N
    A = np.load(DATA / "adj.npy")
    ts = np.load(DATA / "timestamps.npy")
    N = X.shape[1]
    print("X", X.shape, "device", DEVICE, flush=True)

    train_idx, val_idx, test_idx = split_indices(ts)
    print("windows", len(train_idx), len(val_idx), len(test_idx), flush=True)

    # normalize from train observations only
    train_t0 = train_idx[0] - IN_STEPS + 1
    train_t1 = train_idx[-1]
    train_slice = X[train_t0 : train_t1 + 1]
    mean = float(train_slice.mean())
    std = float(train_slice.std() + 1e-6)
    mean_t = torch.tensor(mean, device=DEVICE)
    std_t = torch.tensor(std, device=DEVICE)

    # subsample train for CPU speed if huge
    max_train = 2500
    if len(train_idx) > max_train:
        rng = np.random.default_rng(SEED)
        train_idx = sorted(rng.choice(train_idx, size=max_train, replace=False).tolist())

    train_ds = WindowDataset(X, train_idx)
    val_ds = WindowDataset(X, val_idx[::3][:800])  # light val
    test_ds = WindowDataset(X, test_idx[::3][:1200])
    loaders = (
        DataLoader(train_ds, batch_size=BATCH, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH),
        DataLoader(test_ds, batch_size=BATCH),
    )

    A_t = torch.from_numpy(A).to(DEVICE)
    A_hat = normalize_adj(A_t)

    results = []

    # Temporal-only
    m = TemporalOnly(IN_STEPS, HIDDEN, OUT_STEPS).to(DEVICE)
    r = train_one("temporal_only", m, loaders, mean_t, std_t, A_hat)
    r["adjacency"] = "None"
    results.append(r)

    # Fixed
    m = STGNN(N, IN_STEPS, HIDDEN, OUT_STEPS, mode="fixed").to(DEVICE)
    r = train_one("stgnn_fixed", m, loaders, mean_t, std_t, A_hat)
    r["adjacency"] = "Fixed"
    results.append(r)

    # Adaptive
    m = STGNN(N, IN_STEPS, HIDDEN, OUT_STEPS, mode="adaptive", topk=15).to(DEVICE)
    r = train_one("stgnn_adaptive", m, loaders, mean_t, std_t, A_hat)
    r["adjacency"] = "Adaptive"
    results.append(r)
    A_learned = m.get_learned_adj().numpy()
    np.save(RES / "learned_adj.npy", A_learned.astype(np.float32))
    step = max(1, N // 180)
    heatmap_png(A_learned[::step, ::step], FIG / "q5_learned_adj.png")

    # Hybrid
    m = STGNN(N, IN_STEPS, HIDDEN, OUT_STEPS, mode="hybrid", topk=15, alpha=0.5).to(DEVICE)
    r = train_one("stgnn_hybrid", m, loaders, mean_t, std_t, A_hat)
    r["adjacency"] = "Hybrid"
    results.append(r)
    alpha = float(torch.sigmoid(m.alpha).detach().cpu())
    r["alpha"] = alpha

    # comparison table
    table = []
    for r in results:
        table.append(
            {
                "Model": "Temporal-only" if r["model"] == "temporal_only" else "STGNN",
                "Adjacency": r["adjacency"],
                "MAE": round(r["test_mae"], 4),
                "RMSE": round(r["test_rmse"], 4),
                "Val_MAE": round(r["val_mae"], 4),
                "Val_RMSE": round(r["val_rmse"], 4),
                "Training_time_sec": round(r["train_time_sec"], 1),
            }
        )
    (RES / "comparison_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (RES / "train_details.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "epochs": EPOCHS,
                "hidden": HIDDEN,
                "batch": BATCH,
                "lr": LR,
                "in_steps": IN_STEPS,
                "out_steps": OUT_STEPS,
                "mean": mean,
                "std": std,
                "n_train_windows": len(train_idx),
                "n_val_windows": len(val_ds),
                "n_test_windows": len(test_ds),
                "device": str(DEVICE),
                "results": results,
                "hardware_note": "see report",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # predicted vs actual plots for hybrid (best effort)
    best_name = min(results, key=lambda z: z["test_mae"])["model"]
    if best_name == "temporal_only":
        best = TemporalOnly(IN_STEPS, HIDDEN, OUT_STEPS).to(DEVICE)
    else:
        mode = {"stgnn_fixed": "fixed", "stgnn_adaptive": "adaptive", "stgnn_hybrid": "hybrid"}[best_name]
        best = STGNN(N, IN_STEPS, HIDDEN, OUT_STEPS, mode=mode, topk=15).to(DEVICE)
    best.load_state_dict(torch.load(WTS / f"{best_name}.pt", map_location=DEVICE))
    # pick sensors and a test time
    means = X.mean(0)
    sensors = np.argsort(means)[-3:][::-1].tolist()
    t_end = test_idx[len(test_idx) // 2]
    curves = forecast_curves(best, X, mean_t, std_t, A_hat, sensors, t_end)
    (RES / "forecast_curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    for s, payload in curves.items():
        hist = np.asarray(payload["history"])
        act = np.asarray(payload["actual"])
        pred = np.asarray(payload["pred"])
        # concat history + horizon for display
        line_plot_png(
            [np.concatenate([hist, act]), np.concatenate([hist, pred])],
            ["actual", "pred"],
            FIG / f"q6_sensor_{s}_pred.png",
        )

    # graph structure analysis
    A_bin = (A > 0).astype(np.float32)
    L = A_learned
    # strongest learned edges
    flat = np.argpartition(L.ravel(), -50)[-50:]
    pairs = [(int(i // N), int(i % N)) for i in flat]
    overlap = sum(1 for i, j in pairs if i != j and A_bin[i, j] > 0)
    analysis = {
        "topk_learned_pairs_checked": 50,
        "overlap_with_road_graph": overlap,
        "overlap_frac": overlap / 50,
        "learned_edge_mean_on_road": float(L[A_bin > 0].mean()) if (A_bin > 0).any() else None,
        "learned_edge_mean_off_road": float(L[A_bin == 0].mean()),
        "hybrid_alpha": results[-1].get("alpha"),
    }
    (RES / "graph_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(table, indent=2))
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()

