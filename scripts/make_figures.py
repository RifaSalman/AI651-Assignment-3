"""Regenerate all report figures with axes, colorbars, and legends."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DATA = ROOT / "data" / "sd_processed"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


def q2_heatmaps():
    from q2.q2_ring import bump, fit_undirected_isolated, run_true, run_undirected

    fit = fit_undirected_isolated()
    x0 = bump()
    tru = run_true(x0, 20)
    und = run_undirected(fit["a"], fit["b"], x0, 20)
    vmax = float(max(tru.max(), und.max()))

    for arr, path, title in [
        (tru, FIG / "q2_true_motion.png", "True backward motion"),
        (und, FIG / "q2_undirected_layer.png", r"Undirected layer ($a=0$, $b=1/2$)"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xlabel("Sensor index")
        ax.set_ylabel("Time step $t$")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Slowdown intensity")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    print("q2 ok")


def q3_figures():
    X = np.load(DATA / "flow_15min.npy")
    A = np.load(DATA / "adj.npy")
    ts = np.load(DATA / "timestamps.npy")
    ts = ts.astype("datetime64[ns]")

    # three sensors over one week
    week_mask = (ts >= np.datetime64("2020-06-01")) & (ts <= np.datetime64("2020-06-07"))
    week_idx = np.where(week_mask)[0]
    means = X.mean(0)
    picks = np.argsort(means)[-3:][::-1]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    t_hours = (week_idx - week_idx[0]) * 0.25  # 15-min -> hours from week start
    for c, s in zip(colors, picks):
        ax.plot(t_hours, X[week_idx, int(s)], color=c, lw=1.2, label=f"Sensor {int(s)}")
    ax.set_xlabel("Hours from week start (1–7 Jun 2020)")
    ax.set_ylabel("Traffic flow (15-min mean)")
    ax.set_title("Three high-flow San Diego sensors over one week")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "q3_three_sensors_week.png")
    plt.close(fig)

    # adjacency heatmap
    step = max(1, A.shape[0] // 180)
    heat = A[::step, ::step]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(heat, cmap="magma", aspect="auto", origin="upper", vmin=0, vmax=float(heat.max()))
    ax.set_xlabel("Sensor index (downsampled)")
    ax.set_ylabel("Sensor index (downsampled)")
    ax.set_title("SD road-distance adjacency")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Edge weight $A_{ij}$")
    fig.tight_layout()
    fig.savefig(FIG / "q3_adj_heatmap.png")
    plt.close(fig)

    # degree histogram
    deg = (A > 0).sum(axis=1)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.hist(deg, bins=30, color="#1f77b4", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Number of neighbours (degree)")
    ax.set_ylabel("Number of sensors")
    ax.set_title("Degree distribution of SD graph")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "q3_degree_hist.png")
    plt.close(fig)
    print("q3 ok", picks.tolist())


def q5_learned():
    L = np.load(RES / "learned_adj.npy")
    step = max(1, L.shape[0] // 180)
    heat = L[::step, ::step]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(heat, cmap="viridis", aspect="auto", origin="upper", vmin=0, vmax=float(np.percentile(heat, 99)))
    ax.set_xlabel("Sensor index (downsampled)")
    ax.set_ylabel("Sensor index (downsampled)")
    ax.set_title("Learned adaptive adjacency (row-softmax)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Learned edge weight")
    fig.tight_layout()
    fig.savefig(FIG / "q5_learned_adj.png")
    plt.close(fig)
    print("q5 ok")


def q6_forecasts():
    curves = json.loads((RES / "forecast_curves.json").read_text(encoding="utf-8"))
    labels = ["A", "B", "C"]
    for lab, (sid, payload) in zip(labels, curves.items()):
        hist = np.asarray(payload["history"], dtype=float)
        act = np.asarray(payload["actual"], dtype=float)
        pred = np.asarray(payload["pred"], dtype=float)
        n_h = len(hist)
        t_hist = np.arange(-(n_h - 1), 1)  # ..., -1, 0
        t_fut = np.arange(1, len(act) + 1)

        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(t_hist, hist, color="#7f7f7f", lw=1.5, marker="o", ms=3, label="History (input)")
        ax.plot(t_fut, act, color="#1f77b4", lw=1.8, marker="o", ms=4, label="Actual")
        ax.plot(t_fut, pred, color="#d62728", lw=1.8, marker="s", ms=4, label="Predicted")
        ax.axvline(0.5, color="black", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("Lead relative to forecast time (15-min steps)")
        ax.set_ylabel("Traffic flow")
        ax.set_title(f"Sensor {sid}: history + 4-step forecast")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f"q6_sensor_{lab}_pred.png")
        fig.savefig(FIG / f"q6_sensor_{sid}_pred.png")
        plt.close(fig)
    print("q6 ok")


def main():
    import sys

    sys.path.insert(0, str(ROOT))
    q2_heatmaps()
    q3_figures()
    q5_learned()
    q6_forecasts()
    print("all figures written to", FIG)


if __name__ == "__main__":
    main()
