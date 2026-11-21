"""Prepare LargeST San Diego 2020-2021 at 15-minute resolution."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "large_st"
OUT = ROOT / "data" / "sd_processed"
FIG = ROOT / "figures"
RES = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)


def load_meta_adj():
    meta = pd.read_csv(RAW / "sensors" / "metadata.csv", index_col=0)
    # HF metadata columns may be lower-case
    cols = {c.lower(): c for c in meta.columns}
    district_col = cols.get("district", "district")
    sd_mask = meta[district_col].astype(int).values == 11
    meta_sd = meta.loc[sd_mask].copy()
    adj_pack = np.load(RAW / "sensors" / "adj.npz")
    # keys: edge_index / edge_weight or arrays
    keys = list(adj_pack.keys())
    if "edge_index" in adj_pack and "edge_weight" in adj_pack:
        edge_index = adj_pack["edge_index"]
        edge_weight = adj_pack["edge_weight"]
    else:
        # sometimes stored as arr_0, arr_1 via .values() order
        edge_index, edge_weight = adj_pack[keys[0]], adj_pack[keys[1]]
    n_all = len(meta)
    adj = np.zeros((n_all, n_all), dtype=np.float32)
    adj[tuple(edge_index)] = edge_weight
    adj_sd = adj[sd_mask][:, sd_mask]
    # drop self-loops from raw if present; we add them later when normalizing
    np.fill_diagonal(adj_sd, 0.0)
    return meta_sd, adj_sd, sd_mask


def load_year(year: int, sd_mask: np.ndarray) -> pd.DataFrame:
    """Load yearly readings via h5py (avoids pytables dependency)."""
    import h5py

    path = RAW / str(year) / "data.h5"
    sd_idx = np.where(sd_mask)[0]
    with h5py.File(path, "r") as f:
        # pandas HDF store layout; read only SD columns
        dset = f["readings/block0_values"]  # T x N_all
        if dset.shape[1] != sd_mask.shape[0]:
            raise ValueError(f"unexpected sensor count {dset.shape[1]} vs mask {sd_mask.shape[0]}")
        # h5py needs sorted indices for fancy indexing
        order = np.argsort(sd_idx)
        sorted_idx = sd_idx[order]
        vals_sorted = dset[:, sorted_idx]
        vals = np.empty_like(vals_sorted)
        vals[:, order] = vals_sorted
        t_raw = f["readings/axis1"][:]
    index = pd.to_datetime(t_raw)
    df = pd.DataFrame(vals, index=index)
    # 15-min mean resampling (flow rates -> average over 3×5min bins)
    df15 = df.resample("15min").mean()
    return df15


def main():
    meta_sd, adj_sd, sd_mask = load_meta_adj()
    print("SD nodes", len(meta_sd), "adj", adj_sd.shape)

    parts = []
    for year in (2020, 2021):
        print("loading", year)
        parts.append(load_year(year, sd_mask))
    data = pd.concat(parts, axis=0).sort_index()
    # keep only 2020-01-01 .. 2021-12-31
    data = data.loc["2020-01-01":"2021-12-31"]

    missing_before = float(data.isna().mean().mean())
    # linear interpolate along time per sensor, then fill remaining with 0
    data_filled = data.interpolate(limit_direction="both").fillna(0.0)
    missing_after = float(data_filled.isna().mean().mean())

    X = data_filled.to_numpy(dtype=np.float32)  # T x N
    np.save(OUT / "flow_15min.npy", X)
    np.save(OUT / "adj.npy", adj_sd.astype(np.float32))
    meta_sd.to_csv(OUT / "meta_sd.csv")
    timestamps = data_filled.index
    np.save(OUT / "timestamps.npy", timestamps.to_numpy())

    # graph stats
    nz = int(np.count_nonzero(adj_sd))
    sym_err = float(np.max(np.abs(adj_sd - adj_sd.T)))
    degrees = (adj_sd > 0).sum(axis=1)

    # plots: three sensors, one week
    from src.pngutil import heatmap_png, line_plot_png, write_png

    week = data_filled.loc["2020-06-01":"2020-06-07"]
    # pick three sensors with decent mean flow
    means = data_filled.mean(axis=0).to_numpy()
    picks = np.argsort(means)[-3:][::-1]
    series = [week.iloc[:, int(i)].to_numpy() for i in picks]
    labels = [f"sensor_idx_{int(i)}" for i in picks]
    line_plot_png(series, labels, FIG / "q3_three_sensors_week.png")

    # adjacency heatmap (downsample for image size)
    step = max(1, adj_sd.shape[0] // 180)
    heat = adj_sd[::step, ::step]
    heatmap_png(heat, FIG / "q3_adj_heatmap.png", vmax=float(heat.max() + 1e-8))

    # degree histogram as bars image
    hist, edges = np.histogram(degrees, bins=30)
    # render bar chart
    h, w = 280, 640
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    m = hist.max() if hist.max() > 0 else 1
    bar_w = max(1, (w - 60) // len(hist))
    for i, v in enumerate(hist):
        bh = int(v / m * (h - 50))
        x0 = 30 + i * bar_w
        canvas[h - 30 - bh : h - 30, x0 : x0 + bar_w - 1] = (31, 119, 180)
    write_png(FIG / "q3_degree_hist.png", canvas)

    # one-hop / two-hop for a chosen sensor
    v = int(np.argmax(degrees))
    one = set(np.where(adj_sd[v] > 0)[0].tolist())
    two = set()
    for u in one:
        two.update(np.where(adj_sd[u] > 0)[0].tolist())
    two -= one
    two.discard(v)

    summary = {
        "n_nodes": int(adj_sd.shape[0]),
        "n_nonzero_edges": nz,
        "n_undirected_edges_approx": nz // 2,
        "symmetric_max_abs_diff": sym_err,
        "is_symmetric": sym_err < 1e-6,
        "degree_mean": float(degrees.mean()),
        "degree_max": int(degrees.max()),
        "degree_min": int(degrees.min()),
        "T": int(X.shape[0]),
        "missing_frac_before_fill": missing_before,
        "missing_frac_after_fill": missing_after,
        "aggregation": "15min mean of 5min samples (average flow)",
        "example_sensor": v,
        "one_hop": sorted(one)[:40],
        "one_hop_count": len(one),
        "two_hop_count": len(two),
        "week_sensor_indices": [int(i) for i in picks],
        "split_index_hints": {
            "train_end_2020": str(timestamps[timestamps <= "2020-12-31 23:59:59"][-1]),
            "val_end_2021q1": str(timestamps[timestamps <= "2021-03-31 23:59:59"][-1]),
        },
    }
    (RES / "q3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    main()
