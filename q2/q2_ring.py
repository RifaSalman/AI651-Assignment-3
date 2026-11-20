"""Question 2: undirected fit, bump heatmaps, directed and history solutions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

N = 40


def fit_undirected_isolated():
    """Minimise SSE for isolated slowdown at sensor 20 -> next at 19."""
    # Predictions: only sensors 19,20,21 can be nonzero.
    # y_19 = b, y_20 = a, y_21 = b; target: y_19=1, others 0.
    # SSE = (b-1)^2 + a^2 + b^2 + 0
    # d/da: 2a=0 => a=0
    # d/db: 2(b-1)+2b=0 => 2b-2+2b=0 => b=1/2
    a, b = 0.0, 0.5
    sse = (b - 1) ** 2 + a**2 + b**2
    return {"a": a, "b": b, "sse": sse}


def bump(i0: float = 20.0, scale: float = 8.0):
    i = np.arange(N, dtype=float)
    return np.exp(-((i - i0) ** 2) / scale)


def run_undirected(a: float, b: float, x0: np.ndarray, steps: int = 20):
    seq = [x0.copy()]
    x = x0.copy()
    for _ in range(steps):
        left = np.roll(x, 1)   # x_{i-1}
        right = np.roll(x, -1)  # x_{i+1}
        x = a * x + b * (left + right)
        seq.append(x.copy())
    return np.stack(seq, axis=0)  # (T+1, N)


def run_true(x0: np.ndarray, steps: int = 20):
    seq = [x0.copy()]
    x = x0.copy()
    for _ in range(steps):
        x = np.roll(x, -1)  # against traffic: x_i(t+1)=x_{i+1}(t)
        seq.append(x.copy())
    return np.stack(seq, axis=0)


def save_heatmap_png(arr: np.ndarray, path: Path, title: str, vmax: float):
    """Save a simple heatmap without matplotlib (pure RGB PNG via struct)."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.2))
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xlabel("sensor index")
        ax.set_ylabel("time t")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return
    except Exception:
        pass

    # Fallback: write a crude PPM then note path
    h, w = arr.shape
    norm = np.clip(arr / max(vmax, 1e-8), 0, 1)
    # simple viridis-ish: dark->green->yellow
    r = (norm * 255).astype(np.uint8)
    g = (np.sqrt(norm) * 220).astype(np.uint8)
    b = ((1 - norm) * 180).astype(np.uint8)
    ppm = path.with_suffix(".ppm")
    with open(ppm, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        rgb = np.stack([r, g, b], axis=-1).reshape(-1, 3)
        f.write(rgb.tobytes())
    # Also try PIL
    try:
        from PIL import Image

        img = Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")
        img = img.resize((w * 8, h * 12), Image.NEAREST)
        img.save(path)
    except Exception:
        path = ppm


def verify_directed():
    # a=0, w-=0, w+=1 gives exact backward motion
    a, wm, wp = 0.0, 0.0, 1.0
    x = np.zeros(N)
    x[20] = 1.0
    pred = a * x + wm * np.roll(x, 1) + wp * np.roll(x, -1)
    true = np.roll(x, -1)
    return {
        "a": a,
        "w_minus": wm,
        "w_plus": wp,
        "kernel_[i-1,i,i+1]": [wm, a, wp],
        "max_abs_err": float(np.max(np.abs(pred - true))),
        "boundary": "circular / wrap-around padding (ring)",
    }


def verify_history():
    # true motion: x_i(t+1)=x_{i+1}(t)
    # so x_i(t-1)=x_{i-1}(t)  because x_{i-1}(t)=x_i(t-1)
    # want pred = x_{i+1}(t)
    # with equal neighbour weights: a x_i + b(x_{i-1}+x_{i+1}) + c x_i(t-1)
    # = a x_i + b(x_{i-1}+x_{i+1}) + c x_{i-1}
    # = a x_i + (b+c) x_{i-1} + b x_{i+1}
    # set a=0, b=1, c=-1 => 0 + 0*x_{i-1} + 1*x_{i+1}
    a, b, c = 0.0, 1.0, -1.0
    rng = np.random.default_rng(0)
    x_tm1 = rng.normal(size=N)
    x_t = np.roll(x_tm1, -1)  # one true backward step from t-1
    target = np.roll(x_t, -1)
    pred = a * x_t + b * (np.roll(x_t, 1) + np.roll(x_t, -1)) + c * x_tm1
    # forward motion check with same coeffs
    x_tm1_f = rng.normal(size=N)
    x_t_f = np.roll(x_tm1_f, 1)  # forward: x_i(t)=x_{i-1}(t-1)
    target_f = np.roll(x_t_f, 1)
    pred_f = a * x_t_f + b * (np.roll(x_t_f, 1) + np.roll(x_t_f, -1)) + c * x_tm1_f
    return {
        "a": a,
        "b": b,
        "c": c,
        "backward_max_abs_err": float(np.max(np.abs(pred - target))),
        "forward_max_abs_err": float(np.max(np.abs(pred_f - target_f))),
        "forward_same_coeffs_work": bool(np.allclose(pred_f, target_f)),
    }


def main():
    fit = fit_undirected_isolated()
    x0 = bump()
    und = run_undirected(fit["a"], fit["b"], x0, 20)
    tru = run_true(x0, 20)
    vmax = float(max(und.max(), tru.max()))
    save_heatmap_png(tru, FIG / "q2_true_motion.png", "True backward motion", vmax)
    save_heatmap_png(
        und, FIG / "q2_undirected_layer.png", "Undirected layer (a=0, b=1/2)", vmax
    )

    out = {
        "fit_isolated": fit,
        "directed": verify_directed(),
        "history": verify_history(),
        "bump_true_peak_path": [int(np.argmax(tru[t])) for t in range(tru.shape[0])],
        "bump_und_peak_path": [int(np.argmax(und[t])) for t in range(und.shape[0])],
        "und_energy": [float(np.sum(und[t] ** 2)) for t in range(und.shape[0])],
        "true_energy": [float(np.sum(tru[t] ** 2)) for t in range(tru.shape[0])],
    }
    (RES / "q2_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["fit_isolated"], indent=2))
    print(json.dumps(out["directed"], indent=2))
    print(json.dumps(out["history"], indent=2))
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
