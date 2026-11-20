"""Minimal PNG writer (no Pillow required) + PPM->PNG helper."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: Path, rgb: np.ndarray) -> None:
    """rgb: HxWx3 uint8."""
    path = Path(path)
    h, w, c = rgb.shape
    assert c == 3
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))
    compressed = zlib.compress(raw, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.write_bytes(png)


def heatmap_png(arr: np.ndarray, path: Path, vmax: float | None = None) -> None:
    """Viridis-ish heatmap PNG."""
    a = np.asarray(arr, dtype=float)
    if vmax is None:
        vmax = float(np.nanmax(a)) if a.size else 1.0
    vmax = max(vmax, 1e-8)
    t = np.clip(a / vmax, 0, 1)
    # piecewise fake-viridis
    r = np.clip(0.2 + 1.4 * t - 0.6 * (t**2), 0, 1)
    g = np.clip(0.05 + 1.2 * t - 0.3 * (t**3), 0, 1)
    b = np.clip(0.35 + 0.9 * (1 - t) * t + 0.4 * (1 - t), 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    # upscale for readability
    scale_y, scale_x = 12, 8
    rgb = np.repeat(np.repeat(rgb, scale_y, axis=0), scale_x, axis=1)
    write_png(path, rgb)


def line_plot_png(
    series: list[np.ndarray],
    labels: list[str],
    path: Path,
    width: int = 900,
    height: int = 320,
) -> None:
    """Simple multi-line chart on white background."""
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]
    all_y = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series])
    ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    if abs(ymax - ymin) < 1e-8:
        ymax = ymin + 1.0
    pad = 40
    for si, s in enumerate(series):
        y = np.asarray(s, dtype=float).ravel()
        n = len(y)
        xs = pad + (np.arange(n) * (width - 2 * pad - 1) / max(n - 1, 1)).astype(int)
        ys = (height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)).astype(int)
        col = colors[si % len(colors)]
        for i in range(n - 1):
            x0, y0, x1, y1 = xs[i], ys[i], xs[i + 1], ys[i + 1]
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for t in range(steps + 1):
                x = int(x0 + (x1 - x0) * t / steps)
                yy = int(y0 + (y1 - y0) * t / steps)
                if 0 <= yy < height and 0 <= x < width:
                    canvas[yy, x] = col
                    if yy + 1 < height:
                        canvas[yy + 1, x] = col
    write_png(path, canvas)
