"""Parallel ranged download of LargeST yearly HDF5 files from HuggingFace."""
from __future__ import annotations

import math
import concurrent.futures
import time
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/large_st/{year}/data.h5"
OUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "large_st"
NPARTS = 8
WORKERS = 4


def file_size(u: str) -> int:
    req = urllib.request.Request(u, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def download_range(u: str, start: int, end: int, part_path: Path, retries: int = 5) -> None:
    expect = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expect:
        return
    if part_path.exists():
        part_path.unlink()
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(u)
            req.add_header("Range", f"bytes={start}-{end}")
            with urllib.request.urlopen(req, timeout=300) as r, open(part_path, "wb") as f:
                while True:
                    chunk = r.read(1024 * 512)
                    if not chunk:
                        break
                    f.write(chunk)
            if part_path.stat().st_size == expect:
                return
            last_err = RuntimeError(
                f"size mismatch got {part_path.stat().st_size} expected {expect}"
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"retry {attempt} {part_path.name}: {e}", flush=True)
            time.sleep(2 * attempt)
            if part_path.exists():
                part_path.unlink(missing_ok=True)
    raise RuntimeError(f"failed {part_path}: {last_err}")


def download_year(year: int) -> None:
    u = URL.format(year=year)
    out = OUT_ROOT / str(year) / "data.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1_000_000_000:
        print(year, "already present", out.stat().st_size, flush=True)
        return
    size = file_size(u)
    print(year, "size", size, flush=True)
    part_dir = out.parent / f"parts_{year}"
    part_dir.mkdir(exist_ok=True)
    part_size = math.ceil(size / NPARTS)
    jobs = []
    for i in range(NPARTS):
        s = i * part_size
        e = min(size - 1, (i + 1) * part_size - 1)
        if s <= e:
            jobs.append((i, s, e))

    def run(item):
        i, s, e = item
        p = part_dir / f"part_{i:02d}"
        print(f"{year} start part {i}", flush=True)
        download_range(u, s, e, p)
        print(f"{year} done part {i} ({p.stat().st_size})", flush=True)
        return p

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        parts = list(ex.map(run, jobs))

    tmp = out.with_suffix(".h5.partial")
    with open(tmp, "wb") as fout:
        for p in sorted(parts, key=lambda x: x.name):
            fout.write(p.read_bytes())
    tmp.replace(out)
    print(year, "DONE", out.stat().st_size, flush=True)
    for p in parts:
        p.unlink(missing_ok=True)


def main():
    for y in (2020, 2021):
        download_year(y)
    print("ALL_DATA_DONE", flush=True)


if __name__ == "__main__":
    main()
