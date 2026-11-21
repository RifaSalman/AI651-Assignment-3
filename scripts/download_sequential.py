"""Download LargeST yearly HDF5 one range-part at a time via resolved CDN URL."""
from __future__ import annotations

import math
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "large_st"
NPARTS = 8
SRC = "https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/large_st/{year}/data.h5"


def resolve(year: int):
    url = SRC.format(year=year)
    r = urllib.request.urlopen(url, timeout=120)
    final = r.geturl()
    size = int(r.headers.get("Content-Length") or 0)
    r.close()
    if size <= 0:
        # HEAD fallback
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as h:
            size = int(h.headers["Content-Length"])
            final = h.geturl() if hasattr(h, "geturl") else final
    return final, size


def download_year(year: int):
    out = OUT_ROOT / str(year) / "data.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1_000_000_000:
        print(year, "exists", out.stat().st_size, flush=True)
        return
    part_dir = out.parent / "parts_seq"
    part_dir.mkdir(exist_ok=True)

    cdn, size = resolve(year)
    print(year, "size", size, "cdn_host", urllib.request.urlparse(cdn).netloc, flush=True)
    part_size = math.ceil(size / NPARTS)

    for i in range(NPARTS):
        s = i * part_size
        e = min(size - 1, (i + 1) * part_size - 1)
        part = part_dir / f"part_{i:02d}"
        expect = e - s + 1
        if part.exists() and part.stat().st_size == expect:
            print(year, "skip part", i, flush=True)
            continue
        # refresh CDN URL periodically (signed URLs expire)
        if i % 2 == 0:
            cdn, size2 = resolve(year)
            size = size2
        print(year, f"part {i} bytes {s}-{e}", flush=True)
        # curl range to CDN directly
        cmd = [
            "curl.exe",
            "-L",
            "--retry",
            "20",
            "--retry-delay",
            "3",
            "--connect-timeout",
            "30",
            "-r",
            f"{s}-{e}",
            "-o",
            str(part),
            cdn,
        ]
        for attempt in range(1, 6):
            if part.exists():
                part.unlink()
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.time() - t0
            got = part.stat().st_size if part.exists() else 0
            speed = got / max(dt, 1) / 1e6
            print(
                year,
                f"part {i} attempt {attempt}: got={got} expect={expect} {speed:.2f} MB/s",
                flush=True,
            )
            if got == expect:
                break
            # refresh URL and retry
            cdn, size = resolve(year)
            time.sleep(2 * attempt)
        else:
            raise RuntimeError(f"failed part {i} for {year}")

    tmp = out.with_suffix(".partial")
    with open(tmp, "wb") as fout:
        for i in range(NPARTS):
            fout.write((part_dir / f"part_{i:02d}").read_bytes())
    tmp.replace(out)
    print(year, "DONE", out.stat().st_size, flush=True)


def main():
    for y in (2020, 2021):
        download_year(y)
    print("ALL_DATA_DONE", flush=True)


if __name__ == "__main__":
    main()
