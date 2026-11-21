"""Poll download log; when done, prepare data, train, fill report, build PDF."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "results" / "seq_download.log"
ERR = ROOT / "results" / "pipeline.log"


def run(cmd: list[str]) -> None:
    ERR.parent.mkdir(parents=True, exist_ok=True)
    with open(ERR, "a", encoding="utf-8") as f:
        f.write("\n>>> " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            raise SystemExit(p.returncode)


def main():
    print("waiting for ALL_DATA_DONE...", flush=True)
    while True:
        if LOG.exists() and "ALL_DATA_DONE" in LOG.read_text(encoding="utf-8", errors="ignore"):
            break
        time.sleep(30)
    print("download complete; preparing", flush=True)
    run([PY, str(ROOT / "scripts" / "prepare_sd.py")])
    print("training", flush=True)
    run([PY, str(ROOT / "scripts" / "train_all.py")])
    print("filling report", flush=True)
    run([PY, str(ROOT / "scripts" / "fill_report_table.py")])
    # build PDF with tectonic if available
    tectonic = Path(
        r"D:\LUMS\AI651 Deep Learning for space, time and graphs\Assignments\DL4STG_PA2\DL4STG_PA2 2\report\latex\bin\tectonic.exe"
    )
    if tectonic.exists():
        print("building PDF", flush=True)
        run([str(tectonic), "-o", str(ROOT / "report"), str(ROOT / "report" / "main.tex")])
    print("PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
