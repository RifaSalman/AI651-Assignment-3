# AI651 Assignment 3 — Spatio-Temporal GNNs (LargeST-SD)

**Student:** Rifa Salman (`25280059`)

## Setup

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy pandas h5py
```

## Data

LargeST San Diego (district 11), years 2020–2021.

```bash
# downloads ~1.1GB × 2 from HuggingFace (parallel ranges)
python scripts/parallel_download.py

# build 15-min SD tensors + figures
python scripts/prepare_sd.py
```

## Train

```bash
python scripts/train_all.py
```

## Report

```bash
# after training, refresh table numbers then build PDF
python scripts/fill_report_table.py
# compile report/main.tex (tectonic or pdflatex)
```

## Layout

- `q1/`, `q2/` — theory checks and ring-road plots
- `src/` — models + PNG helpers
- `scripts/` — data prep and training
- `results/`, `figures/`, `weights/`
- `report/main.tex` — submission report
