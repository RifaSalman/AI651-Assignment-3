"""Replace placeholder dashes in report/main.tex with comparison_table.json values."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "report" / "main.tex"
TAB = ROOT / "results" / "comparison_table.json"
CURVES = ROOT / "results" / "forecast_curves.json"
Q3 = ROOT / "results" / "q3_summary.json"


def main():
    table = json.loads(TAB.read_text(encoding="utf-8"))
    rows = []
    for r in table:
        rows.append(
            f"{r['Model']} & {r['Adjacency']} & {r['MAE']:.4f} & {r['RMSE']:.4f} & {r['Training_time_sec']:.1f}\\\\"
        )
    block = "\n".join(rows)
    tex = TEX.read_text(encoding="utf-8")
    # replace the tabular body between midrule and end{tabular} inside Q6 table — simple marker
    pattern = r"(Temporal-only & None & --- & --- & ---\\\\.*?STGNN & Hybrid & --- & --- & ---\\\\)"
    # Build fixed four lines in known order
    ordered = { (r["Model"], r["Adjacency"]): r for r in table }
    keys = [
        ("Temporal-only", "None"),
        ("STGNN", "Fixed"),
        ("STGNN", "Adaptive"),
        ("STGNN", "Hybrid"),
    ]
    lines = []
    for k in keys:
        r = ordered.get(k)
        if r is None:
            # try adjacency field variants
            for rr in table:
                if rr["Adjacency"] == k[1] or (k[1] == "None" and rr["Adjacency"] == "None"):
                    if (k[0] == "Temporal-only" and "temporal" in rr["Model"].lower()) or (
                        k[0] == "STGNN" and "STGNN" in rr["Model"]
                    ):
                        r = rr
                        break
        if r is None:
            lines.append(f"{k[0]} & {k[1]} & --- & --- & ---\\\\")
        else:
            lines.append(
                f"{k[0]} & {k[1]} & {r['MAE']:.4f} & {r['RMSE']:.4f} & {r['Training_time_sec']:.1f}\\\\"
            )
    new_body = "\n".join(lines)
    tex2, n = re.subn(
        r"Temporal-only & None & --- & --- & ---\\\\\s*"
        r"STGNN & Fixed & --- & --- & ---\\\\\s*"
        r"STGNN & Adaptive & --- & --- & ---\\\\\s*"
        r"STGNN & Hybrid & --- & --- & ---\\\\",
        new_body,
        tex,
        count=1,
    )
    if n == 0:
        print("table pattern not found; writing side file")
        (ROOT / "results" / "table_rows.tex").write_text(new_body, encoding="utf-8")
    else:
        TEX.write_text(tex2, encoding="utf-8")
        print("updated", TEX)

    # rename prediction figures to A/B/C if needed
    if CURVES.exists():
        import shutil

        keys_s = list(json.loads(CURVES.read_text(encoding="utf-8")).keys())
        fig = ROOT / "figures"
        for label, s in zip(["A", "B", "C"], keys_s[:3]):
            src = fig / f"q6_sensor_{s}_pred.png"
            dst = fig / f"q6_sensor_{label}_pred.png"
            if src.exists():
                shutil.copyfile(src, dst)
                print("copied", src.name, "->", dst.name)

    if Q3.exists():
        print("q3", json.dumps(json.loads(Q3.read_text(encoding="utf-8")), indent=2)[:500])


if __name__ == "__main__":
    main()
