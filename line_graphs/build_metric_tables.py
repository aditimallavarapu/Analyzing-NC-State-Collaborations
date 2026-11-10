"""Rebuild the collaboration-metric tables behind the line graphs.

Inputs (this folder), one row per unit pair per window, with raw counts
C_ij (collaborations), N_i/N_j (faculty), P_i/P_j (papers):
  table8_college_normalized_metrics.csv
  table9_department_normalized_metrics.csv

Outputs (written to the folder given as argv[1], default ./rebuilt):
  table8_college_metrics_summary_by_block.csv
  table8_college_metrics_summary.csv
  table9_department_metrics_summary_by_block.csv
  table9_department_metrics_summary.csv

Metric definitions (recovered from the stored per-pair values, and re-checked
against them below):
  SNCI = C / (N_i * N_j)
  PNCI = C / sqrt(P_i * P_j)
  FNCI = C / (N_i * N_j * sqrt(P_i * P_j))
  CIR  = C / (P_i + P_j - C)

A summary value is the unweighted mean of the per-pair metric over the pairs
of that type in that window that have at least one collaboration (C > 0).
Pairs with C == 0 are dropped, so the number of pairs averaged changes from
window to window. Pass --include-zero to average over every pair instead.
"""
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
METRICS = ["SNCI", "PNCI", "FNCI", "CIR"]
BLOCKS = ["2006-2010", "2011-2015", "2016-2020", "2021-2025"]
FULL = "2006-2025"

LEVELS = {
    "table8_college": dict(src="table8_college_normalized_metrics.csv", type_col="pair_type",
                           intra={"intra"}),
    "table9_department": dict(src="table9_department_normalized_metrics.csv", type_col="colab_type",
                              intra={"intra", "same"}),
}


def compute(c, ni, nj, pi, pj):
    snci = c / (ni * nj)
    pnci = c / math.sqrt(pi * pj)
    return {
        "SNCI": snci,
        "PNCI": pnci,
        "FNCI": snci / math.sqrt(pi * pj),
        "CIR": c / (pi + pj - c),
    }


def load(path, type_col, intra_types):
    rows = []
    for r in csv.DictReader(open(path, newline="")):
        c, ni, nj, pi, pj = (float(r[k]) for k in ("C_ij", "N_i", "N_j", "P_i", "P_j"))
        rows.append(dict(window=r["window"],
                         kind="intra" if r[type_col] in intra_types else "inter",
                         c=c, stored={m: float(r[m]) for m in METRICS},
                         calc=compute(c, ni, nj, pi, pj) if c > 0 else {m: 0.0 for m in METRICS}))
    return rows


def check_formulas(rows, name):
    worst = 0.0
    for r in rows:
        for m in METRICS:
            a, b = r["stored"][m], r["calc"][m]
            worst = max(worst, abs(a - b) / max(abs(a), 1e-300) if a else abs(b))
    print(f"{name}: recomputed vs stored per-pair metrics, max relative diff = {worst:.2e}")


def summarize(rows, window, kind, include_zero):
    sel = [r for r in rows if r["window"] == window and r["kind"] == kind and (include_zero or r["c"] > 0)]
    return len(sel), {m: sum(r["calc"][m] for r in sel) / len(sel) for m in METRICS}


def write(path, header, records):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(records)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    include_zero = "--include-zero" in sys.argv
    out = Path(args[0]) if args else HERE / "rebuilt"
    out.mkdir(parents=True, exist_ok=True)
    for name, cfg in LEVELS.items():
        rows = load(HERE / cfg["src"], cfg["type_col"], cfg["intra"])
        check_formulas(rows, name)
        by_block, full = [], []
        for kind_idx, kind in enumerate(("intra", "inter")):
            n, s = summarize(rows, FULL, kind, include_zero)
            full.append([kind] + [repr(s[m]) for m in METRICS])
        for b in BLOCKS:
            for kind in ("intra", "inter"):
                n, s = summarize(rows, b, kind, include_zero)
                print(f"  {name} {b} {kind}: averaged over {n} pairs")
                by_block.append([b, kind] + [repr(s[m]) for m in METRICS])
        write(out / f"{name}_metrics_summary_by_block.csv", ["block", "colab_type"] + METRICS, by_block)
        write(out / f"{name}_metrics_summary.csv", ["colab_type"] + METRICS, full)
    print("wrote", out)


if __name__ == "__main__":
    main()
