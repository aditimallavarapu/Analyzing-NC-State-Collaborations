"""
03_normalized_metrics.py

Computes SNCI, PNCI, FNCI, and CIR for every college pair and every
department pair, over 2006-2025 and within each 5-year block.

Reads:
  - clean/faculty_authorship_long.csv  (step 1)

N_i is unique publishing faculty (uid) in that file over 2006-2025,
matching Table 2 n_faculty — not the full Faculty.csv roster.

Department pairs (table9) use Table 7's colab_type labels:
  same  — two faculty in the same department (diagonal)
  intra — two different departments in the same college
  inter — departments in different colleges

Summary CSVs collapse pairs to two rows (intra, inter) by taking the
mean of each index over pairs with C_ij > 0. Department `same` is
folded into intra so the summary matches college intra = within-college
vs inter = across-college.

Usage:
    python 03_normalized_metrics.py --clean clean/ --outdir results/
"""

import argparse
from io import StringIO
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK_EDGES = [2006, 2011, 2016, 2021, 2026]
BLOCK_LABELS = ["2006-2010", "2011-2015", "2016-2020", "2021-2025"]
OVERALL_WINDOW = "2006-2025"
WINDOWS = [OVERALL_WINDOW] + BLOCK_LABELS


def assign_block(year: int) -> str:
    for i in range(len(BLOCK_EDGES) - 1):
        lo, hi = BLOCK_EDGES[i], BLOCK_EDGES[i + 1]
        if lo <= year < hi:
            return BLOCK_LABELS[i]
    return "outside_window"


def canon(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def published_faculty_counts(long_df: pd.DataFrame, unit_col: str) -> dict[str, int]:
    """N_i: unique faculty who published in 2006-2025, from the long file."""
    return long_df.groupby(unit_col)["uid"].nunique().astype(int).to_dict()


def modal_college_by_department(long_df: pd.DataFrame) -> dict[str, str]:
    """One college label per department, for tagging only.

    Biological and Agricultural Engineering is split across CALS and
    Engineering; the mode (CALS) is used for college_x/y. N_i counts
    every published faculty member in that department.
    """
    return (
        long_df.groupby("department")["college"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )


def distinct_pub_counts(long_df: pd.DataFrame, unit_col: str) -> dict[str, int]:
    return long_df.groupby(unit_col)["pub_id"].nunique().astype(int).to_dict()


def empty_pair_counter(units: list[str]) -> dict[tuple[str, str], int]:
    return {pair: 0 for pair in combinations_with_replacement(units, 2)}


def collaboration_counts(
    long_df: pd.DataFrame,
    unit_col: str,
    units: list[str],
) -> dict[str, dict[tuple[str, str], int]]:
    """C_ii / C_ij per window.

    C_ii: paper has >= 2 matched faculty from unit i (intra-unit).
    C_ij, i!=j: paper has >= 1 matched faculty from i and from j.
    A paper increments every unordered pair it belongs to, including
    intra and inter at once when both apply.

    Denominators are NOT computed here. Only the numerator C varies
    by window; N_i and P_i are 2006-2025 totals.
    """
    counts = {window: empty_pair_counter(units) for window in WINDOWS}
    unit_set = set(units)

    for _, grp in long_df.groupby("pub_id", sort=False):
        block = assign_block(int(grp["year"].iloc[0]))
        if block == "outside_window":
            continue

        authors = grp[["uid", unit_col]].dropna(subset=[unit_col]).drop_duplicates("uid")
        if authors.empty:
            continue

        faculty_per_unit = authors.groupby(unit_col)["uid"].nunique()
        present = [u for u in faculty_per_unit.index if u in unit_set]
        if not present:
            continue

        intra_units = [u for u in present if faculty_per_unit[u] >= 2]
        inter_pairs = list(combinations(sorted(present), 2))

        for window in (OVERALL_WINDOW, block):
            c = counts[window]
            for u in intra_units:
                c[(u, u)] += 1
            for a, b in inter_pairs:
                c[canon(a, b)] += 1

    return counts


def metric_row(C: int, n_i: int, n_j: int, p_i: int, p_j: int) -> dict:
    """Apply Equations 1-4, including the diagonal (i == j).

    SNCI = C / (N_i * N_j)                         [Eq 1; N_i*N_i on diagonal]
    PNCI = C / sqrt(P_i * P_j)                     [Eq 2]
    FNCI = C / ((N_i * N_j) * sqrt(P_i * P_j))     [Eq 3]
    CIR  = C / (P_i + P_j - C)                     [Eq 4, also on diagonal]

    Undefined (NaN) when the relevant denominator is 0.
    """
    snci_den = n_i * n_j
    pnci_den = np.sqrt(p_i * p_j)
    fnci_den = snci_den * pnci_den
    cir_den = p_i + p_j - C

    return {
        "C_ij": C,
        "N_i": n_i,
        "N_j": n_j,
        "P_i": p_i,
        "P_j": p_j,
        "SNCI": (C / snci_den) if snci_den > 0 else np.nan,
        "PNCI": (C / pnci_den) if pnci_den > 0 else np.nan,
        "FNCI": (C / fnci_den) if fnci_den > 0 else np.nan,
        "CIR": (C / cir_den) if cir_den > 0 else np.nan,
    }


def build_pair_table(
    counts: dict[str, dict[tuple[str, str], int]],
    units: list[str],
    n_map: dict[str, int],
    p_map: dict[str, int],
    unit_x_name: str,
    unit_y_name: str,
) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        c = counts[window]
        for x, y in combinations_with_replacement(units, 2):
            metrics = metric_row(
                C=c[(x, y)],
                n_i=n_map.get(x, 0),
                n_j=n_map.get(y, 0),
                p_i=p_map.get(x, 0),
                p_j=p_map.get(y, 0),
            )
            rows.append({
                "window": window,
                unit_x_name: x,
                unit_y_name: y,
                "pair_type": "intra" if x == y else "inter",
                **metrics,
            })
    return pd.DataFrame(rows)


METRIC_COLS = ["SNCI", "PNCI", "FNCI", "CIR"]


def pair_summary_type(table: pd.DataFrame) -> pd.Series:
    """Map each pair to intra or inter for the 2-row summaries.

    College: uses pair_type (diagonal = intra, off-diagonal = inter).
    Department: Table 7 `same` and `intra` both count as intra
    (within-college); `inter` stays inter.
    """
    if "pair_type" in table.columns:
        return table["pair_type"]
    return np.where(table["colab_type"] == "inter", "inter", "intra")


def summarize_metrics(table: pd.DataFrame, by_block: bool) -> pd.DataFrame:
    """Mean SNCI/PNCI/FNCI/CIR for intra vs inter pairs with C_ij > 0.

    Overall: 2 rows. By block: 2 rows per 5-year block. Missing type/block
    combinations stay in the table with NaN metrics.
    """
    df = table.copy()
    df["colab_type"] = pair_summary_type(df)
    df = df[df["C_ij"] > 0]
    if by_block:
        df = df[df["window"].isin(BLOCK_LABELS)].copy()
        group_cols = ["window", "colab_type"]
        skeleton = pd.MultiIndex.from_product(
            [BLOCK_LABELS, ["intra", "inter"]], names=group_cols
        )
    else:
        df = df[df["window"] == OVERALL_WINDOW].copy()
        group_cols = ["colab_type"]
        skeleton = pd.Index(["intra", "inter"], name="colab_type")

    out = df.groupby(group_cols, observed=True)[METRIC_COLS].mean()
    out = out.reindex(skeleton).reset_index()
    type_order = pd.Categorical(out["colab_type"], categories=["intra", "inter"], ordered=True)
    out["colab_type"] = type_order
    if by_block:
        out["window"] = pd.Categorical(out["window"], categories=BLOCK_LABELS, ordered=True)
        out = out.sort_values(["window", "colab_type"]).reset_index(drop=True)
        out = out.rename(columns={"window": "block"})
        cols = ["block", "colab_type"] + METRIC_COLS
    else:
        out = out.sort_values("colab_type").reset_index(drop=True)
        cols = ["colab_type"] + METRIC_COLS
    return out[cols]


def load_table6_long(path: Path) -> tuple[pd.DataFrame, bool]:
    """Load table6 as block, college_x, college_y, n_publications.

    Supports the original long-form CSV and the stacked n×n matrix CSV.
    For matrices, only the upper triangle plus diagonal is kept so each
    unordered pair (and each C_ii) appears once. Returns (table, is_matrix).
    """
    with path.open() as f:
        first = f.readline()
    if "college_x" in first:
        return pd.read_csv(path), False

    chunks: list[tuple[str, list[str]]] = []
    current_block: str | None = None
    buf: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("block,"):
            if current_block is not None and buf:
                chunks.append((current_block, buf))
            current_block = line.split(",", 1)[1].strip()
            buf = []
        elif line.strip():
            buf.append(line)
    if current_block is not None and buf:
        chunks.append((current_block, buf))

    rows = []
    for block, lines in chunks:
        mat = pd.read_csv(StringIO("\n".join(lines)), index_col=0)
        mat.index = mat.index.astype(str).str.strip()
        mat.columns = mat.columns.astype(str).str.strip()
        for i in mat.index:
            for j in mat.columns:
                if str(i) <= str(j):
                    rows.append({
                        "block": block,
                        "college_x": i,
                        "college_y": j,
                        "n_publications": int(mat.loc[i, j]),
                    })
    return pd.DataFrame(rows), True


def check_against_step2(pair_table: pd.DataFrame, step2_path: Path, level: str) -> None:
    """C_ij by block should match table6 (college) / table7 (dept)."""
    if not step2_path.exists():
        print(f"  skip {level} check; {step2_path.name} not found")
        return

    if level == "college":
        step2, is_matrix = load_table6_long(step2_path)
        ours = pair_table[pair_table["window"].isin(BLOCK_LABELS)].copy()
        if not is_matrix:
            ours = ours[ours["pair_type"] == "inter"].copy()
        ours = ours.rename(columns={"window": "block"})
        merged = ours.merge(
            step2, on=["block", "college_x", "college_y"], how="outer",
            suffixes=("_ours", "_step2"),
        )
        label = "all pairs" if is_matrix else "off-diagonal"
    else:
        step2 = pd.read_csv(step2_path)
        step2["dept_x"] = step2["dept_x"].astype(str).str.strip()
        step2["dept_y"] = step2["dept_y"].astype(str).str.strip()
        ours = pair_table[pair_table["window"].isin(BLOCK_LABELS)].copy()
        ours = ours.rename(columns={"window": "block"})
        merge_keys = ["block", "dept_x", "dept_y"]
        has_same = (
            "colab_type" in step2.columns
            and "same" in set(step2["colab_type"].dropna().astype(str))
        )
        if has_same and "colab_type" in ours.columns:
            merge_keys.append("colab_type")
            label = "same/intra/inter"
        else:
            ours = ours[ours["dept_x"] != ours["dept_y"]].copy()
            if "colab_type" in step2.columns:
                step2 = step2[step2["colab_type"] != "same"].copy()
            step2 = (
                step2.groupby(["block", "dept_x", "dept_y"], as_index=False)["n_publications"]
                .sum()
            )
            label = "off-diagonal (table7 has no 'same' yet)"
        merged = ours.merge(
            step2, on=merge_keys, how="outer", suffixes=("_ours", "_step2"),
        )

    merged["C_ours"] = merged["C_ij"].fillna(0)
    merged["C_step2"] = merged["n_publications"].fillna(0)
    disagree = merged[merged["C_ours"] != merged["C_step2"]]
    n_nonzero_ours = int((ours["C_ij"] > 0).sum())
    print(
        f"  {level} {label} vs {step2_path.name}: "
        f"{len(merged)} compared, {len(disagree)} mismatches, "
        f"{n_nonzero_ours} nonzero pairs in this script"
    )
    if len(disagree) > 0:
        cols = [c for c in [
            "block", "college_x", "college_y", "dept_x", "dept_y", "colab_type",
            "C_ours", "C_step2",
        ] if c in disagree.columns]
        print(disagree[cols].head(10).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    clean_dir = Path(args.clean)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading authorship table...", flush=True)
    long_df = pd.read_csv(clean_dir / "faculty_authorship_long.csv")
    long_df = long_df[(long_df["year"] >= 2006) & (long_df["year"] <= 2025)].copy()
    long_df["college"] = long_df["college"].astype(str).str.strip()
    long_df["department"] = long_df["department"].astype(str).str.strip()

    colleges = sorted(long_df["college"].dropna().unique())
    departments = sorted(long_df["department"].dropna().unique())
    n_college = published_faculty_counts(long_df, "college")
    n_dept = published_faculty_counts(long_df, "department")
    p_college = distinct_pub_counts(long_df, "college")
    p_dept = distinct_pub_counts(long_df, "department")
    dept_college = modal_college_by_department(long_df)

    print(
        f"  published faculty: {long_df['uid'].nunique()}, "
        f"{len(colleges)} colleges, {len(departments)} departments",
        flush=True,
    )
    print("Counting college collaborations...", flush=True)
    college_C = collaboration_counts(long_df, "college", colleges)
    print("Counting department collaborations...", flush=True)
    dept_C = collaboration_counts(long_df, "department", departments)

    college_table = build_pair_table(
        college_C, colleges, n_college, p_college, "college_x", "college_y"
    )
    dept_table = build_pair_table(
        dept_C, departments, n_dept, p_dept, "dept_x", "dept_y"
    )
    dept_table["college_x"] = dept_table["dept_x"].map(dept_college)
    dept_table["college_y"] = dept_table["dept_y"].map(dept_college)
    dept_table["colab_type"] = np.where(
        dept_table["dept_x"] == dept_table["dept_y"],
        "same",
        np.where(
            dept_table["college_x"] == dept_table["college_y"],
            "intra",
            "inter",
        ),
    )
    dept_table = dept_table.drop(columns=["pair_type"])
    dept_cols = [
        "window", "dept_x", "dept_y", "colab_type", "college_x", "college_y",
        "C_ij", "N_i", "N_j", "P_i", "P_j", "SNCI", "PNCI", "FNCI", "CIR",
    ]
    dept_table = dept_table[dept_cols]

    college_path = outdir / "table8_college_normalized_metrics.csv"
    dept_path = outdir / "table9_department_normalized_metrics.csv"
    college_sum_path = outdir / "table8_college_metrics_summary.csv"
    dept_sum_path = outdir / "table9_department_metrics_summary.csv"
    college_block_path = outdir / "table8_college_metrics_summary_by_block.csv"
    dept_block_path = outdir / "table9_department_metrics_summary_by_block.csv"

    college_summary = summarize_metrics(college_table, by_block=False)
    dept_summary = summarize_metrics(dept_table, by_block=False)
    college_by_block = summarize_metrics(college_table, by_block=True)
    dept_by_block = summarize_metrics(dept_table, by_block=True)

    college_table.to_csv(college_path, index=False)
    dept_table.to_csv(dept_path, index=False)
    college_summary.to_csv(college_sum_path, index=False)
    dept_summary.to_csv(dept_sum_path, index=False)
    college_by_block.to_csv(college_block_path, index=False)
    dept_by_block.to_csv(dept_block_path, index=False)

    print("\nSanity checks")
    check_against_step2(college_table, outdir / "table6_college_collaboration_by_block.csv", "college")
    check_against_step2(dept_table, outdir / "table7_department_collaboration_by_block.csv", "department")

    overall_c = college_table[college_table["window"] == OVERALL_WINDOW]
    print(
        f"  college pairs per window: {len(overall_c)} "
        f"({len(colleges)} units, unordered incl. diagonal)"
    )
    print(
        f"  department pairs per window: {len(dept_table) // len(WINDOWS)} "
        f"({len(departments)} units, unordered incl. diagonal)"
    )
    print(
        "  table9 colab_type counts (all windows): "
        + ", ".join(
            f"{k}={v}" for k, v in dept_table["colab_type"].value_counts().items()
        )
    )
    print(
        f"  CIR range (non-NaN): "
        f"{college_table['CIR'].min(skipna=True):.4f} – "
        f"{college_table['CIR'].max(skipna=True):.4f}"
    )

    print("\nCollege intra/inter summary, 2006-2025")
    print(college_summary.to_string(index=False))
    print("\nDepartment intra/inter summary, 2006-2025")
    print(dept_summary.to_string(index=False))

    print(
        f"\nWrote:\n  {college_path}\n  {dept_path}\n"
        f"  {college_sum_path}\n  {dept_sum_path}\n"
        f"  {college_block_path}\n  {dept_block_path}"
    )


if __name__ == "__main__":
    main()
