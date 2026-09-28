### This is till in-progress. We wil get complete outputs after we perform topic modeling #

"""
05_phase_identification.py

Phase identification via contiguous time-series clustering on metric series.

Computes block-level and rolling 5-year normalized metrics + network
structural metrics directly from clean/faculty_authorship_long.csv (same
conventions as step 3). Phase boundaries are identified by clustering the
block-level (and, as robustness, rolling-window) multivariate metric
series into contiguous temporal segments — not by treating the four
5-year blocks as phases a priori.

Each candidate phase is characterized by:
  (a) structural signature — density, modularity, average FNCI
  (b) thematic signature  — dominant topic strings (topic model is next step;
      for now, top topics from the authorship table)

"Emerging," "maturing," and "saturated" are empirical labels assigned to
ordered phases, not assumed. Robustness: compare phase boundaries on fixed
blocks vs rolling 5-year windows.

Usage:
    python 05_phase_identification.py --clean clean/ --outdir results/
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import ruptures as rpt

# ── Shared study window (matches steps 02 / 03) ─────────────────────────
YEAR_MIN, YEAR_MAX = 2006, 2025
BLOCK_EDGES = [2006, 2011, 2016, 2021, 2026]
BLOCK_LABELS = ["2006-2010", "2011-2015", "2016-2020", "2021-2025"]
ROLL_WIDTH = 5
METRIC_COLS = ["SNCI", "PNCI", "FNCI", "CIR"]
PHASE_LABELS_3 = ["emerging", "maturing", "saturated"]

SIGNAL_COLS = [
    "SNCI_intra", "SNCI_inter", "PNCI_intra", "PNCI_inter",
    "FNCI_intra", "FNCI_inter", "CIR_intra", "CIR_inter",
    "density", "modularity", "inter_intra_ratio_CIR",
]


# ── Window helpers ──────────────────────────────────────────────────────

def block_windows() -> list[tuple[str, int, int]]:
    return [
        (label, BLOCK_EDGES[i], BLOCK_EDGES[i + 1])
        for i, label in enumerate(BLOCK_LABELS)
    ]


def rolling_windows(width: int = ROLL_WIDTH) -> list[tuple[str, int, int]]:
    """5-year windows advanced one year at a time: 2006-2010 … 2021-2025.

    The four fixed blocks are a subset of these labels, so one collaboration
    pass over rolling windows also yields the block series.
    """
    out = []
    for start in range(YEAR_MIN, YEAR_MAX - width + 2):
        end_inclusive = start + width - 1
        out.append((f"{start}-{end_inclusive}", start, end_inclusive + 1))
    return out


def window_midyear(label: str) -> float:
    a, b = label.split("-")
    return (int(a) + int(b)) / 2.0


# ── Metric numerators (same conventions as 03_normalized_metrics) ───────

def canon(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def published_faculty_counts(long_df: pd.DataFrame, unit_col: str) -> dict[str, int]:
    return long_df.groupby(unit_col)["uid"].nunique().astype(int).to_dict()


def distinct_pub_counts(long_df: pd.DataFrame, unit_col: str) -> dict[str, int]:
    return long_df.groupby(unit_col)["pub_id"].nunique().astype(int).to_dict()


def modal_college_by_department(long_df: pd.DataFrame) -> dict[str, str]:
    return (
        long_df.groupby("department")["college"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )


def empty_pair_counter(units: list[str]) -> dict[tuple[str, str], int]:
    return {pair: 0 for pair in combinations_with_replacement(units, 2)}


def collaboration_counts_for_windows(
    long_df: pd.DataFrame,
    unit_col: str,
    units: list[str],
    windows: list[tuple[str, int, int]],
) -> dict[str, dict[tuple[str, str], int]]:
    """C_ii / C_ij per window. N_i and P_i stay 2006-2025 totals (step 3)."""
    counts = {label: empty_pair_counter(units) for label, _, _ in windows}
    year_to_windows: dict[int, list[str]] = {}
    for label, lo, hi in windows:
        for y in range(lo, hi):
            year_to_windows.setdefault(y, []).append(label)

    unit_set = set(units)
    for _, grp in long_df.groupby("pub_id", sort=False):
        year = int(grp["year"].iloc[0])
        if year not in year_to_windows:
            continue

        authors = (
            grp[["uid", unit_col]]
            .dropna(subset=[unit_col])
            .drop_duplicates("uid")
        )
        if authors.empty:
            continue

        faculty_per_unit = authors.groupby(unit_col)["uid"].nunique()
        present = [u for u in faculty_per_unit.index if u in unit_set]
        if not present:
            continue

        intra_units = [u for u in present if faculty_per_unit[u] >= 2]
        inter_pairs = list(combinations(sorted(present), 2))

        for label in year_to_windows[year]:
            c = counts[label]
            for u in intra_units:
                c[(u, u)] += 1
            for a, b in inter_pairs:
                c[canon(a, b)] += 1

    return counts


def metric_row(C: int, n_i: int, n_j: int, p_i: int, p_j: int) -> dict:
    snci_den = n_i * n_j
    pnci_den = np.sqrt(p_i * p_j)
    fnci_den = snci_den * pnci_den
    cir_den = p_i + p_j - C
    return {
        "C_ij": C,
        "SNCI": (C / snci_den) if snci_den > 0 else np.nan,
        "PNCI": (C / pnci_den) if pnci_den > 0 else np.nan,
        "FNCI": (C / fnci_den) if fnci_den > 0 else np.nan,
        "CIR": (C / cir_den) if cir_den > 0 else np.nan,
    }


def summarize_pair_metrics(
    counts: dict[str, dict[tuple[str, str], int]],
    units: list[str],
    n_map: dict[str, int],
    p_map: dict[str, int],
    windows: list[tuple[str, int, int]],
    level: str,
    dept_college: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Mean SNCI/PNCI/FNCI/CIR for intra vs inter pairs with C_ij > 0."""
    rows = []
    for label, _, _ in windows:
        c = counts[label]
        pair_rows = []
        for x, y in combinations_with_replacement(units, 2):
            m = metric_row(
                C=c[(x, y)],
                n_i=n_map.get(x, 0),
                n_j=n_map.get(y, 0),
                p_i=p_map.get(x, 0),
                p_j=p_map.get(y, 0),
            )
            if level == "college":
                colab_type = "intra" if x == y else "inter"
            else:
                assert dept_college is not None
                # same-dept folded into intra (matches step 3 summaries)
                if x == y or dept_college.get(x) == dept_college.get(y):
                    colab_type = "intra"
                else:
                    colab_type = "inter"
            pair_rows.append({
                "colab_type": colab_type,
                "C_ij": m["C_ij"],
                **{k: m[k] for k in METRIC_COLS},
            })

        pdf = pd.DataFrame(pair_rows)
        pdf = pdf[pdf["C_ij"] > 0]
        for ctype in ("intra", "inter"):
            sub = pdf[pdf["colab_type"] == ctype]
            row = {"window": label, "level": level, "colab_type": ctype}
            for k in METRIC_COLS:
                row[k] = float(sub[k].mean()) if len(sub) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ── Network structural metrics ──────────────────────────────────────────

def build_unit_graph(
    counts: dict[tuple[str, str], int],
    units: list[str],
) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(units)
    for (a, b), cval in counts.items():
        if a == b or cval <= 0:
            continue
        g.add_edge(a, b, weight=float(cval))
    return g


def graph_density(g: nx.Graph) -> float:
    if g.number_of_nodes() < 2:
        return np.nan
    return nx.density(g)


def graph_modularity(g: nx.Graph, partition: dict[str, str] | None = None) -> float:
    if g.number_of_edges() == 0:
        return np.nan
    if partition is not None:
        communities: dict[str, set[str]] = {}
        for node in g.nodes:
            comm = partition.get(node, f"_iso_{node}")
            communities.setdefault(comm, set()).add(node)
        comm_sets = [s for s in communities.values() if s]
        if len(comm_sets) < 2:
            return np.nan
        return nx.community.modularity(g, comm_sets, weight="weight")
    try:
        comms = list(nx.community.greedy_modularity_communities(g, weight="weight"))
    except Exception:
        return np.nan
    if len(comms) < 2:
        return np.nan
    return nx.community.modularity(g, comms, weight="weight")


def inter_intra_ratio(intra: pd.Series, inter: pd.Series, metric: str = "CIR") -> float:
    a = intra.get(metric, np.nan)
    b = inter.get(metric, np.nan)
    if pd.isna(a) or pd.isna(b) or a == 0:
        return np.nan
    return float(b) / float(a)


def structural_series_for_level(
    counts: dict[str, dict[tuple[str, str], int]],
    metric_summary: pd.DataFrame,
    units: list[str],
    windows: list[tuple[str, int, int]],
    level: str,
    partition: dict[str, str] | None,
) -> pd.DataFrame:
    rows = []
    for label, _, _ in windows:
        g = build_unit_graph(counts[label], units)
        sub = metric_summary[
            (metric_summary["window"] == label) & (metric_summary["level"] == level)
        ]
        intra = sub[sub["colab_type"] == "intra"]
        inter = sub[sub["colab_type"] == "inter"]
        intra_s = intra.iloc[0] if len(intra) else pd.Series(dtype=float)
        inter_s = inter.iloc[0] if len(inter) else pd.Series(dtype=float)
        rows.append({
            "window": label,
            "level": level,
            "density": graph_density(g),
            "modularity": graph_modularity(g, partition),
            "inter_intra_ratio_CIR": inter_intra_ratio(intra_s, inter_s, "CIR"),
            "inter_intra_ratio_SNCI": inter_intra_ratio(intra_s, inter_s, "SNCI"),
            "n_edges": g.number_of_edges(),
            "n_nodes": g.number_of_nodes(),
        })
    return pd.DataFrame(rows)


def wide_series(metric_summary: pd.DataFrame, structural: pd.DataFrame, level: str) -> pd.DataFrame:
    m = metric_summary[metric_summary["level"] == level].copy()
    pivot = m.pivot(index="window", columns="colab_type", values=METRIC_COLS)
    pivot.columns = [f"{metric}_{ctype}" for metric, ctype in pivot.columns]
    pivot = pivot.reset_index()
    s = structural[structural["level"] == level][[
        "window", "density", "modularity",
        "inter_intra_ratio_CIR", "inter_intra_ratio_SNCI",
    ]]
    out = pivot.merge(s, on="window", how="left")
    out["level"] = level
    out["midyear"] = out["window"].map(window_midyear)
    return out.sort_values("midyear").reset_index(drop=True)


# ── Build all series from clean data ────────────────────────────────────

def build_series_from_clean(
    long_df: pd.DataFrame,
    windows: list[tuple[str, int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalized + network metric series for the given windows, from clean."""
    colleges = sorted(long_df["college"].dropna().unique())
    departments = sorted(long_df["department"].dropna().unique())
    n_college = published_faculty_counts(long_df, "college")
    n_dept = published_faculty_counts(long_df, "department")
    p_college = distinct_pub_counts(long_df, "college")
    p_dept = distinct_pub_counts(long_df, "department")
    dept_college = modal_college_by_department(long_df)

    print(
        f"  {long_df['uid'].nunique()} faculty, "
        f"{len(colleges)} colleges, {len(departments)} departments",
        flush=True,
    )
    print(f"  counting college pairs over {len(windows)} windows...", flush=True)
    college_C = collaboration_counts_for_windows(long_df, "college", colleges, windows)
    print(f"  counting department pairs over {len(windows)} windows...", flush=True)
    dept_C = collaboration_counts_for_windows(long_df, "department", departments, windows)

    college_sum = summarize_pair_metrics(
        college_C, colleges, n_college, p_college, windows, "college"
    )
    dept_sum = summarize_pair_metrics(
        dept_C, departments, n_dept, p_dept, windows, "department", dept_college
    )
    metric_summary = pd.concat([college_sum, dept_sum], ignore_index=True)

    college_struct = structural_series_for_level(
        college_C, metric_summary, colleges, windows, "college", partition=None
    )
    dept_struct = structural_series_for_level(
        dept_C, metric_summary, departments, windows, "department", partition=dept_college
    )
    structural = pd.concat([college_struct, dept_struct], ignore_index=True)

    return (
        metric_summary,
        structural,
        wide_series(metric_summary, structural, "college"),
        wide_series(metric_summary, structural, "department"),
    )


def soft_check_step3(block_metrics: pd.DataFrame, results_dir: Path) -> None:
    """Warn if step-3 summaries exist and disagree; never fails the run."""
    mapping = {
        "college": results_dir / "table8_college_metrics_summary_by_block.csv",
        "department": results_dir / "table9_department_metrics_summary_by_block.csv",
    }
    for level, path in mapping.items():
        if not path.exists():
            print(f"  soft check: {path.name} not found — skip")
            continue
        ref = pd.read_csv(path).rename(columns={"block": "window"})
        ours = block_metrics[block_metrics["level"] == level]
        merged = ours.merge(
            ref, on=["window", "colab_type"], suffixes=("_ours", "_step3"), how="inner"
        )
        if merged.empty:
            print(
                f"  soft check ({level}): no overlapping rows "
                f"(unit lists likely differ — re-run step 3 when clean is final)"
            )
            continue
        diffs = []
        for m in METRIC_COLS:
            diffs.append(float(np.nanmax(np.abs(
                merged[f"{m}_ours"] - merged[f"{m}_step3"]
            ))))
        print(
            f"  soft check ({level}): {len(merged)} rows, "
            f"max |Δ| over {METRIC_COLS} = {max(diffs):.3e}"
        )


# ── Contiguous time-series clustering ───────────────────────────────────

def _zscore_matrix(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return np.nan_to_num((X - mu) / sd, nan=0.0)


def _signal_matrix(series: pd.DataFrame, signal_cols: list[str]) -> np.ndarray:
    cols = [c for c in signal_cols if c in series.columns]
    return _zscore_matrix(series[cols].to_numpy(dtype=float))


def _segment_sse(X: np.ndarray, bkps: list[int]) -> float:
    """Within-segment sum of squared errors for L2 contiguous clusters."""
    total = 0.0
    start = 0
    for end in bkps:
        chunk = X[start:end]
        if len(chunk) == 0:
            continue
        total += float(np.sum((chunk - chunk.mean(axis=0)) ** 2))
        start = end
    return total


def contiguous_cluster(
    X: np.ndarray,
    max_k: int = 3,
    min_size: int = 1,
) -> tuple[list[int], int, float]:
    """Cluster ordered windows into K contiguous phases (Dynp + L2).

    Time-series clustering with a contiguity constraint: each phase is a
    consecutive run of windows. K in {1..max_k} is chosen by BIC on
    within-cluster SSE. Returns (bkps, K, bic).
    """
    n, p = X.shape
    max_k = min(max_k, n)
    best_bkps = [n]
    best_k = 1
    best_bic = _segment_sse(X, best_bkps) + best_k * p * np.log(max(n, 2))

    for k in range(2, max_k + 1):
        n_bkps = k - 1
        if n < k * min_size:
            continue
        bkps = list(
            rpt.Dynp(model="l2", min_size=min_size, jump=1)
            .fit(X)
            .predict(n_bkps=n_bkps)
        )
        sse = _segment_sse(X, bkps)
        bic = sse + k * p * np.log(max(n, 2))
        if bic < best_bic:
            best_bic = bic
            best_bkps = bkps
            best_k = k

    return best_bkps, best_k, float(best_bic)


def bkps_to_boundaries(windows: list[str], bkps: list[int]) -> list[str]:
    return [windows[b] for b in bkps if b < len(windows)]


def segments_from_bkps(windows: list[str], bkps: list[int]) -> list[tuple[str, str, int, int]]:
    starts = [0] + [b for b in bkps if b < len(windows)]
    ends = [b for b in bkps if b < len(windows)] + [len(windows)]
    return [
        (windows[s], windows[e - 1], s, e)
        for s, e in zip(starts, ends)
        if s < e
    ]


def assign_empirical_labels(n_phases: int) -> list[str]:
    if n_phases <= 0:
        return []
    if n_phases == 1:
        return ["undifferentiated"]
    if n_phases == 2:
        return ["emerging", "maturing"]
    if n_phases == 3:
        return list(PHASE_LABELS_3)
    labels = ["maturing"] * n_phases
    labels[0] = "emerging"
    labels[-1] = "saturated"
    return labels


def _avg_fnci(chunk: pd.DataFrame) -> float:
    """Average FNCI across intra and inter (structural signature)."""
    vals = []
    for col in ("FNCI_intra", "FNCI_inter"):
        if col in chunk.columns:
            vals.append(float(chunk[col].mean()))
    if not vals:
        return np.nan
    return float(np.nanmean(vals))


def characterize_segments(
    series: pd.DataFrame,
    segments: list[tuple[str, str, int, int]],
    level: str,
    granularity: str,
) -> pd.DataFrame:
    labels = assign_empirical_labels(len(segments))
    rows = []
    for (w0, w1, s, e), label in zip(segments, labels):
        chunk = series.iloc[s:e]
        rows.append({
            "granularity": granularity,
            "level": level,
            "phase_label": label,
            "phase_index": len(rows) + 1,
            "window_start": w0,
            "window_end": w1,
            "n_windows": e - s,
            "density_mean": float(chunk["density"].mean()),
            "modularity_mean": float(chunk["modularity"].mean()),
            "FNCI_mean": _avg_fnci(chunk),
            "FNCI_intra_mean": float(chunk["FNCI_intra"].mean()) if "FNCI_intra" in chunk else np.nan,
            "FNCI_inter_mean": float(chunk["FNCI_inter"].mean()) if "FNCI_inter" in chunk else np.nan,
            "inter_intra_ratio_CIR_mean": float(chunk["inter_intra_ratio_CIR"].mean())
            if "inter_intra_ratio_CIR" in chunk else np.nan,
        })
    return pd.DataFrame(rows)


def detect_for_series(
    series: pd.DataFrame,
    level: str,
    granularity: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = series["window"].tolist()
    X = _signal_matrix(series, SIGNAL_COLS)
    min_size = 1 if granularity == "block" else 2
    bkps, k, bic = contiguous_cluster(X, max_k=3, min_size=min_size)
    cps = bkps_to_boundaries(windows, bkps)
    segs = segments_from_bkps(windows, bkps)

    if cps:
        cp_rows = [{
            "granularity": granularity,
            "level": level,
            "method": "contiguous_clustering",
            "n_clusters": k,
            "bic": bic,
            "phase_boundary_window": cp,
            "phase_boundary_midyear": window_midyear(cp),
            "n_boundaries": len(cps),
            "n_phases": len(segs),
        } for cp in cps]
    else:
        cp_rows = [{
            "granularity": granularity,
            "level": level,
            "method": "contiguous_clustering",
            "n_clusters": k,
            "bic": bic,
            "phase_boundary_window": None,
            "phase_boundary_midyear": np.nan,
            "n_boundaries": 0,
            "n_phases": len(segs),
        }]

    return pd.DataFrame(cp_rows), characterize_segments(series, segs, level, granularity)


def compare_boundaries(
    block_cps: list[str],
    rolling_cps: list[str],
    level: str,
) -> pd.DataFrame:
    rows = []
    for src, cps in (("block", block_cps), ("rolling", rolling_cps)):
        if not cps:
            rows.append({
                "level": level,
                "granularity": src,
                "phase_boundary_window": None,
                "phase_boundary_midyear": np.nan,
                "n_boundaries": 0,
            })
            continue
        for cp in cps:
            rows.append({
                "level": level,
                "granularity": src,
                "phase_boundary_window": cp,
                "phase_boundary_midyear": window_midyear(cp),
                "n_boundaries": len(cps),
            })
    return pd.DataFrame(rows)


def parse_topics(cell) -> list[str]:
    if pd.isna(cell):
        return []
    return [p.strip() for p in re.split(r";", str(cell)) if p.strip()]


    ############################### Remaining work: Needs Topic Modeling ##############################

def thematic_signature(long_df: pd.DataFrame, phases: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Dominant topics per phase (until dedicated topic-model step)."""
    pub_topics = (
        long_df.dropna(subset=["topics"])
        .groupby("pub_id", as_index=False)
        .agg(year=("year", "first"), topics=("topics", "first"))
    )
    rows = []
    for _, phase in phases.iterrows():
        y0 = int(str(phase["window_start"]).split("-")[0])
        y1 = int(str(phase["window_end"]).split("-")[1])
        mask = (pub_topics["year"] >= y0) & (pub_topics["year"] <= y1)
        counter: Counter[str] = Counter()
        for cell in pub_topics.loc[mask, "topics"]:
            counter.update(parse_topics(cell))
        for rank, (topic, count) in enumerate(counter.most_common(top_k), start=1):
            rows.append({
                "granularity": phase["granularity"],
                "level": phase["level"],
                "phase_label": phase["phase_label"],
                "phase_index": phase["phase_index"],
                "window_start": phase["window_start"],
                "window_end": phase["window_end"],
                "topic_rank": rank,
                "topic": topic,
                "n_publications": count,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="Directory with faculty_authorship_long.csv")
    ap.add_argument(
        "--results",
        default="results",
        help="Optional step-3 dir for soft consistency check",
    )
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    clean_dir = Path(args.clean)
    results_dir = Path(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading authorship table from clean/...", flush=True)
    long_path = clean_dir / "faculty_authorship_long.csv"
    if not long_path.exists():
        raise FileNotFoundError(f"Missing {long_path}")
    long_df = pd.read_csv(long_path)
    long_df = long_df[(long_df["year"] >= YEAR_MIN) & (long_df["year"] <= YEAR_MAX)].copy()
    long_df["college"] = long_df["college"].astype(str).str.strip()
    long_df["department"] = long_df["department"].astype(str).str.strip()

    rolls = rolling_windows(ROLL_WIDTH)
    print(f"\nBuilding metric series ({len(rolls)} rolling windows)...", flush=True)
    all_metrics, all_struct, college_wide, dept_wide = build_series_from_clean(long_df, rolls)

    block_metrics = all_metrics[all_metrics["window"].isin(BLOCK_LABELS)].copy()
    roll_metrics = all_metrics.copy()
    block_struct = all_struct[all_struct["window"].isin(BLOCK_LABELS)].copy()
    roll_struct = all_struct.copy()
    block_college = college_wide[college_wide["window"].isin(BLOCK_LABELS)].copy()
    block_dept = dept_wide[dept_wide["window"].isin(BLOCK_LABELS)].copy()
    roll_college = college_wide.copy()
    roll_dept = dept_wide.copy()

    print("\nSoft check vs step-3 summaries (if present)...", flush=True)
    soft_check_step3(block_metrics, results_dir)

    block_metrics.assign(granularity="block").to_csv(
        outdir / "table10_normalized_metrics_by_block.csv", index=False
    )
    roll_metrics.assign(granularity="rolling").to_csv(
        outdir / "table10_normalized_metrics_rolling.csv", index=False
    )
    block_struct.assign(granularity="block").to_csv(
        outdir / "table11_network_metrics_by_block.csv", index=False
    )
    roll_struct.assign(granularity="rolling").to_csv(
        outdir / "table11_network_metrics_rolling.csv", index=False
    )
    pd.concat([block_college, block_dept], ignore_index=True).assign(
        granularity="block"
    ).to_csv(outdir / "table12_phase_signals_block.csv", index=False)
    pd.concat([roll_college, roll_dept], ignore_index=True).assign(
        granularity="rolling"
    ).to_csv(outdir / "table12_phase_signals_rolling.csv", index=False)

    print("\nRunning contiguous clustering for phase boundaries...", flush=True)
    cp_frames, phase_frames, comparison_frames = [], [], []
    for level, bseries, rseries in (
        ("college", block_college, roll_college),
        ("department", block_dept, roll_dept),
    ):
        b_cp, b_ph = detect_for_series(bseries, level, "block")
        r_cp, r_ph = detect_for_series(rseries, level, "rolling")
        cp_frames.extend([b_cp, r_cp])
        phase_frames.extend([b_ph, r_ph])
        b_list = b_cp.loc[
            b_cp["phase_boundary_window"].notna(), "phase_boundary_window"
        ].tolist()
        r_list = r_cp.loc[
            r_cp["phase_boundary_window"].notna(), "phase_boundary_window"
        ].tolist()
        comparison_frames.append(compare_boundaries(b_list, r_list, level))

    boundaries = pd.concat(cp_frames, ignore_index=True)
    phases = pd.concat(phase_frames, ignore_index=True)
    comparison = pd.concat(comparison_frames, ignore_index=True)

    boundaries.to_csv(outdir / "table13_phase_boundaries.csv", index=False)
    phases.to_csv(outdir / "table14_phases_structural.csv", index=False)
    comparison.to_csv(outdir / "table15_block_vs_rolling_boundaries.csv", index=False)

    print("\nBuilding thematic signatures...", flush=True)
    thematic_signature(long_df, phases, top_k=10).to_csv(
        outdir / "table16_phases_thematic.csv", index=False
    )

    print("\n=== Phase identification (contiguous clustering) ===")
    for gran in ("block", "rolling"):
        for level in ("college", "department"):
            sub = boundaries[
                (boundaries["granularity"] == gran) & (boundaries["level"] == level)
            ]
            cps = sub.loc[
                sub["phase_boundary_window"].notna(), "phase_boundary_window"
            ].tolist()
            n_phases = int(sub["n_phases"].iloc[0]) if len(sub) else 0
            k = int(sub["n_clusters"].iloc[0]) if len(sub) else 0
            print(
                f"  {gran:8s} {level:12s}: {n_phases} phase(s) "
                f"(K={k}), boundaries at {cps if cps else '—'}"
            )

    print("\nPhase structural signatures (density, modularity, avg FNCI):")
    show_cols = [
        "granularity", "level", "phase_index", "phase_label",
        "window_start", "window_end",
        "density_mean", "modularity_mean", "FNCI_mean",
    ]
    print(phases[show_cols].to_string(index=False))

    print(
        f"\nWrote:\n"
        f"  {outdir / 'table10_normalized_metrics_by_block.csv'}\n"
        f"  {outdir / 'table10_normalized_metrics_rolling.csv'}\n"
        f"  {outdir / 'table11_network_metrics_by_block.csv'}\n"
        f"  {outdir / 'table11_network_metrics_rolling.csv'}\n"
        f"  {outdir / 'table12_phase_signals_block.csv'}\n"
        f"  {outdir / 'table12_phase_signals_rolling.csv'}\n"
        f"  {outdir / 'table13_phase_boundaries.csv'}\n"
        f"  {outdir / 'table14_phases_structural.csv'}\n"
        f"  {outdir / 'table15_block_vs_rolling_boundaries.csv'}\n"
        f"  {outdir / 'table16_phases_thematic.csv'}"
    )


if __name__ == "__main__":
    main()
