"""
02_descriptive_stats.py

Reads the cleaned tables from step 01 and produces the descriptive-stats
deliverables for the Results section: reproduces your existing Table 2 /
Figure 2 (college-level output) and Table 3 (team size by time block), plus
department-level versions and a few additional tables that later sections
(normalized metrics, network construction, topic modeling) will need.

Usage:
    python 02_descriptive_stats.py --clean clean/ --outdir results/
"""

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

BLOCK_EDGES = [2006, 2011, 2016, 2021, 2026]  # 2006-2010, 2011-2015, 2016-2020, 2021-2025
BLOCK_LABELS = ["2006-2010", "2011-2015", "2016-2020", "2021-2025"]


def assign_block(year: int) -> str:
    for i in range(len(BLOCK_EDGES) - 1):
        lo, hi = BLOCK_EDGES[i], BLOCK_EDGES[i + 1]
        if lo <= year < hi:
            return BLOCK_LABELS[i]
    return "outside_window"  # catches 2005 stragglers / 2026 partial year


def unit_level_output_table(long_df: pd.DataFrame, unit_col: str) -> pd.DataFrame:
    """Replicates your Table 2 / Figure 2 logic at any organizational level
    (college or department).

    Reports BOTH counting conventions, since they answer different questions
    and your existing Table 2 numbers (e.g. 25,867 for Engineering) match the
    authorship-instance convention, not distinct publications:
      - n_publications_distinct: count each publication once per unit, even
        if multiple co-authors from that unit worked on it.
      - n_authorship_instances: count once per (publication, faculty) pair,
        so a paper with 3 co-authors from the same college adds 3 to that
        college's total. This is the convention your current Table 2 uses.
    """
    pubs_distinct = long_df.groupby(unit_col)["pub_id"].nunique().rename("n_publications_distinct")
    instances = long_df.groupby(unit_col).size().rename("n_authorship_instances")
    faculty_per_unit = long_df.groupby(unit_col)["uid"].nunique().rename("n_faculty")
    table = pd.concat([pubs_distinct, faculty_per_unit, instances], axis=1).reset_index()
    table["per_faculty_output_distinct"] = table["n_publications_distinct"] / table["n_faculty"]
    table["per_faculty_output_instances"] = table["n_authorship_instances"] / table["n_faculty"]

    table = table[[
        unit_col, "n_publications_distinct", "n_faculty", "per_faculty_output_distinct",
        "n_authorship_instances", "per_faculty_output_instances",
    ]]
    return table.sort_values("n_publications_distinct", ascending=False)


def team_size_by_block(pubs: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """Replicates your Table 3: publication count, average team size, and
    average faculty share of the team, per 5-year block.
    """
    pubs = pubs.copy()
    pubs["block"] = pubs["year"].apply(assign_block)
    pubs = pubs[pubs["block"] != "outside_window"]

    faculty_authors_per_pub = long_df.groupby("pub_id")["uid"].nunique().rename("n_faculty_authors")
    pubs = pubs.merge(faculty_authors_per_pub, on="pub_id", how="left")
    pubs["n_faculty_authors"] = pubs["n_faculty_authors"].fillna(0)

    pubs["faculty_share"] = pubs["n_faculty_authors"] / pubs["team_size"].replace(0, pd.NA)

    out = pubs.groupby("block").agg(
        n_publications=("pub_id", "count"),
        avg_team_size=("team_size", "mean"),
        avg_faculty_share=("faculty_share", "mean"),
    ).reindex(BLOCK_LABELS)
    return out.reset_index()


def temporal_participation(long_df: pd.DataFrame) -> pd.DataFrame:
    """Active faculty and publication counts per block — the input series
    for the phase-identification changepoint analysis in a later step.
    """
    long_df = long_df.copy()
    long_df["block"] = long_df["year"].apply(assign_block)
    long_df = long_df[long_df["block"] != "outside_window"]

    out = long_df.groupby("block").agg(
        n_publications=("pub_id", "nunique"),
        n_active_faculty=("uid", "nunique"),
    ).reindex(BLOCK_LABELS).reset_index()
    out["pubs_per_active_faculty"] = out["n_publications"] / out["n_active_faculty"]
    return out


def publications_per_capita_by_year_college(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per-year publication output per faculty member for each college.

    This is a direct answer to the per-capita output question: how many distinct
    publications did a college produce per active faculty member in each year.
    The study window is fixed to 2006-2025, matching the cleaned publication
    dataset and excluding all other years.
    """
    long_df = long_df.copy()
    long_df = long_df[(long_df["year"] >= 2006) & (long_df["year"] <= 2025)].copy()

    out = (
        long_df.groupby(["year", "college"], as_index=False)
        .agg(
            n_publications_distinct=("pub_id", "nunique"),
            n_active_faculty=("uid", "nunique"),
        )
    )

    out["publications_per_capita"] = (
        out["n_publications_distinct"] / out["n_active_faculty"]
    )

    return out.sort_values(["year", "college"]).reset_index(drop=True)


def publications_per_capita_by_department_total(long_df: pd.DataFrame) -> pd.DataFrame:
    """Total per-capita publications for each department across 2006-2025.

    Returns one row per department with total distinct publications,
    active faculty, and publications per faculty.
    """
    df = long_df.copy()
    df = df[(df["year"] >= 2006) & (df["year"] <= 2025)].copy()
    out = (
        df.groupby(["department"], as_index=False)
        .agg(
            n_publications_distinct=("pub_id", "nunique"),
            n_active_faculty=("uid", "nunique"),
        )
    )

    out["publications_per_capita"] = out["n_publications_distinct"] / out["n_active_faculty"]
    return out.sort_values("n_publications_distinct", ascending=False).reset_index(drop=True)


def college_collab_matrix_by_block(long_df: pd.DataFrame) -> pd.DataFrame:
    """Produce a block-wise list of unique college pairs and their
    publication counts. Returns columns: `block`, `college_x`, `college_y`,
    `n_publications`. Each unordered pair appears once per block (A,B same
    as B,A).
    """
    df = long_df.copy()
    df = df[(df["year"] >= 2006) & (df["year"] <= 2025)].copy()
    counts = {}

    for pub_id, grp in df.groupby("pub_id"):
        if grp.empty:
            continue
        block = assign_block(int(grp["year"].iloc[0]))
        if block == "outside_window":
            continue

        present_colleges = sorted(grp["college"].dropna().unique())
        if len(present_colleges) < 2:
            continue

        seen_pairs = set()
        for a, b in combinations(present_colleges, 2):
            pair = (a, b) if a <= b else (b, a)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            counts.setdefault((block, pair[0], pair[1]), 0)
            counts[(block, pair[0], pair[1])] += 1

    rows = [
        {"block": k[0], "college_x": k[1], "college_y": k[2], "n_publications": v}
        for k, v in counts.items()
    ]

    if not rows:
        return pd.DataFrame(columns=["block", "college_x", "college_y", "n_publications"])

    out = pd.DataFrame(rows)
    out["block"] = pd.Categorical(out["block"], categories=BLOCK_LABELS, ordered=True)
    out = out.sort_values(["block", "n_publications"], ascending=[True, False]).reset_index(drop=True)
    return out


def department_pair_counts_by_block(long_df: pd.DataFrame) -> pd.DataFrame:
    """Return department-level collaboration counts per block as rows:
    block, dept_x, dept_y, colab_type (intra/inter), n_publications.

    Inter-department:
        Department X from college A and department Y from college B.

    Intra-department:
        Two different departments from the same college.

    Each publication contributes at most 1 to a given dept pair in a block.
    """
    df = long_df.copy()
    df = df[(df["year"] >= 2006) & (df["year"] <= 2025)].copy()
    counts = {}

    for pub_id, grp in df.groupby("pub_id"):
        if grp.empty:
            continue
        block = assign_block(int(grp["year"].iloc[0]))
        if block == "outside_window":
            continue

        authors = grp[["uid", "college", "department"]].drop_duplicates("uid")
        if len(authors) < 2:
            continue

        seen_pairs = set()
        for a, b in combinations(authors.itertuples(index=False), 2):
            da = a.department if pd.notna(a.department) else ""
            db = b.department if pd.notna(b.department) else ""

            if da == "" or db == "":
                continue

            # Intra-department collaboration:
            # two different departments from the same college.
            if a.college == b.college and da != db:
                pair = (da, db) if da <= db else (db, da)
                colab_type = "intra"

            # Inter-department collaboration:
            # department X from college A and department Y from college B.
            elif a.college != b.college:
                pair = (da, db) if da <= db else (db, da)
                colab_type = "inter"

            # Same department within the same college is neither.
            else:
                continue

            key = (block, pair[0], pair[1], colab_type)
            seen_pairs.add(key)

        for key in seen_pairs:
            counts.setdefault(key, 0)
            counts[key] += 1

    rows = [
        {"block": k[0], "dept_x": k[1], "dept_y": k[2], "colab_type": k[3], "n_publications": v}
        for k, v in counts.items()
    ]

    if not rows:
        return pd.DataFrame(columns=["block", "dept_x", "dept_y", "colab_type", "n_publications"])

    out = pd.DataFrame(rows)
    out["block"] = pd.Categorical(out["block"], categories=BLOCK_LABELS, ordered=True)
    out = out.sort_values(["block", "n_publications"], ascending=[True, False]).reset_index(drop=True)
    return out


def top_collabs_exports(long_df: pd.DataFrame, dept_pair_block_df: pd.DataFrame, outdir: Path):
    """Produce CSVs with top collaborations per college and overall.
    - `top2_dept_collabs_by_college.csv`: for each college, top 2 inter
      and top 2 intra department collaborations (aggregated across blocks).
    - `top2_dept_collabs_overall.csv`: 4 rows (top2 inter, top2 intra) at
      block-level with the block in which they occurred.
    """
    # department -> college mapping (most common observed college)
    mapping = (
        long_df.groupby("department")["college"]
        .agg(lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) > 0 else "")
        .to_dict()
    )

    agg_pairs = (
        dept_pair_block_df.groupby(["dept_x", "dept_y", "colab_type"], as_index=False)
        .agg(n_publications=("n_publications", "sum"))
    )

    colleges = sorted(set(mapping.values()))
    rows = []

    for college in colleges:
        # inter: exactly one dept maps to this college
        inter_mask = agg_pairs.apply(
            lambda r: (mapping.get(r["dept_x"], "") == college) ^
                      (mapping.get(r["dept_y"], "") == college),
            axis=1
        )
        inter_candidates = agg_pairs[inter_mask].sort_values("n_publications", ascending=False).head(2)

        for _, r in inter_candidates.iterrows():
            rows.append({
                "college": college,
                "colab_type": "inter",
                "dept_x": r["dept_x"],
                "dept_y": r["dept_y"],
                "n_publications": int(r["n_publications"])
            })

        # intra: both depts map to this college
        intra_mask = agg_pairs.apply(
            lambda r: (mapping.get(r["dept_x"], "") == college) and
                      (mapping.get(r["dept_y"], "") == college),
            axis=1
        )
        intra_candidates = agg_pairs[intra_mask].sort_values("n_publications", ascending=False).head(2)

        for _, r in intra_candidates.iterrows():
            rows.append({
                "college": college,
                "colab_type": "intra",
                "dept_x": r["dept_x"],
                "dept_y": r["dept_y"],
                "n_publications": int(r["n_publications"])
            })

    per_college_df = pd.DataFrame(rows)

    if per_college_df.empty:
        per_college_df = pd.DataFrame(
            columns=["college", "colab_type", "dept_x", "dept_y", "n_publications"]
        )

    per_college_df.to_csv(outdir / "top2_dept_collabs_by_college.csv", index=False)

    inter_overall = (
        dept_pair_block_df[dept_pair_block_df["colab_type"] == "inter"]
        .sort_values("n_publications", ascending=False)
        .head(2)
    )

    intra_overall = (
        dept_pair_block_df[dept_pair_block_df["colab_type"] == "intra"]
        .sort_values("n_publications", ascending=False)
        .head(2)
    )

    overall = pd.concat([inter_overall, intra_overall], ignore_index=True)[
        ["block", "colab_type", "dept_x", "dept_y", "n_publications"]
    ]

    overall.to_csv(outdir / "top2_dept_collabs_overall.csv", index=False)


def unit_block_collaboration_summary(long_df: pd.DataFrame, unit_col: str) -> pd.DataFrame:
    """For a unit (college or department), count inter-unit collaborations by
    5-year block, separating inter-college and inter-department collaboration
    activity.

    Each cross-unit pair contributes +1 to both units involved in that pair.
    This gives a more interpretable block-level contribution measure than a single
    total cross-unit count. Only 2006-2025 publications are included.
    """
    long_df = long_df.copy()
    long_df = long_df[(long_df["year"] >= 2006) & (long_df["year"] <= 2025)].copy()
    counts = {}

    for pub_id, grp in long_df.groupby("pub_id"):
        if grp.empty:
            continue

        block = assign_block(int(grp["year"].iloc[0]))
        if block == "outside_window":
            continue

        authors = grp[["uid", "college", "department", unit_col]].drop_duplicates("uid")

        if len(authors) < 2:
            continue

        for (a, b) in combinations(authors.itertuples(index=False), 2):
            a_unit = getattr(a, unit_col)
            b_unit = getattr(b, unit_col)

            if a.college != b.college:
                for unit_value in {a_unit, b_unit}:
                    entry = counts.setdefault((block, unit_value), {
                        "block": block,
                        unit_col: unit_value,
                        "n_intercollege_collaborations": 0,
                        "n_interdepartment_collaborations": 0,
                    })
                    entry["n_intercollege_collaborations"] += 1

            # Inter-department:
            # departments from different colleges.
            if a.college != b.college:
                for unit_value in {a_unit, b_unit}:
                    entry = counts.setdefault((block, unit_value), {
                        "block": block,
                        unit_col: unit_value,
                        "n_intercollege_collaborations": 0,
                        "n_interdepartment_collaborations": 0,
                    })
                    entry["n_interdepartment_collaborations"] += 1

            # Intra-department:
            # two different departments from the same college.
            elif a.college == b.college and a.department != b.department:
                for unit_value in {a_unit, b_unit}:
                    entry = counts.setdefault((block, unit_value), {
                        "block": block,
                        unit_col: unit_value,
                        "n_intercollege_collaborations": 0,
                        "n_interdepartment_collaborations": 0,
                    })
                    entry["n_interdepartment_collaborations"] += 1

    out = pd.DataFrame(counts.values())

    if out.empty:
        return pd.DataFrame(
            columns=[
                "block",
                unit_col,
                "n_intercollege_collaborations",
                "n_interdepartment_collaborations"
            ]
        )

    out["block"] = pd.Categorical(out["block"], categories=BLOCK_LABELS, ordered=True)
    out = out.sort_values(["block", unit_col]).reset_index(drop=True)

    return out[
        [
            "block",
            unit_col,
            "n_intercollege_collaborations",
            "n_interdepartment_collaborations"
        ]
    ]


def collaboration_counts(long_df: pd.DataFrame) -> dict:
    """Reproduces the summary sentence style you already have in the draft:
    total collaborations, unique faculty pairs, intra/inter college and
    department collaboration counts.
    """
    from itertools import combinations

    n_faculty = long_df["uid"].nunique()
    n_pubs = long_df["pub_id"].nunique()

    pairs_college = set()
    pairs_dept = set()
    intra_college = 0
    inter_college = 0
    intra_dept = 0
    inter_dept = 0

    for pub_id, grp in long_df.groupby("pub_id"):
        authors = grp[["uid", "college", "department"]].drop_duplicates("uid")

        if len(authors) < 2:
            continue

        for (a, b) in combinations(authors.itertuples(index=False), 2):
            uid_pair = tuple(sorted((a.uid, b.uid)))
            pairs_college.add(uid_pair)

            if a.college == b.college:
                intra_college += 1
            else:
                inter_college += 1

            # Intra-department:
            # two different departments from the same college.
            if a.college == b.college and a.department != b.department:
                intra_dept += 1

            # Inter-department:
            # department X from college A and department Y from college B.
            elif a.college != b.college:
                inter_dept += 1

    return {
        "n_faculty_total": n_faculty,
        "n_publications_total": n_pubs,
        "n_unique_faculty_pairs": len(pairs_college),
        "collaborative_instances_intra_college": intra_college,
        "collaborative_instances_inter_college": inter_college,
        "collaborative_instances_intra_department": intra_dept,
        "collaborative_instances_inter_department": inter_dept,
    }


def missingness_report(pubs: pd.DataFrame) -> pd.DataFrame:
    n = len(pubs)
    miss = pubs.isnull().sum()

    return pd.DataFrame({
        "column": miss.index,
        "n_missing": miss.values,
        "pct_missing": (100 * miss.values / n).round(2),
    }).sort_values("pct_missing", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    clean_dir = Path(args.clean)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pubs = pd.read_csv(clean_dir / "publications_clean.csv")
    long_df = pd.read_csv(clean_dir / "faculty_authorship_long.csv")

    print("Table: college-level output (replicates Table 2 / Figure 2)")
    college_table = unit_level_output_table(long_df, "college")
    college_table.to_csv(outdir / "table2_college_output.csv", index=False)
    print(college_table.to_string(index=False))

    print("\nTable: department-level output")
    dept_table = unit_level_output_table(long_df, "department")
    dept_table.to_csv(outdir / "table2b_department_output.csv", index=False)

    print("\nTable: team size by block (replicates Table 3)")
    team_table = team_size_by_block(pubs, long_df)
    team_table.to_csv(outdir / "table3_team_size_by_block.csv", index=False)
    print(team_table.to_string(index=False))

    print("\nTable: temporal participation by block")
    part_table = temporal_participation(long_df)
    part_table.to_csv(outdir / "table4_temporal_participation.csv", index=False)
    print(part_table.to_string(index=False))

    print("\nTable: publications per capita by department (2006-2025)")
    per_capita_table = publications_per_capita_by_department_total(long_df)
    per_capita_table.to_csv(
        outdir / "table5_publications_per_capita_by_department.csv",
        index=False
    )
    print(per_capita_table.head(20).to_string(index=False))

    print("\nTable: college collaboration matrix by block")
    college_block_table = college_collab_matrix_by_block(long_df)
    college_block_table.to_csv(
        outdir / "table6_college_collaboration_by_block.csv",
        index=False
    )
    print(college_block_table.head(20).to_string(index=False))

    print("\nTable: department collaboration pairs by block")
    department_block_table = department_pair_counts_by_block(long_df)
    department_block_table.to_csv(
        outdir / "table7_department_collaboration_by_block.csv",
        index=False
    )
    print(department_block_table.head(40).to_string(index=False))

    print("\nTop collaborations exports")
    top_collabs_exports(long_df, department_block_table, outdir)

    print("\nCollaboration summary counts")
    collab = collaboration_counts(long_df)
    pd.Series(collab).to_csv(outdir / "collaboration_summary.csv")

    for k, v in collab.items():
        print(f"  {k}: {v}")

    print("\nMissingness report")
    miss = missingness_report(pubs)
    miss.to_csv(outdir / "missingness_report.csv", index=False)
    print(miss.to_string(index=False))

    print(f"\nAll tables written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()