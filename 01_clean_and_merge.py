"""
01_clean_and_merge.py

Cleans publications.csv and Faculty.csv and produces:
  - publications_clean.csv      : one row per publication, cleaned
  - faculty_authorship_long.csv : one row per (publication, faculty co-author),
                                   with college/department attached — this is
                                   the base table every later analysis (network
                                   construction, normalized metrics, topic
                                   modeling aggregation) builds on.
  - data_quality_report.txt     : counts of dropped/unmatched rows so the
                                   cleaning is auditable and reproducible.

Usage:
    python 01_clean_and_merge.py \
        --pubs exported_works_faculty-2005-present-2_20260731.csv \
        --faculty Faculty.csv \
        --allowlist depts.txt \
        --outdir clean/
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REAL_PUB_COLS = [
    "title", "authors", "nc_state_people","nc_state_faculty", "nc_state_faculty_active", "DOI", "PMID",
    "year", "url", "topics", "abstract", "openalex_cited_by_count",
]

AUTHOR_RE = re.compile(r"^(.*)\((\w+)\)$")

# Colleges in depts.txt that are not kept in the analysis set.
EXCLUDED_ALLOWLIST_COLLEGES = {
    "NC State Administration and Offices",
    "Interdisciplinary Programs",
    "Graduate School",
}

# Expected unique-pub count after allowlist filter (Option A); for the report note only.
EXPECTED_ALLOWLIST_PUBS = 62949


def load_allowlist(path: Path) -> tuple[set[str], pd.DataFrame]:
    """Load depts.txt (TSV: department, type, college).

    Drops rows whose college is in EXCLUDED_ALLOWLIST_COLLEGES. Returns
    (allowed department names, table of excluded allowlist rows for logging).
    """
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        rows.append({
            "department": parts[0].strip(),
            "unit_type": parts[1].strip() if len(parts) > 1 else "",
            "college": parts[2].strip() if len(parts) > 2 else "",
        })
    df = pd.DataFrame(rows)
    excluded = df[df["college"].isin(EXCLUDED_ALLOWLIST_COLLEGES)].copy()
    kept = df[~df["college"].isin(EXCLUDED_ALLOWLIST_COLLEGES)]
    return set(kept["department"]), excluded


def load_publications(path: str) -> tuple[pd.DataFrame, dict]:
    """Load publications.csv, keeping only the real columns.

    The raw export has ~1,229 columns because of unescaped commas inside a
    small number of abstract/title fields during the source export; the
    first 4 rows are metadata/junk, and the next row contains the real
    schema. We skip those junk rows before parsing and then keep the first
    12 columns, which are the actual publication fields.
    """
    df = pd.read_csv(
        path,
        skiprows=4,
        usecols=range(len(REAL_PUB_COLS)),
        low_memory=False,
    )
    df.columns = REAL_PUB_COLS

    n_total = len(df)
    year_numeric = pd.to_numeric(df["year"], errors="coerce")
    ### AMwhy 1990 and 2027, shouldnt this be 2006-2025? 
    ### The default year window is 2006-2025, but this is just a check for malformed years, 
    ### so it is more lenient. The year window is applied later in the main function.
    malformed_mask = year_numeric.isna() | (year_numeric < 1990) | (year_numeric > 2027)
    n_malformed = int(malformed_mask.sum())

    df_clean = df.loc[~malformed_mask].copy()
    df_clean["year"] = year_numeric.loc[~malformed_mask].astype(int)
    df_clean = df_clean.reset_index(drop=True)
    df_clean["pub_id"] = df_clean.index

    report = {
        "publications_raw_rows": n_total,
        "publications_dropped_malformed_year": n_malformed,
        "publications_clean_rows": len(df_clean),
        "year_min": int(df_clean["year"].min()),
        "year_max": int(df_clean["year"].max()),
    }
    return df_clean, report


def parse_nc_state_faculty(value) -> list[tuple[str, str]]:
    """Parse 'Name (uid); Name (uid)' into [(name, uid), ...]."""
    if pd.isna(value):
        return []
    out = []
    for entry in str(value).split(";"):
        entry = entry.strip()
        if not entry:
            continue
        m = AUTHOR_RE.match(entry)
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out


def count_team_size(authors_field) -> int:
    """Total author count for a publication, from the free-text `authors` field."""
    if pd.isna(authors_field):
        return 0
    return len([a for a in str(authors_field).split(";") if a.strip()])


def build_long_table(pubs: pd.DataFrame, faculty: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per (publication, NC State faculty co-author)."""
    faculty_lookup = faculty.set_index("uid")[["name", "college", "department"]]

    records = []
    n_people_entries = 0
    n_unmatched_uid = 0
    unmatched_uids = set()
    ### where is this pub_id coming from? No such column exists in the publications.csv, is it the name of the publication? 
    for pub_id, nc_people in zip(pubs["pub_id"], pubs["nc_state_faculty"]):
        for name, uid in parse_nc_state_faculty(nc_people):
            n_people_entries += 1
            if uid in faculty_lookup.index:
                row = faculty_lookup.loc[uid]
                records.append({
                    "pub_id": pub_id,  ### where is this pub_id in publications.csv?
                    "uid": uid,
                    "name": name,
                    "college": row["college"],
                    "department": row["department"],
                })
            else:
                n_unmatched_uid += 1
                unmatched_uids.add(uid)

    long_df = pd.DataFrame.from_records(records)
    long_df = long_df.merge(
        pubs[["pub_id", "year", "title", "topics"]], on="pub_id", how="left"
    )

    report = {
        "nc_state_faculty_entries_total": n_people_entries,
        "nc_state_faculty_matched_to_faculty": len(records),
        "nc_state_faculty_unmatched_uid": n_unmatched_uid,
        "unmatched_unique_uids": len(unmatched_uids),
        "unmatched_uids_sample": list(sorted(unmatched_uids))[:20],
    }
    return long_df, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pubs", required=True)
    ap.add_argument("--faculty", required=True)
    ap.add_argument(
        "--allowlist",
        default="depts.txt",
        help="TSV allowlist (department, type, college). "
             "Non-academic colleges listed in EXCLUDED_ALLOWLIST_COLLEGES are dropped.",
    )
    ap.add_argument("--outdir", default="clean")
    ap.add_argument("--year-min", type=int, default=2006,
                     help="Restrict to publications in [year-min, year-max]. "
                          "Default 2006-2025 matches the 66,073-publication "
                          "figure already stated in your intro/methods.")
    ap.add_argument("--year-max", type=int, default=2025)
    args = ap.parse_args()


    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading publications...", flush=True)
    pubs, pub_report = load_publications(args.pubs)

    n_before_window = len(pubs)
    # filter to the year window specified by the user (default 2006-2025)
    pubs = pubs[(pubs["year"] >= args.year_min) & (pubs["year"] <= args.year_max)].reset_index(drop=True)
    # this is where the pub_id is coming from, it is the index of the filtered publications dataframe
    pubs["pub_id"] = pubs.index  # re-index after windowing
    pub_report["year_window_applied"] = f"{args.year_min}-{args.year_max}"
    pub_report["publications_after_year_window"] = len(pubs)
    pub_report["publications_dropped_outside_window"] = n_before_window - len(pubs)


    print("Loading faculty...", flush=True)
    faculty = pd.read_csv(args.faculty)
    # sanity check: this file should already be faculty-only
    if "shib_role" in faculty.columns:
        n_before = len(faculty)
        faculty = faculty[faculty["shib_role"] == "faculty"].copy()
        if len(faculty) != n_before:
            print(f"  dropped {n_before - len(faculty)} non-faculty rows from Faculty.csv")

    print("Adding team size (total author count) per publication...", flush=True)
    pubs["team_size"] = pubs["authors"].apply(count_team_size)

    print("Building faculty-publication long table (this is the slow step)...", flush=True)
    long_df, link_report = build_long_table(pubs, faculty)

    print(f"Loading allowlist from {args.allowlist}...", flush=True)
    allow_depts, excluded_allowlist_rows = load_allowlist(Path(args.allowlist))

    # --- Before allowlist drop (for the quality report) ---
    n_pubs_year_window = len(pubs)
    n_pubs_faculty_matched = int(long_df["pub_id"].nunique())
    n_rows_before = len(long_df)

    long_df["department"] = long_df["department"].astype(str).str.strip()
    long_df["college"] = long_df["college"].astype(str).str.strip()

    excluded_mask = ~long_df["department"].isin(allow_depts)
    excluded_rows = long_df.loc[excluded_mask]
    excluded_pub_ids = set(excluded_rows["pub_id"].unique())
    # Pubs that disappear entirely once non-allowlist authorship rows are removed
    pubs_only_non_allowlist = excluded_pub_ids - set(
        long_df.loc[~excluded_mask, "pub_id"].unique()
    )

    excluded_by_college = (
        excluded_rows.groupby("college")
        .agg(n_authorship_rows=("uid", "size"), n_faculty=("uid", "nunique"),
             n_publications=("pub_id", "nunique"))
        .sort_values("n_publications", ascending=False)
    )
    excluded_by_dept = (
        excluded_rows.groupby(["college", "department"])
        .agg(n_authorship_rows=("uid", "size"), n_faculty=("uid", "nunique"),
             n_publications=("pub_id", "nunique"))
        .sort_values("n_publications", ascending=False)
    )

    # Option A: keep only authorship rows whose department is allowlisted
    long_df = long_df.loc[~excluded_mask].copy()
    keep_pubs = set(long_df["pub_id"].unique())
    pubs = pubs[pubs["pub_id"].isin(keep_pubs)].copy()

    n_pubs_after_allowlist = int(long_df["pub_id"].nunique())
    n_rows_after = len(long_df)

    pubs.to_csv(outdir / "publications_clean.csv", index=False)
    long_df.to_csv(outdir / "faculty_authorship_long.csv", index=False)

    n_colleges = long_df["college"].nunique()
    n_depts = long_df["department"].nunique()

    delta = n_pubs_after_allowlist - EXPECTED_ALLOWLIST_PUBS
    allowlist_note = (
        f"publications_after_allowlist ({n_pubs_after_allowlist}) matches "
        f"expected ~{EXPECTED_ALLOWLIST_PUBS}."
        if abs(delta) <= 50
        else (
            f"WARNING: publications_after_allowlist is {n_pubs_after_allowlist}, "
            f"expected ~{EXPECTED_ALLOWLIST_PUBS} (delta={delta:+d}). "
            f"Check allowlist path and department name spelling."
        )
    )

    report_lines = [
        "DATA QUALITY REPORT",
        "====================",
        "",
        "Publications",
        "-------------",
        *(f"{k}: {v}" for k, v in pub_report.items()),
        "",
        "Faculty",
        "-------",
        f"faculty_rows_in_Faculty_csv: {len(faculty)}",
        f"colleges_in_analysis_long_table: {n_colleges}",
        f"departments_in_analysis_long_table: {n_depts}",
        "",
        "Faculty-publication linkage (before allowlist filter)",
        "-----------------------------------------------------",
        *(f"{k}: {v}" for k, v in link_report.items()),
        f"publications_faculty_matched_before_allowlist: {n_pubs_faculty_matched}",
        f"authorship_rows_before_allowlist: {n_rows_before}",
        "",
        "Allowlist filter (Option A)",
        "---------------------------",
        f"allowlist_file: {args.allowlist}",
        f"allowlist_departments_kept: {len(allow_depts)}",
        f"allowlist_rows_excluded_by_nonacademic_college: {len(excluded_allowlist_rows)}",
        f"excluded_allowlist_colleges: {', '.join(sorted(EXCLUDED_ALLOWLIST_COLLEGES))}",
        f"publications_in_year_window: {n_pubs_year_window}",
        f"publications_faculty_matched_before_allowlist: {n_pubs_faculty_matched}",
        f"publications_after_allowlist: {n_pubs_after_allowlist}",
        f"authorship_rows_after_allowlist: {n_rows_after}",
        f"authorship_rows_dropped: {n_rows_before - n_rows_after}",
        f"publications_lost_entirely_to_allowlist: {len(pubs_only_non_allowlist)}",
        allowlist_note,
        "",
        "Excluded authorship activity before drop (by college)",
        "-----------------------------------------------------",
        "(Counts below are from rows whose department is NOT in the kept allowlist.",
        " A publication can appear here and still be kept if it also has an",
        " allowlisted co-author.)",
        excluded_by_college.to_string() if len(excluded_by_college) else "(none)",
        "",
        "Excluded authorship activity before drop (by college, department)",
        "-----------------------------------------------------------------",
        excluded_by_dept.to_string() if len(excluded_by_dept) else "(none)",
        "",
        "Allowlist departments dropped as non-academic (from depts.txt)",
        "----------------------------------------------------------------",
        (
            excluded_allowlist_rows[["department", "college"]]
            .drop_duplicates()
            .sort_values(["college", "department"])
            .to_string(index=False)
            if len(excluded_allowlist_rows)
            else "(none)"
        ),
    ]
    (outdir / "data_quality_report.txt").write_text("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nWrote outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
