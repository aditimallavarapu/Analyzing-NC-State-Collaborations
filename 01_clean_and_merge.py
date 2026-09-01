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
    python 01_clean_and_merge.py --pubs exported_works_faculty-2005-present-2_20260731.csv --faculty Faculty.csv --outdir clean/
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

    pubs.to_csv(outdir / "publications_clean.csv", index=False)
    long_df.to_csv(outdir / "faculty_authorship_long.csv", index=False)

    n_colleges = faculty["college"].nunique()
    n_depts = faculty["department"].nunique()

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
        f"faculty_rows: {len(faculty)}",
        f"colleges_in_faculty_file: {n_colleges}",
        f"departments_in_faculty_file: {n_depts}",
        "",
        "NOTE: Faculty.csv currently lists more college categories than the",
        "10 academic colleges used in your existing Table 2 / Figure 2 (e.g.",
        "'NC State Administration and Offices', 'University College',",
        "'Interdisciplinary Programs', 'Graduate School' also appear as",
        "college values). Decide and document whether these 4 non-academic",
        "units are included in your '14 colleges' count or excluded, and",
        "apply that decision consistently everywhere a college count is",
        "reported (intro, methods, and every table/figure).",
        "",
        "Faculty-publication linkage",
        "----------------------------",
        *(f"{k}: {v}" for k, v in link_report.items()),
    ]
    (outdir / "data_quality_report.txt").write_text("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nWrote outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
