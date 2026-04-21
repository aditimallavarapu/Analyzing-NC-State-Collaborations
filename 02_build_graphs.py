"""
02_build_graphs.py
==================
Reads cleaned data and produces per-year graph CSVs plus two JSON support files.

INPUT (same folder):
  publications_clean.csv
  faculty_clean.csv
  faculty_pub_counts.csv
  CFEP_2026_PARTNERS.xlsx   (optional — needed for CFEP tab)

OUTPUT (same folder):
  network_data/nodes_YYYY.csv   — one file per year 2006-2025
  network_data/edges_YYYY.csv   — one file per year 2006-2025
  leaderboard.json              — top-10 per college per year
  cfep_timelines.json           — CFEP member publication trends

Run:
  python 02_build_graphs.py
"""

import pandas as pd
import re
import json
import csv
import os
from collections import defaultdict, Counter

# ── Paths ────────────────────────────────────────────────────────────
PUB_CSV    = "publications_clean.csv"
FAC_CSV    = "faculty_clean.csv"
COUNTS_CSV = "faculty_pub_counts.csv"
CFEP_XLSX  = "CFEP_2026_PARTNERS.xlsx"
OUT_DIR    = "network_data"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Valid colleges (no Unknown / Admin / Interdisciplinary) ──────────
VALID_COLLEGES = {
    "College of Agriculture and Life Sciences",
    "College of Design",
    "College of Education",
    "College of Engineering",
    "College of Humanities and Social Sciences",
    "College of Natural Resources",
    "College of Sciences",
    "College of Veterinary Medicine",
    "Poole College of Management",
    "Wilson College of Textiles",
}

COLLEGE_COLORS = {
    "College of Engineering":                   "#0072B2",
    "College of Sciences":                      "#009E73",
    "College of Agriculture and Life Sciences":  "#E69F00",
    "College of Humanities and Social Sciences": "#CC79A7",
    "College of Veterinary Medicine":            "#56B4E9",
    "College of Education":                      "#D55E00",
    "Poole College of Management":               "#F0E442",
    "College of Natural Resources":              "#9B59B6",
    "Wilson College of Textiles":                "#1ABC9C",
    "College of Design":                         "#E74C3C",
}

# ── Load data ─────────────────────────────────────────────────────────
print("Loading data...")
pub    = pd.read_csv(PUB_CSV, low_memory=False)
fac    = pd.read_csv(FAC_CSV, low_memory=False)
counts = pd.read_csv(COUNTS_CSV, low_memory=False)

fac_by_uid    = fac.set_index("uid").to_dict("index")
counts_by_uid = dict(zip(counts["uid"], counts["total_pubs"]))
valid_uids    = set(fac["uid"])

# ── CFEP data ─────────────────────────────────────────────────────────
cfep_uids        = set()
cfep_cluster_map = {}
cfep_rank_map    = {}
cfep_members_list = []

if os.path.exists(CFEP_XLSX):
    print("Loading CFEP data...")
    cfep_df = pd.read_excel(CFEP_XLSX, sheet_name="CFEP_Personnel_April_2026_By_Cl")
    for _, row in cfep_df.iterrows():
        uid = str(row.get("Unity ID", "")).strip().lower()
        if not uid or uid == "nan":
            continue
        cluster = str(row.get("Cluster", "")).strip()
        rank    = str(row.get("Starting Rank", "")).strip()
        name_raw = str(row.get("Faculty Name", "")).strip()
        # Convert "Last, First" → "First Last"
        if "," in name_raw:
            parts = name_raw.split(",", 1)
            display_name = parts[1].strip() + " " + parts[0].strip()
        else:
            display_name = name_raw
        cfep_uids.add(uid)
        cfep_cluster_map[uid] = cluster
        cfep_rank_map[uid]    = rank
        cfep_members_list.append({"uid": uid, "name": display_name,
                                   "cluster": cluster, "starting_rank": rank})
    print(f"  {len(cfep_uids)} CFEP members loaded")
else:
    print(f"  WARNING: {CFEP_XLSX} not found — CFEP features will be empty")

# ── Parser ────────────────────────────────────────────────────────────
def parse_people(val):
    """'Name (uid); ...' → list of (name, uid)"""
    if not val or pd.isna(val):
        return []
    result = []
    for part in str(val).split(";"):
        part = part.strip()
        m = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", part)
        if m:
            result.append((m.group(1).strip(), m.group(2).strip()))
    return result

# ── Build per-year graphs ─────────────────────────────────────────────
YEARS = list(range(2006, 2026))
leaderboard = {}

print(f"\nBuilding graphs for {len(YEARS)} years...")
for yr in YEARS:
    df_yr = pub[pub["year"] == yr]

    # Count publications per faculty member this year
    person_pubs = defaultdict(int)
    for _, row in df_yr.iterrows():
        for name, uid in parse_people(row["nc_state_people"]):
            if uid in valid_uids:
                person_pubs[uid] += 1

    # Active = faculty with ≥1 pub this year AND valid college
    active = {
        uid for uid, cnt in person_pubs.items()
        if fac_by_uid.get(uid, {}).get("college", "") in VALID_COLLEGES
    }

    # Build edges between NC State co-authors
    edge_counter = Counter()
    edge_papers  = defaultdict(list)
    edge_cross   = {}

    for _, row in df_yr.iterrows():
        ppl  = parse_people(row["nc_state_people"])
        uids = [uid for _, uid in ppl if uid in active]
        if len(uids) < 2:
            continue
        for i in range(len(uids)):
            for j in range(i + 1, len(uids)):
                a, b = min(uids[i], uids[j]), max(uids[i], uids[j])
                key  = (a, b)
                edge_counter[key] += 1
                if len(edge_papers[key]) < 3:
                    edge_papers[key].append(str(row["title"])[:70])
                ca = fac_by_uid.get(uids[i], {}).get("college", "")
                cb = fac_by_uid.get(uids[j], {}).get("college", "")
                edge_cross[key] = (ca != cb and
                                   ca in VALID_COLLEGES and
                                   cb in VALID_COLLEGES)

    # ── Write nodes CSV ───────────────────────────────────────────────
    node_rows = []
    for uid in active:
        fi = fac_by_uid.get(uid, {})
        college = fi.get("college", "")
        node_rows.append({
            "id":           uid,
            "name":         fi.get("name", uid),
            "college":      college,
            "dept":         fi.get("department", ""),
            "title":        fi.get("shib_title", ""),
            "is_asst":      bool(fi.get("is_asst_prof", False)),
            "pub_year":     person_pubs[uid],
            "pub_total":    counts_by_uid.get(uid, person_pubs[uid]),
            "color":        COLLEGE_COLORS.get(college, "#888888"),
            "is_cfep":      uid in cfep_uids,
            "cfep_cluster": cfep_cluster_map.get(uid, ""),
        })

    nodes_df = pd.DataFrame(node_rows)
    nodes_df.to_csv(f"{OUT_DIR}/nodes_{yr}.csv", index=False, quoting=csv.QUOTE_ALL)

    # ── Write edges CSV ───────────────────────────────────────────────
    edge_rows = []
    for (a, b), cnt in edge_counter.items():
        if a in active and b in active:
            edge_rows.append({
                "source":        a,
                "target":        b,
                "weight":        cnt,
                "cross_college": edge_cross.get((a, b), False),
                "papers":        "|".join(edge_papers[(a, b)]),
            })

    edges_df = pd.DataFrame(edge_rows)
    edges_df.to_csv(f"{OUT_DIR}/edges_{yr}.csv", index=False, quoting=csv.QUOTE_ALL)

    # ── Leaderboard: top-10 per college ──────────────────────────────
    lb_yr = {}
    for college in VALID_COLLEGES:
        col_nodes = sorted(
            [n for n in node_rows if n["college"] == college],
            key=lambda x: -x["pub_year"]
        )[:10]
        lb_yr[college] = [
            {"uid": n["id"], "name": n["name"],
             "pub_year": n["pub_year"], "is_asst": n["is_asst"],
             "is_cfep": n["is_cfep"]}
            for n in col_nodes
        ]
    leaderboard[yr] = lb_yr

    isolated = sum(1 for uid in active
                   if uid not in {e["source"] for e in edge_rows}
                   and uid not in {e["target"] for e in edge_rows})
    cross    = sum(1 for e in edge_rows if e["cross_college"])
    cfep_n   = sum(1 for uid in active if uid in cfep_uids)
    print(f"  {yr}: {len(active):>4} nodes ({isolated:>4} isolated) | "
          f"{len(edge_rows):>5} edges ({cross:>3} cross-college) | {cfep_n} CFEP")

# ── Save leaderboard JSON ─────────────────────────────────────────────
with open("leaderboard.json", "w", encoding="utf-8") as f:
    json.dump(leaderboard, f, separators=(",", ":"))
print(f"\nSaved leaderboard.json")

# ── Build CFEP timelines ──────────────────────────────────────────────
print("Building CFEP timelines...")
# Count pubs per uid per year from full publication record
uid_year_counts = defaultdict(lambda: defaultdict(int))
for _, row in pub.iterrows():
    yr = int(row["year"])
    for name, uid in parse_people(row["nc_state_people"]):
        uid_year_counts[uid][yr] += 1

members_timeline = []
cluster_year     = defaultdict(lambda: defaultdict(int))

for m in cfep_members_list:
    uid  = m["uid"]
    pby  = {yr: uid_year_counts[uid].get(yr, 0) for yr in YEARS}
    total = sum(pby.values())
    members_timeline.append({
        "uid":          uid,
        "name":         m["name"],
        "cluster":      m["cluster"],
        "starting_rank": m["starting_rank"],
        "pubs_by_year": pby,
        "total_pubs":   total,
        "in_2006":      pby.get(2006, 0) > 0,
    })
    for yr, cnt in pby.items():
        cluster_year[m["cluster"]][yr] += cnt

cfep_data = {
    "members":      members_timeline,
    "cluster_year": {k: dict(v) for k, v in cluster_year.items()},
}
with open("cfep_timelines.json", "w", encoding="utf-8") as f:
    json.dump(cfep_data, f, separators=(",", ":"))
print(f"Saved cfep_timelines.json ({len(members_timeline)} members)")

print("\n✓  All done. Run 03_build_html.py next.")
