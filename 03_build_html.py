"""
03_build_html.py
================
Reads per-year CSVs from network_data/ plus leaderboard.json and
cfep_timelines.json, then injects them into network_template.html
and writes ncstate_research_networks.html.

INPUT (same folder):
  network_data/nodes_YYYY.csv   (from 02_build_graphs.py)
  network_data/edges_YYYY.csv   (from 02_build_graphs.py)
  leaderboard.json              (from 02_build_graphs.py)
  cfep_timelines.json           (from 02_build_graphs.py)
  network_template.html         (the visual template)

OUTPUT (same folder):
  ncstate_research_networks.html   — open this in any browser

Run:
  python 03_build_html.py
"""

import json
import os
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────
YEARS     = list(range(2006, 2026))
NET_DIR   = "network_data"
TEMPLATE  = "network_template.html"
OUTPUT    = "ncstate_research_networks.html"

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

# ── Load per-year graph data from CSVs ────────────────────────────────
print("Reading per-year CSVs...")
graphs = {}

for yr in YEARS:
    nf = f"{NET_DIR}/nodes_{yr}.csv"
    ef = f"{NET_DIR}/edges_{yr}.csv"

    if not os.path.exists(nf):
        print(f"  WARNING: missing {nf} — skipping year {yr}")
        continue

    nodes_df = pd.read_csv(nf, low_memory=False)
    edges_df = pd.read_csv(ef, low_memory=False) if os.path.exists(ef) else pd.DataFrame()

    # Convert node rows to slim dicts
    nodes = []
    for _, row in nodes_df.iterrows():
        col = str(row.get("college", ""))
        nodes.append({
            "id":  str(row["id"]),
            "nm":  str(row["name"]),
            "col": col,
            "dept": str(row.get("dept", "") or ""),
            "ttl": str(row.get("title", "") or ""),
            "ia":  bool(str(row.get("is_asst", "False")).lower() in ("true", "1")),
            "py":  int(row.get("pub_year", 0) or 0),
            "pt":  int(row.get("pub_total", 0) or 0),
            "clr": COLLEGE_COLORS.get(col, "#888888"),
            "cf":  bool(str(row.get("is_cfep", "False")).lower() in ("true", "1")),
            "cc":  str(row.get("cfep_cluster", "") or ""),
        })

    # Convert edge rows to slim dicts
    edges = []
    for _, row in edges_df.iterrows():
        papers_raw = str(row.get("papers", "") or "")
        papers = [p for p in papers_raw.split("|") if p.strip()]
        edges.append({
            "s":  str(row["source"]),
            "t":  str(row["target"]),
            "w":  int(row.get("weight", 1) or 1),
            "xc": bool(str(row.get("cross_college", "False")).lower() in ("true", "1")),
            "p":  papers[:3],
        })

    # Total papers this year = edges imply collaborative papers, but
    # we derive total from node count (each node = 1+ pub this year)
    total_papers = int(nodes_df["pub_year"].sum()) if not nodes_df.empty else 0

    graphs[yr] = {"n": nodes, "e": edges, "tp": total_papers}
    print(f"  {yr}: {len(nodes)} nodes, {len(edges)} edges")

# ── Load leaderboard JSON ─────────────────────────────────────────────
lb_path = "leaderboard.json"
if os.path.exists(lb_path):
    with open(lb_path, encoding="utf-8") as f:
        lb_raw = json.load(f)
    # Keys in file are strings; keep as strings to match JS year access
    leaderboard = lb_raw
    print(f"Loaded leaderboard.json ({len(leaderboard)} years)")
else:
    leaderboard = {}
    print("WARNING: leaderboard.json not found")

# ── Load CFEP timelines JSON ──────────────────────────────────────────
cfep_path = "cfep_timelines.json"
if os.path.exists(cfep_path):
    with open(cfep_path, encoding="utf-8") as f:
        cfep_data = json.load(f)
    print(f"Loaded cfep_timelines.json ({len(cfep_data.get('members', []))} members)")
else:
    cfep_data = {"members": [], "cluster_year": {}}
    print("WARNING: cfep_timelines.json not found")

# ── Serialize to JSON strings ─────────────────────────────────────────
graphs_js = json.dumps(graphs,     separators=(",", ":"))
lb_js     = json.dumps(leaderboard, separators=(",", ":"))
cfep_js   = json.dumps(cfep_data,   separators=(",", ":"))

print(f"\nData sizes: graphs={len(graphs_js)//1024}KB  lb={len(lb_js)//1024}KB  cfep={len(cfep_js)//1024}KB")

# ── Load template and inject ──────────────────────────────────────────
if not os.path.exists(TEMPLATE):
    raise FileNotFoundError(f"Template not found: {TEMPLATE}")

with open(TEMPLATE, encoding="utf-8") as f:
    html = f.read()

html = html.replace("__GRAPHS__", graphs_js)
html = html.replace("__LB__",     lb_js)
html = html.replace("__CFEP__",   cfep_js)

# ── Verify placeholders were replaced ────────────────────────────────
for placeholder in ("__GRAPHS__", "__LB__", "__CFEP__"):
    if placeholder in html:
        print(f"  ERROR: placeholder {placeholder} was not replaced!")

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(OUTPUT) // 1024
print(f"\n✓  Saved {OUTPUT}  ({size_kb} KB)")
print(f"   Open in browser: http://localhost:8000/{OUTPUT}")
