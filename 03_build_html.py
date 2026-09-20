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
import re
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────
YEARS     = list(range(2006, 2026))
NET_DIR   = "network_data"
TEMPLATE  = "network_template.html"
OUTPUT    = "ncstate_research_networks.html"
PSI_PUB_CSV = "PSI_NC_affiliation.csv"

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
    # NOTE: not red — red is reserved for cross-college edge highlighting
    "College of Design":                         "#34495E",
}

# ── PSI (Plant Sciences Initiative) collaboration data ──────────────────
# Parsed straight from PSI_NC_affiliation.csv's "nc_state_people" column
# (NC State co-authors only). College/department are deliberately NOT
# resolved here — most of these 1,441 co-authors aren't in any faculty
# roster (students, postdocs, staff), so college is only meaningful for
# whoever is also a real node in the main network. That filtering happens
# client-side against GRAPHS, where a person's college is already known.
def _parse_people(val):
    if not isinstance(val, str) or not val.strip():
        return []
    out = []
    for part in val.split(";"):
        part = part.strip()
        m = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", part)
        if m:
            out.append((m.group(1).strip(), m.group(2).strip().lower()))
    return out

PSI_NET = {}
if os.path.exists(PSI_PUB_CSV):
    psi_pub_df = pd.read_csv(PSI_PUB_CSV, low_memory=False)
    psi_nodes_by_year = {}   # year -> {uid: {"nm":..., "py":count}}
    psi_edges_by_year = {}   # year -> {(a,b): {"w":count, "p":[papers]}}

    for _, row in psi_pub_df.iterrows():
        try:
            yr = int(row.get("year"))
        except (TypeError, ValueError):
            continue
        people = _parse_people(row.get("nc_state_people"))
        if not people:
            continue
        title = str(row.get("title", "") or "")[:70]
        yn = psi_nodes_by_year.setdefault(yr, {})
        ye = psi_edges_by_year.setdefault(yr, {})
        for nm, uid in people:
            ent = yn.setdefault(uid, {"nm": nm, "py": 0})
            ent["py"] += 1
        uids = [uid for _, uid in people]
        for i in range(len(uids)):
            for j in range(i + 1, len(uids)):
                a, b = min(uids[i], uids[j]), max(uids[i], uids[j])
                e = ye.setdefault((a, b), {"w": 0, "p": []})
                e["w"] += 1
                if title and len(e["p"]) < 3 and title not in e["p"]:
                    e["p"].append(title)

    for yr in sorted(psi_nodes_by_year):
        nodes = [{"id": uid, "nm": info["nm"], "py": info["py"]}
                 for uid, info in psi_nodes_by_year[yr].items()]
        edges = [{"s": a, "t": b, "w": info["w"], "p": info["p"]}
                 for (a, b), info in psi_edges_by_year.get(yr, {}).items()]
        PSI_NET[yr] = {"n": nodes, "e": edges}
    print(f"Loaded PSI collaboration data: {sum(len(v['n']) for v in PSI_NET.values())} "
          f"person-year rows across {len(PSI_NET)} years from {PSI_PUB_CSV}")
else:
    print(f"  WARNING: {PSI_PUB_CSV} not found — PSI Focus tab will be empty")

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
psi_js    = json.dumps(PSI_NET,    separators=(",", ":"))

print(f"\nData sizes: graphs={len(graphs_js)//1024}KB  lb={len(lb_js)//1024}KB  "
      f"cfep={len(cfep_js)//1024}KB  psi={len(psi_js)//1024}KB")

# ── Load template and inject ──────────────────────────────────────────
if not os.path.exists(TEMPLATE):
    raise FileNotFoundError(f"Template not found: {TEMPLATE}")

with open(TEMPLATE, encoding="utf-8") as f:
    html = f.read()

html = html.replace("__GRAPHS__", graphs_js)
html = html.replace("__LB__",     lb_js)
html = html.replace("__CFEP__",   cfep_js)
html = html.replace("__PSI_NET__", psi_js)

# ── Verify placeholders were replaced ────────────────────────────────
for placeholder in ("__GRAPHS__", "__LB__", "__CFEP__", "__PSI_NET__"):
    if placeholder in html:
        print(f"  ERROR: placeholder {placeholder} was not replaced!")

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(OUTPUT) // 1024
print(f"\n✓  Saved {OUTPUT}  ({size_kb} KB)")
print(f"   Open in browser: http://localhost:8000/{OUTPUT}")
