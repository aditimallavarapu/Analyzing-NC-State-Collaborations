"""
04_build_ui.py
==============
Builds the single-page collaboration explorer from the per-year graph CSVs.

INPUT (same folder):
  network_data/nodes_YYYY.csv   (from 02_build_graphs.py)
  network_data/edges_YYYY.csv   (from 02_build_graphs.py)
  psi_net.json                  (PSI collaboration data, written by 03_build_html.py;
                                 optional - the PSI filter is empty without it)
  CFEP_2026_PARTNERS - CFEP_Personnel_Oct_2025_By_Cluster.csv
                                (CFEP personnel by cluster; optional - without it no one is a CFEP member)
  ui_template.html              (the page; contains the __DATA__ placeholder)

OUTPUT (same folder):
  ncstate_research_networks_new.html   - open this in any browser

Each 5-year block is the union of its years: a person's publications are
summed, edge weights are summed, and people with no publications in the block
(and any edge touching them) are dropped. Only the 10 colleges below are kept.
Each edge carries up to MAX_PAPERS shared-paper titles (already truncated to
70 characters by 02_build_graphs.py).

Run:
  python 04_build_ui.py
"""

import csv
import json
import os

NET_DIR = "network_data"
TEMPLATE = "ui_template.html"
PSI_JSON = "psi_net.json"
CFEP_CSV = "CFEP_2026_PARTNERS - CFEP_Personnel_Oct_2025_By_Cluster.csv"
OUTPUT = "ncstate_research_networks_new.html"

# key = block start year; the page's period dropdown is built from these
BLOCKS = {2006: range(2006, 2011), 2011: range(2011, 2016),
          2016: range(2016, 2021), 2021: range(2021, 2026)}

# Order matters: the position of a college here is the index the page uses to
# pick its colour, so keep it in sync with COL in ui_template.html.
COLLEGES = [
    ("College of Engineering", "Engineering"),
    ("College of Sciences", "Sciences"),
    ("College of Agriculture and Life Sciences", "Agriculture & Life Sci."),
    ("College of Humanities and Social Sciences", "Humanities & Social Sci."),
    ("College of Veterinary Medicine", "Veterinary Medicine"),
    ("College of Education", "Education"),
    ("Poole College of Management", "Management (Poole)"),
    ("College of Natural Resources", "Natural Resources"),
    ("Wilson College of Textiles", "Textiles"),
    ("College of Design", "Design"),
]
COLLEGE_INDEX = {full: i for i, (full, _) in enumerate(COLLEGES)}
MAX_PAPERS = 3


def load_cfep(path):
    """CFEP personnel sheet -> (list of cluster names, {unity id: cluster index}).
    Rows without a Unity ID (open searches) are skipped."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found - no one will be marked as a CFEP member")
        return [], {}
    clusters, members = [], {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            uid, cluster = (r.get("Unity ID") or "").strip().lower(), " ".join((r.get("Cluster") or "").split())
            if not uid or not cluster:
                continue
            if cluster not in clusters:
                clusters.append(cluster)
            members[uid] = clusters.index(cluster)
    return clusters, members


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_block(years, cfep):
    people, pairs = {}, {}
    for yr in years:
        for r in read(os.path.join(NET_DIR, f"nodes_{yr}.csv")):
            if r["college"] not in COLLEGE_INDEX:
                continue
            p = people.setdefault(r["id"], {"nm": r["name"], "c": COLLEGE_INDEX[r["college"]],
                                            "d": r["dept"], "p": 0, "f": 0})
            p["p"] += int(r["pub_year"] or 0)
            p["f"] = p["f"] or int(r["is_cfep"] == "True")
        for r in read(os.path.join(NET_DIR, f"edges_{yr}.csv")):
            key = tuple(sorted((r["source"], r["target"])))
            e = pairs.setdefault(key, [0, []])
            e[0] += int(r["weight"] or 1)
            for t in r["papers"].split("|"):
                if t.strip() and t not in e[1] and len(e[1]) < MAX_PAPERS:
                    e[1].append(t)
    keep = {i for i, p in people.items() if p["p"] > 0}
    # node row: [id, name, college, dept, publications, is_cfep, cfep_cluster_index (-1 if none)]
    nodes = [[i, p["nm"], p["c"], p["d"], p["p"], int(i.lower() in cfep), cfep.get(i.lower(), -1)]
             for i, p in people.items() if i in keep]
    edges = [[a, b, w, papers] for (a, b), (w, papers) in pairs.items() if a in keep and b in keep]
    return {"n": nodes, "e": edges}


def build_psi_block(psi_net, years, block):
    """PSI-tagged people and co-authorships for one block, limited to people who
    are also nodes of that block's main graph (so they have a college)."""
    keep = {n[0] for n in block["n"]}
    people, pairs = {}, {}
    for yr in years:
        y = psi_net.get(str(yr))
        if not y:
            continue
        for n in y["n"]:
            if n["id"] in keep:
                people[n["id"]] = people.get(n["id"], 0) + int(n["py"])
        for e in y["e"]:
            if e["s"] in keep and e["t"] in keep:
                key = tuple(sorted((e["s"], e["t"])))
                cur = pairs.setdefault(key, [0, []])
                cur[0] += int(e["w"])
                for t in e.get("p", []):
                    if t.strip() and t not in cur[1] and len(cur[1]) < MAX_PAPERS:
                        cur[1].append(t)
    return {"n": people, "e": [[a, b, w, papers] for (a, b), (w, papers) in pairs.items()]}


def main():
    cfep_clusters, cfep = load_cfep(CFEP_CSV)
    blocks = {}
    for start, years in BLOCKS.items():
        missing = [y for y in years if not os.path.exists(os.path.join(NET_DIR, f"nodes_{y}.csv"))]
        if missing:
            raise SystemExit(f"Missing {NET_DIR}/nodes_YYYY.csv for {missing} - run 02_build_graphs.py first")
        blocks[str(start)] = build_block(years, cfep)
        b = blocks[str(start)]
        print(f"  {start}-{years[-1]}: {len(b['n']):,} faculty, {len(b['e']):,} collaborations, "
              f"{sum(n[5] for n in b['n'])} CFEP members")

    psi_net = {}
    if os.path.exists(PSI_JSON):
        with open(PSI_JSON, encoding="utf-8") as f:
            psi_net = json.load(f)
    else:
        print(f"  WARNING: {PSI_JSON} not found - the PSI filter will be empty")
    psi = {str(start): build_psi_block(psi_net, years, blocks[str(start)]) for start, years in BLOCKS.items()}
    for start in psi:
        print(f"  PSI {start}: {len(psi[start]['n']):,} faculty, {len(psi[start]['e']):,} collaborations")

    data = {"colleges": [short for _, short in COLLEGES], "cfep": cfep_clusters, "blocks": blocks, "psi": psi}
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    if "__DATA__" not in html:
        raise SystemExit(f"{TEMPLATE} has no __DATA__ placeholder")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"Wrote {OUTPUT} ({os.path.getsize(OUTPUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
