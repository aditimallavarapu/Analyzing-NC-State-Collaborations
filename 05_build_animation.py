"""
05_build_animation.py
=====================
Builds the year-by-year animation page from the per-year graph CSVs.

INPUT (same folder):
  network_data/nodes_YYYY.csv   (from 02_build_graphs.py)
  network_data/edges_YYYY.csv   (from 02_build_graphs.py)
  CFEP_2026_PARTNERS - CFEP_Personnel_Oct_2025_By_Cluster.csv
                                (CFEP personnel by cluster; optional - without it no one is a CFEP member)
  animation_template.html       (the page; contains the __DATA__ placeholder)

OUTPUT (same folder):
  animation.html   - open this in any browser

Unlike 04_build_ui.py, years are kept separate: each frame of the animation is
one year's faculty and co-authorships. People are stored once and referred to
by index, and shared-paper titles (already truncated to 70 characters by
02_build_graphs.py) are stored once and referred to by index, which keeps the
page small. The college list, college order and CFEP loader come from
04_build_ui.py so both pages agree.

Run:
  python 05_build_animation.py
"""

import importlib
import json
import os

ui = importlib.import_module("04_build_ui")   # module name starts with a digit, so a plain import can't load it

TEMPLATE = "animation_template.html"
OUTPUT = "animation.html"
YEARS = range(2006, 2026)


def main():
    cfep_clusters, cfep = ui.load_cfep(ui.CFEP_CSV)
    people, pidx, titles, tidx = [], {}, [], {}

    def person(i, r):
        # later years overwrite name / college / department, so each person ends up with their most recent details
        row = [i, r["name"], ui.COLLEGE_INDEX[r["college"]], (r["dept"] or "").strip(),
               int(i.lower() in cfep), cfep.get(i.lower(), -1)]
        if i not in pidx:
            pidx[i] = len(people)
            people.append(row)
        else:
            people[pidx[i]] = row
        return pidx[i]

    def title(t):
        if t not in tidx:
            tidx[t] = len(titles)
            titles.append(t)
        return tidx[t]

    years = {}
    for yr in YEARS:
        npath = os.path.join(ui.NET_DIR, f"nodes_{yr}.csv")
        if not os.path.exists(npath):
            raise SystemExit(f"Missing {npath} - run 02_build_graphs.py first")
        here = {}
        for r in ui.read(npath):
            if r["college"] in ui.COLLEGE_INDEX and int(r["pub_year"] or 0) > 0:
                here[r["id"]] = (person(r["id"], r), int(r["pub_year"]))
        edges = []
        for r in ui.read(os.path.join(ui.NET_DIR, f"edges_{yr}.csv")):
            if r["source"] in here and r["target"] in here:
                papers = [title(t) for t in r["papers"].split("|") if t.strip()][:ui.MAX_PAPERS]
                edges.append([here[r["source"]][0], here[r["target"]][0], int(r["weight"] or 1), papers])
        # node row: [person index, publications that year]; edge row: [person index, person index, shared papers, [title indexes]]
        years[str(yr)] = {"n": list(here.values()), "e": edges}
        print(f"  {yr}: {len(here):,} faculty, {len(edges):,} collaborations")

    data = {"colleges": [short for _, short in ui.COLLEGES], "cfep": cfep_clusters, "people": people,
            "titles": titles, "years": years}
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    if "__DATA__" not in html:
        raise SystemExit(f"{TEMPLATE} has no __DATA__ placeholder")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"  {len(people):,} people, {len(titles):,} paper titles")
    print(f"Wrote {OUTPUT} ({os.path.getsize(OUTPUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
