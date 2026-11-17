# Step 1 — clean the raw data (run once per data update)
python 01_clean_and_merge.py --pubs publications.csv --faculty Faculty.csv --outdir .
# produces: publications_clean.csv, faculty_authorship_long.csv, data_quality_report.txt

# Step 2 — build per-year graph CSVs (run once per data update)
python 02_build_graphs.py
# produces: network_data/nodes_YYYY.csv, network_data/edges_YYYY.csv
#           (also leaderboard.json and cfep_timelines.json, which only the old pages in old_versions/ use)

# Step 3 — build the page (run whenever ui_template.html or the data changes)
python 04_build_ui.py
# reads:    network_data/, psi_net.json, CFEP_2026_PARTNERS - CFEP_Personnel_Oct_2025_By_Cluster.csv, ui_template.html
# produces: index.html (GitHub Pages serves it as the site's front page)
# Edit ui_template.html, never index.html.

# Step 3b — build the year-by-year animation (same inputs as step 3, but keeps each year separate)
python 05_build_animation.py
# reads:    network_data/, psi_net.json, the CFEP CSV, animation_template.html
# produces: animation.html
# Edit animation_template.html, never animation.html.

# Step 4 — open
# The page has its data built in, so you can open index.html directly in a browser.

# old_versions/ holds superseded scripts, templates and pages (the previous index.html,
# 01_clean_data.py, 03_build_html.py, ...). 03_build_html.py is also what regenerated
# psi_net.json from PSI_NC_affiliation.csv, so you would need it again for new PSI data.
