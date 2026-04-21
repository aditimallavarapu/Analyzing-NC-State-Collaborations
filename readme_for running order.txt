bash# Step 1 — clean the raw data (run once per data update)
python 01_clean_data.py
# produces: publications_clean.csv, faculty_clean.csv, 
#           faculty_pub_counts.csv, top10_collaborators.csv

# Step 2 — build per-year graph CSVs and support JSONs (run once per data update)
python 02_build_graphs.py
# produces: network_data/nodes_YYYY.csv, network_data/edges_YYYY.csv
#           leaderboard.json, cfep_timelines.json

# Step 3 — build the final HTML (run whenever you want to refresh the viz)
python 03_build_html.py
# produces: ncstate_research_networks.html

# Step 4 — serve and open
python -m http.server 8000
# Open: http://localhost:8000/ncstate_research_networks.html
Why the server is needed: