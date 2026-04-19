# Analyzing-NC-State-Collaborations
This project looks at past 20 years of publication data and analyzes collaborations across colleges and departments.


1) The publications.csv is avaialble on Aditi Local, can be requested. 
2) Clean_csv - focuses on cleaning the file and calculates faculty level publication counts.
3) The html file is built out of the cleaned publication and faculty datasets.
4) Visualizations tool's functionality:

1. Interactive Collaboration Graph
Force-Directed Network: Visualize relationships between hundreds of faculty members can be filtered by year.

Dynamic Scaling: Node sizes represent the volume of publications within the selected year, while edges denote shared co-authorship.

Symbology: 
Diamonds: Identify members of the CFEP cluster.
Stars: Highlight Top-10 Assistant Professors to showcase emerging research leaders.

2. Multi-Level Filtering & Navigation
Temporal Analysis: A year-selection dropdown allowing users to scrub through data from 2006 to 2025.
College-Specific Views: A sidebar legend that breaks down faculty counts by college (e.g., Engineering, Agriculture & Life Sciences, Veterinary Medicine) and likely acts as a filter.
Search Functionality: A real-time "Search by name" feature to quickly locate specific researchers within the network.
Focus Mode: A "Show CFEP only" toggle to isolate and analyze specific research clusters.

3. Data Insights & Portability
Live Metrics: High-level counters for total faculty, collaborations, and papers (e.g., 4,236 papers in 2023) that update based on filters.
Export Capabilities: A "Download Network CSV" feature, enabling users to take the raw collaboration data into other analytical tools.
Comparative Impact Views: Toggle buttons for "Collaboration Network" and "CFEP Cluster Impact" to switch between relational and performance-based visualizations.

4. Technical UI/UX Highlights
Visual Legend: Integrated color-coding for different university colleges to make interdisciplinary links immediately apparent.
Responsive Design: Clean, overlay-based UI that maximizes the graph area while keeping controls accessible.

5. CFEP Cluster Impact Analysis
The CFEP Cluster Impact view provides a comparative analysis of the Chancellor’s Faculty Excellence Program (CFEP) cluster’s growth and productivity over a 20-year period (2006–2025). This view is designed to track how specific interdisciplinary cohorts evolve from their baseline to their current state.

Key Features
Dual-Pane Temporal Comparison: A side-by-side visualization showing the cluster's network density and membership in 2006 vs. 2025.

Active Membership vs. Publication Presence: 

Bar Chart (Active Members): Tracks the total number of faculty affiliated with the cluster per year.

Network View (Publication Record): Filters to show only those members with active, indexed publications in the selected year, highlighting the gap between "on-campus presence" and "research output."

Growth Metrics Visualization: 
Total Publications (Blue Bars): Displays the aggregate research output for the cluster.

Membership Trends (Pink Line): Overlays the number of active faculty members to show the correlation between cluster size and research volume.

Cluster-Specific Filtering: Users can toggle between All Faculty and CFEP Members Only to isolate the cluster's internal collaborative structure from the broader university network.

Interactive History: A 20-year timeline that allows users to identify specific years of rapid growth or transitional shifts in cluster activity.

Visual Cues
Node Clusters: In the 2025 view, nodes are grouped and color-coded by college, visualizing how the CFEP cluster bridges multiple academic departments (e.g., Engineering, Sciences, and Agriculture).

Empty Baseline: The 2006 view illustrates the "pre-impact" state, often showing active members who had not yet established their collaborative publication record within the system.

Use Case
This view is particularly useful for university administrators and researchers looking to evaluate the long-term ROI of interdisciplinary hiring initiatives by seeing exactly when a cluster reaches "critical mass" in research output.
