"""Configuration for the LLM-judge topic validation pipeline.

Paths are anchored to this file's location (not the process cwd) so the
script behaves the same whether it's launched as `python judge_validate.py`
from inside 06_topic_modeling/ (the pipeline convention) or invoked with a
full path from elsewhere.
"""

from pathlib import Path

from dotenv import load_dotenv

_THIS_DIR = Path(__file__).resolve().parent

# Loads GEMINI_API_KEY (and anything else) from 06_topic_modeling/.env into
# the environment, if that file exists. Real keys go in .env (gitignored,
# never committed) — .env.example shows the expected format.
load_dotenv(_THIS_DIR / ".env")

# Paths
TOPIC_MODELING_DIR = str(_THIS_DIR)
TOPICS_DIR = str(_THIS_DIR / "topics")

# Confirmed against the repo root: PSI_NC_affiliation.csv is the only one of
# PSI_NC_affiliation.csv / PSI_updated.csv / PSI.csv with zero missing
# abstracts (PSI_updated.csv has 928 missing abstracts out of 3221 rows).
PUB_DATA_PATH = str(_THIS_DIR.parent / "PSI_NC_affiliation.csv")

# Confirmed column names (from the actual CSVs, 2026-09-19)
# PSI_NC_affiliation.csv: Unnamed: 0,title,authors,nc_state_people,college,
#   department,DOI,PMID,year,url,topics,abstract
# -> "Unnamed: 0" is renamed to "pub_id" on load.
PUB_ID_COL = "pub_id"
TITLE_COL = "title"
ABSTRACT_COL = "abstract"

# topics_<model>_<nr_topics>.csv: topic_id,size,topic_label,subtopics
TOPIC_ID_COL = "topic_id"
TOPIC_LABEL_COL = "topic_label"
TOPIC_SIZE_COL = "size"
SUBTOPICS_COL = "subtopics"

# doc_topics_<model>_<nr_topics>.csv: pub_id,topic_id,topic_label,subtopic_id,subtopic
DOC_TOPIC_PUB_ID_COL = "pub_id"
DOC_TOPIC_TOPIC_ID_COL = "topic_id"

# Gemini settings
GEMINI_MODEL = "gemini-3.5-flash-lite"  # gemini-1.5-flash-8b was retired by Google; this is its successor
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"  # read from environment variable, never hardcode

# Sampling settings
SAMPLE_PCT = 0.20
SAMPLE_CAP = 30
SAMPLE_FLOOR = 5
RANDOM_SEED = 42

# Sleep between API calls (seconds) — stay under 15 RPM
SLEEP_BETWEEN_CALLS = 5
