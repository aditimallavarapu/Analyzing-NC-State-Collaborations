#!/bin/bash
# run_topic_model.sbatch — CPU-only job (UMAP/HDBSCAN don't need a GPU
# partition). Already configured for this project's Hazel setup
# (partition, conda env, data paths). Submit from the 06_topic_modeling/
# directory, with -J so each run's logs are distinguishable:
#   sbatch -J topic_scibert_15   --export=MODEL=scibert,NR_TOPICS=15   sbatch/run_topic_model.sbatch
#   sbatch -J topic_scibert_auto --export=MODEL=scibert,NR_TOPICS=auto sbatch/run_topic_model.sbatch
#SBATCH --job-name=topic_model
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
# (logs/ must already exist at submit time -- SLURM opens the log files
# before this script starts; see README)

# --- environment: edit for your cluster ---
module purge
# if `conda` isn't found here, uncomment and load whatever module puts it
# on PATH on a compute node (it may already be on PATH via your shell rc,
# as it was on the login node when this env was created):
# module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate topicmodel

# --- data location ---
# Code/sbatch/logs live in this checkout; all input/output DATA (PSI_NC_affiliation.csv,
# embeddings/, topics/) lives under IDR_DATA_DIR instead. run_topic_model.py
# reads this env var directly (and already defaults to the same path if
# unset) -- this export just makes it explicit and overridable via
# --export=IDR_DATA_DIR=... per submission.
export IDR_DATA_DIR="${IDR_DATA_DIR:-/rs1/researchers/a/amallav/IDR}"

# --- run config (pass with --export) ---
MODEL="${MODEL:-specter2}"              # specter2, scincl or scibert
NR_TOPICS="${NR_TOPICS:-15}"            # number of topics, or "auto"
MAX_TOPICS="${MAX_TOPICS:-15}"          # hard limit on topics when NR_TOPICS=auto
MIN_TOPIC_SIZE="${MIN_TOPIC_SIZE:-12}"  # smallest subtopic (HDBSCAN min_cluster_size)

# SLURM copies this script to a spool dir before running it, so $0/dirname
# tricks resolve to the wrong place -- SLURM_SUBMIT_DIR is the directory
# `sbatch` was actually invoked from (06_topic_modeling/, per the README's
# usage examples) and is reliable regardless of spooling.
cd "$SLURM_SUBMIT_DIR"

python run_topic_model.py \
    --model "$MODEL" \
    --nr-topics "$NR_TOPICS" \
    --max-topics "$MAX_TOPICS" \
    --min-topic-size "$MIN_TOPIC_SIZE"
