#!/bin/bash
# extract_embeddings.sbatch — single GPU job (no array; ~3,221 docs finishes
# comfortably within a normal time limit). Already configured for this
# project's Hazel setup (partition, conda env, CUDA module, data paths) —
# submit from the 06_topic_modeling/ directory, with -J so each model's
# logs are distinguishable (logs/<jobname>_<jobid>.out/.err):
#   sbatch -J emb_scibert --export=MODEL=scibert sbatch/extract_embeddings.sbatch
#SBATCH --job-name=extract_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1                  # Hazel requires gpu:<type>:<count>; a30
                                           # had the most idle capacity when checked
                                           # (gpu05/07/08) — change via `sinfo -p gpu
                                           # -o "%N|%40G|%T"` if that's no longer true
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
# (logs/ must already exist at submit time -- SLURM opens the log files
# before this script starts; see README)

# --- run config ---
# MODEL is passed in via --export: specter2, scincl, scibert, gtr_t5 or mxbai
MODEL="${MODEL:-specter2}"

# --- environment: edit for your cluster ---
module purge
module load cuda/12.1                     # matches the torch wheel installed in `topicmodel` (see README); change if you installed against a different CUDA version
# if `conda` isn't found here, uncomment and load whatever module puts it
# on PATH on a compute node (it may already be on PATH via your shell rc,
# as it was on the login node when this env was created):
# module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate topicmodel

# --- data location ---
# Code/sbatch/logs live in this checkout; all input/output DATA (PSI_NC_affiliation.csv,
# embeddings/, topics/) lives under IDR_DATA_DIR instead, so the two never
# mix. extract_embeddings.py reads this env var directly (and already
# defaults to the same path if unset) -- this export just makes it
# explicit and overridable via --export=IDR_DATA_DIR=... per submission.
export IDR_DATA_DIR="${IDR_DATA_DIR:-/rs1/researchers/a/amallav/IDR}"

# HF model cache is environment/software, not project data, so it stays
# alongside the code rather than under IDR_DATA_DIR.
HF_CACHE="${HF_CACHE:-/share/ftrscape/apethan/hf_cache}"

# SLURM copies this script to a spool dir before running it, so $0/dirname
# tricks resolve to the wrong place -- SLURM_SUBMIT_DIR is the directory
# `sbatch` was actually invoked from (06_topic_modeling/, per the README's
# usage examples) and is reliable regardless of spooling.
cd "$SLURM_SUBMIT_DIR"

python extract_embeddings.py \
    --model "$MODEL" \
    --batch-size 16 \
    --device cuda \
    --fp16 \
    --cache-dir "$HF_CACHE" \
    --offline
