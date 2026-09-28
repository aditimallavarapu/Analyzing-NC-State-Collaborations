"""
extract_embeddings.py — step 6, part 1 (GPU).

Reads titles/abstracts from PSI_NC_affiliation.csv and writes one embedding vector
per publication for a single embedding model (see embedders.py for the
available --model values). Run it once per model:

    python extract_embeddings.py --model scibert
    python extract_embeddings.py --model scincl
    python extract_embeddings.py --model specter2

Outputs (in $IDR_DATA_DIR/embeddings by default):
    embeddings_<model>.npy   (n_docs, dim) float32, in input-file row order
    ids_<model>.csv          pub_id of each row of the .npy

There is no checkpointing: a run takes a minute or two, so if it fails just
run it again (it overwrites the outputs).

Data location: --input/--output-dir default to $IDR_DATA_DIR/PSI_NC_affiliation.csv
and $IDR_DATA_DIR/embeddings (IDR_DATA_DIR defaults to
/rs1/researchers/a/amallav/IDR; the sbatch scripts export it).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from embedders import available_embedders, get_embedder

# All input/output *data* (PSI_NC_affiliation.csv, embeddings, topics) lives under one
# data directory, separate from this code/sbatch/logs checkout — code on
# /share/ftrscape/apethan, data on /rs1/researchers/a/amallav/IDR. Point
# IDR_DATA_DIR at it (both sbatch scripts export this) rather than editing
# path flags everywhere; falls back to this project's known data dir if unset.
DEFAULT_DATA_DIR = os.environ.get("IDR_DATA_DIR", "/rs1/researchers/a/amallav/IDR")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=f"{DEFAULT_DATA_DIR}/PSI_NC_affiliation.csv",
                    help="path to the publications CSV (default: "
                         "$IDR_DATA_DIR/PSI_NC_affiliation.csv)")
    p.add_argument("--id-col", default="Unnamed: 0",
                    help="publication id column of the input file "
                         "(written to ids_<model>.csv as 'pub_id')")
    p.add_argument("--title-col", default="title")
    p.add_argument("--abstract-col", default="abstract")
    p.add_argument("--output-dir", default=f"{DEFAULT_DATA_DIR}/embeddings",
                    help="output directory (default: $IDR_DATA_DIR/embeddings)")
    p.add_argument("--model", required=True, choices=available_embedders(),
                    help="embedder registry key (see embedders.py)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                    help="use fp16 autocast on GPU (ignored on CPU)")
    p.add_argument("--cache-dir", default=None,
                    help="HuggingFace cache dir. Pre-download models into this "
                         "directory on a login node if compute nodes are offline.")
    p.add_argument("--offline", action="store_true",
                    help="set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE before loading "
                         "the model — use once models are already cached locally")
    return p


def main(args: argparse.Namespace) -> None:
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    emb_path = outdir / f"embeddings_{args.model}.npy"
    ids_path = outdir / f"ids_{args.model}.csv"

    # dtype=str on the id column avoids pandas silently coercing ids to
    # int64 (which would drop e.g. leading zeros) before we ever see them
    df = pd.read_csv(args.input, dtype={args.id_col: str})
    for col in (args.id_col, args.title_col, args.abstract_col):
        if col not in df.columns:
            raise SystemExit(
                f"Column '{col}' not found in {args.input}. "
                f"Available columns: {list(df.columns)}"
            )
    ids = df[args.id_col].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise SystemExit(f"--id-col '{args.id_col}' has duplicate values — it "
                          f"must uniquely identify each publication")
    n = len(df)
    n_missing_abs = df[args.abstract_col].isna().sum()
    print(f"[info] {n} publications, {n_missing_abs} missing abstracts "
          f"({100 * n_missing_abs / n:.1f}%)")

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device} fp16={args.fp16 and device == 'cuda'} model={args.model}")

    embedder = get_embedder(
        args.model, device=device, fp16=args.fp16, max_length=args.max_length,
        cache_dir=args.cache_dir,
    )
    texts = [
        embedder.prepare_input(t, a)
        for t, a in zip(df[args.title_col], df[args.abstract_col])
    ]

    # length-bucketed batching: sort rows by text length so batches have
    # similar length and pad less, then scatter results back to their original
    # positions (implicit "unsort").
    embeddings = np.zeros((n, embedder.dim), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])
    batches = [order[i:i + args.batch_size] for i in range(0, n, args.batch_size)]
    print(f"[info] encoding {n} docs in {len(batches)} batches")
    t0 = time.time()
    for batch_idx in tqdm(batches, file=sys.stdout):
        embeddings[batch_idx] = embedder.encode_batch([texts[i] for i in batch_idx])
    print(f"[info] encoding took {time.time() - t0:.1f}s")

    if np.isnan(embeddings).any():
        raise SystemExit("Embeddings contain NaNs — re-run with --no-fp16")

    np.save(emb_path, embeddings)
    pd.DataFrame({"pub_id": ids}).to_csv(ids_path, index=False)
    print(f"[info] wrote {emb_path} and {ids_path}")
    print("[done] run run_topic_model.py next")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
