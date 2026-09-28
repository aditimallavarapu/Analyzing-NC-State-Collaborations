"""
download_models.py — pre-populate the Hugging Face cache on a node with
internet access (the login node), so later `sbatch` jobs on offline
compute nodes never need network access.

This ONLY instantiates the tokenizer/model (and, for SPECTER2, loads its
adapter) — it does not read the input CSV and never runs encode_batch on any
text, so it's cheap and appropriate to run directly on the login node
rather than through sbatch. extract_embeddings.py, by contrast, always
does real work (encoding the whole corpus) and belongs on a GPU compute
node via sbatch/extract_embeddings.sbatch, even though it happens to
accept --device cpu too.

Usage:
    python download_models.py --model specter2 --cache-dir /share/ftrscape/apethan/hf_cache
    python download_models.py --model scincl   --cache-dir /share/ftrscape/apethan/hf_cache
    python download_models.py --model scibert  --cache-dir /share/ftrscape/apethan/hf_cache
    python download_models.py --model gtr_t5   --cache-dir /share/ftrscape/apethan/hf_cache
    python download_models.py --model mxbai    --cache-dir /share/ftrscape/apethan/hf_cache
    # or all registered embedders in one call:
    python download_models.py --all --cache-dir /share/ftrscape/apethan/hf_cache
"""
from __future__ import annotations

import argparse

from embedders import available_embedders, get_embedder


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", choices=available_embedders(),
                   help="embedder registry key to download")
    g.add_argument("--all", action="store_true",
                   help="download every registered embedder")
    p.add_argument("--cache-dir", required=True,
                   help="HuggingFace cache dir — pass the same path as "
                        "extract_embeddings.py's --cache-dir")
    return p


def main(args: argparse.Namespace) -> None:
    keys = available_embedders() if args.all else [args.model]
    for key in keys:
        print(f"[info] downloading weights for '{key}' into {args.cache_dir}")
        embedder = get_embedder(key, device="cpu", cache_dir=args.cache_dir)
        print(f"[info] '{key}' ready (dim={embedder.dim})")
    print("[done] weights cached — sbatch jobs with --offline can now find them")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
