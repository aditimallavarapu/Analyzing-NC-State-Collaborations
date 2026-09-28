"""
run_topic_model.py — step 6, part 2 (CPU).

Fits BERTopic on the embeddings from extract_embeddings.py and produces a
two-level hierarchy:

  subtopics  fine-grained: BERTopic's natural HDBSCAN clusters, with the
             documents HDBSCAN left unassigned (-1) moved to their nearest
             cluster using BERTopic's reduce_outliers(). Labelled with the
             top 4 c-TF-IDF keywords.
  topics     coarse-grained: the subtopics merged by BERTopic's topic
             reduction (--nr-topics, default 15, or 'auto'). Labelled with
             the 4 highest-scoring keywords among its subtopics' keywords, as
             a comma-separated keyword list (not a sentence), so every topic
             keyword also appears in one of its subtopic labels.

Every publication ends up in exactly one subtopic and one topic; nothing is
left as -1.

Run once per embedding model:

    python run_topic_model.py --model scibert                   # 15 topics
    python run_topic_model.py --model scibert --nr-topics auto  # data-driven, at most --max-topics

Outputs (in $IDR_DATA_DIR/topics by default; <n> is the --nr-topics value):
    topics_<model>_<n>.csv       topic_id, size, topic_label, subtopics
    doc_topics_<model>_<n>.csv   pub_id, topic_id, topic_label, subtopic_id, subtopic
    model_comparison.csv         one row per (model, <n>): model, nr_topics,
                                 n_subtopics, n_topics, coherence_cv, topic_diversity
"""
from __future__ import annotations

import argparse
import fcntl
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd

# data on /rs1/researchers/a/amallav/IDR, code on /share/ftrscape/apethan
# (both sbatch scripts export IDR_DATA_DIR to match)
DEFAULT_DATA_DIR = os.environ.get("IDR_DATA_DIR", "/rs1/researchers/a/amallav/IDR")

SUBTOPIC_LABEL_TERMS = 4   # keywords in a subtopic label
TOPIC_LABEL_TERMS = 4      # keywords in a topic label
TOP_N_WORDS = 25           # c-TF-IDF words kept per topic (topic diversity uses all 25)
COHERENCE_TOP_N = 10       # words per topic used for C_v


def _nr_topics_arg(value: str):
    if value.lower() == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--nr-topics must be an integer or 'auto', got '{value}'")
    if n < 2:
        raise argparse.ArgumentTypeError("--nr-topics must be >= 2")
    return n


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                    help="embedding model whose embeddings to cluster; must "
                         "match the --model used in extract_embeddings.py")
    p.add_argument("--embeddings-dir", default=f"{DEFAULT_DATA_DIR}/embeddings",
                    help="default: $IDR_DATA_DIR/embeddings")
    p.add_argument("--input", default=f"{DEFAULT_DATA_DIR}/PSI_NC_affiliation.csv",
                    help="publications CSV, for the title/abstract text used "
                         "to pick topic keywords (default: "
                         "$IDR_DATA_DIR/PSI_NC_affiliation.csv)")
    p.add_argument("--id-col", default="Unnamed: 0",
                    help="publication id column of the input file "
                         "(written to the outputs as 'pub_id')")
    p.add_argument("--title-col", default="title")
    p.add_argument("--abstract-col", default="abstract")
    p.add_argument("--output-dir", default=f"{DEFAULT_DATA_DIR}/topics",
                    help="default: $IDR_DATA_DIR/topics")

    p.add_argument("--min-topic-size", type=int, default=12,
                    help="smallest allowed subtopic, in publications "
                         "(HDBSCAN min_cluster_size); smaller gives more "
                         "subtopics")
    p.add_argument("--nr-topics", type=_nr_topics_arg, default=15,
                    help="number of coarse topics to merge the subtopics "
                         "into: an integer (default 15) or 'auto' to let "
                         "BERTopic decide")
    p.add_argument("--max-topics", type=int, default=15,
                    help="hard limit on the number of topics when "
                         "--nr-topics auto (default 15); ignored otherwise")
    p.add_argument("--seed", type=int, default=42,
                    help="random_state for UMAP, fixed so runs across "
                         "different embedding models are comparable")
    return p


def load_embeddings_and_ids(args) -> tuple[np.ndarray, list[str]]:
    d = Path(args.embeddings_dir)
    emb_path, ids_path = d / f"embeddings_{args.model}.npy", d / f"ids_{args.model}.csv"
    for path in (emb_path, ids_path):
        if not path.exists():
            raise SystemExit(f"{path} not found — run "
                             f"extract_embeddings.py --model {args.model} first")
    embeddings = np.load(emb_path)
    ids = pd.read_csv(ids_path, dtype=str)["pub_id"].tolist()
    if len(ids) != embeddings.shape[0]:
        raise SystemExit(f"Row count mismatch: {embeddings.shape[0]} embedding "
                         f"rows vs {len(ids)} ids")
    if np.isnan(embeddings).any():
        raise SystemExit("Embeddings contain NaNs — re-run extract_embeddings.py")
    print(f"[info] loaded {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")
    return embeddings, ids


def keyword_label(terms, n: int) -> str:
    """The top-n c-TF-IDF terms as 'a, b, c'."""
    return ", ".join(term for term, _ in terms[:n] if term)


def topic_keywords(children_terms, n: int) -> str:
    """The n highest-scoring keywords among a topic's subtopic keywords. A
    keyword's score is its c-TF-IDF weight summed over the subtopics that
    list it."""
    score = {}
    for terms in children_terms:
        for term, weight in terms:
            if term:
                score[term] = score.get(term, 0) + weight
    return ", ".join(sorted(score, key=score.get, reverse=True)[:n])


def assign_leftover_outliers(topics: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Guard: reduce_outliers() can leave a document at -1 (e.g. zero
    similarity to every cluster). Give any such document the cluster whose
    embedding centroid is nearest by cosine similarity."""
    left = topics == -1
    if not left.any():
        return topics
    ids = np.unique(topics[~left])
    unit = lambda m: m / np.linalg.norm(m, axis=1, keepdims=True)
    centroids = np.stack([embeddings[topics == t].mean(axis=0) for t in ids])
    topics = topics.copy()
    topics[left] = ids[(unit(embeddings[left]) @ unit(centroids).T).argmax(axis=1)]
    return topics


def main(args: argparse.Namespace) -> None:
    from bertopic import BERTopic
    from bertopic.cluster import BaseCluster
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.vectorizers import ClassTfidfTransformer
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    if args.nr_topics == "auto" and args.max_topics < 2:
        raise SystemExit("--max-topics must be >= 2")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    n_label = str(args.nr_topics)

    embeddings, ids = load_embeddings_and_ids(args)

    df = pd.read_csv(args.input, dtype={args.id_col: str})
    for col in (args.id_col, args.title_col, args.abstract_col):
        if col not in df.columns:
            raise SystemExit(
                f"Column '{col}' not found in {args.input}. "
                f"Available columns: {list(df.columns)}"
            )
    if df[args.id_col].duplicated().any():
        raise SystemExit(f"--id-col '{args.id_col}' has duplicate values in "
                          f"{args.input} — it must uniquely identify each publication")
    df = df.set_index(df[args.id_col])
    missing = [pid for pid in ids if pid not in df.index]
    if missing:
        raise SystemExit(
            f"{len(missing)} id(s) from the embeddings not found in {args.input} "
            f"(first few: {missing[:5]})"
        )
    df = df.loc[ids]

    # Human-readable text for c-TF-IDF keywords; independent of the
    # model-specific input format used for the embeddings.
    titles = df[args.title_col].fillna("").astype(str)
    abstracts = df[args.abstract_col].fillna("").astype(str)
    docs = [f"{t}. {a}" if a else t for t, a in zip(titles, abstracts)]

    def new_model(**parts):
        return BERTopic(
            vectorizer_model=CountVectorizer(stop_words="english",
                                             ngram_range=(1, 2), min_df=2),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            top_n_words=TOP_N_WORDS,
            calculate_probabilities=False,
            verbose=True,
            **parts,
        )

    # 1. Natural clusters (HDBSCAN marks unassigned documents as -1).
    print("[info] fitting BERTopic on precomputed embeddings")
    model = new_model(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric="cosine", random_state=args.seed),
        hdbscan_model=HDBSCAN(min_cluster_size=args.min_topic_size,
                              metric="euclidean", cluster_selection_method="eom"),
    )
    topics, _ = model.fit_transform(docs, embeddings=embeddings)
    n_outliers = sum(t == -1 for t in topics)
    print(f"[info] {len(set(topics) - {-1})} clusters, {n_outliers} unassigned "
          f"documents ({100 * n_outliers / len(topics):.1f}%)")

    # 2. Move every unassigned document to its nearest cluster.
    if n_outliers:
        topics = model.reduce_outliers(docs, topics, strategy="embeddings",
                                       embeddings=embeddings)
    topics = assign_leftover_outliers(np.array(topics), embeddings)

    # 3. Subtopics = those clusters. Rebuild a clean model from the fixed
    #    assignments (BERTopic's "manual topic modeling": no UMAP/HDBSCAN,
    #    topics given via y) so c-TF-IDF and the merging below start from a
    #    consistent state with no -1 group.
    model = new_model(umap_model=BaseDimensionalityReduction(),
                      hdbscan_model=BaseCluster())
    sub_ids, _ = model.fit_transform(docs, embeddings=embeddings, y=topics)
    sub_ids = np.array(sub_ids)
    sub_terms = {t: model.get_topic(t)[:SUBTOPIC_LABEL_TERMS]
                 for t in sorted(set(sub_ids))}
    sub_labels = {t: keyword_label(terms, SUBTOPIC_LABEL_TERMS)
                  for t, terms in sub_terms.items()}
    n_subtopics = len(sub_labels)
    print(f"[info] {n_subtopics} subtopics")

    # 4. Topics = subtopics merged by BERTopic's topic reduction.
    model.reduce_topics(docs, nr_topics=args.nr_topics)
    if args.nr_topics == "auto" and len(set(model.topics_)) > args.max_topics:
        print(f"[info] auto gave {len(set(model.topics_))} topics; "
              f"reducing to --max-topics {args.max_topics}")
        model.reduce_topics(docs, nr_topics=args.max_topics)
    topic_of_doc = np.array(model.topics_)
    topic_ids = sorted(t for t in model.get_topics() if t != -1)
    topic_of_sub = {}
    for s, t in zip(sub_ids, topic_of_doc):
        if topic_of_sub.setdefault(s, t) != t:
            raise SystemExit("internal error: a subtopic was split across topics")
    topic_labels = {
        t: topic_keywords([sub_terms[s] for s, tt in topic_of_sub.items() if tt == t],
                          TOPIC_LABEL_TERMS)
        for t in topic_ids}
    print(f"[info] {len(topic_ids)} topics")

    # ---- output tables
    doc_df = pd.DataFrame({
        "pub_id": ids,
        "topic_id": topic_of_doc,
        "topic_label": [topic_labels[t] for t in topic_of_doc],
        "subtopic_id": sub_ids,
        "subtopic": [sub_labels[t] for t in sub_ids],
    })
    if (doc_df["topic_id"] == -1).any() or (doc_df["subtopic_id"] == -1).any():
        raise SystemExit("internal error: a publication was left at topic -1")

    rows = []
    for t in topic_ids:
        members = doc_df[doc_df["topic_id"] == t]
        subs = (members.groupby(["subtopic_id", "subtopic"]).size()
                .reset_index(name="n").sort_values("n", ascending=False))
        rows.append({"topic_id": t, "size": len(members),
                     "topic_label": topic_labels[t],
                     "subtopics": "; ".join(subs["subtopic"])})
    topics_df = pd.DataFrame(rows)

    doc_df.to_csv(outdir / f"doc_topics_{args.model}_{n_label}.csv", index=False)
    topics_df.to_csv(outdir / f"topics_{args.model}_{n_label}.csv", index=False)

    # ---- evaluation, at the topic level
    def topic_terms(t, n):
        return [w for w, _ in model.get_topic(t) if w][:n]

    # C_v (gensim) over each topic's top words vs. the tokenized corpus.
    # Topic terms can be bigrams, so split them into single words first.
    analyzer = CountVectorizer(stop_words="english").build_analyzer()
    texts = [analyzer(d) for d in docs]
    dictionary = Dictionary(texts)
    word_lists = []
    for t in topic_ids:
        words = []
        for term in topic_terms(t, COHERENCE_TOP_N):
            words += [w for w in term.split()
                      if w in dictionary.token2id and w not in words]
        word_lists.append(words[:COHERENCE_TOP_N])
    coherence_cv = CoherenceModel(topics=word_lists, texts=texts,
                                  dictionary=dictionary, coherence="c_v",
                                  processes=1).get_coherence()

    # Topic diversity (Dieng et al. 2020): unique words among the top-25
    # words of all topics / (25 * n_topics).
    top25 = [w for t in topic_ids for w in topic_terms(t, TOP_N_WORDS)]
    topic_diversity = len(set(top25)) / (TOP_N_WORDS * len(topic_ids))
    print(f"[info] coherence C_v = {coherence_cv:.4f}, "
          f"topic diversity = {topic_diversity:.4f}")

    # One row per (model, nr_topics): re-running replaces its row instead of
    # adding a duplicate. The lock stops jobs that finish at the same time
    # from overwriting each other's rows.
    row = {"model": args.model, "nr_topics": n_label,
           "n_subtopics": n_subtopics, "n_topics": len(topic_ids),
           "coherence_cv": round(coherence_cv, 4),
           "topic_diversity": round(topic_diversity, 4)}
    comparison = outdir / "model_comparison.csv"
    with open(comparison, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        text = f.read()
        table = pd.DataFrame([row])
        if text.strip():
            old = pd.read_csv(io.StringIO(text), dtype={"nr_topics": str})
            old = old[~((old["model"] == row["model"])
                        & (old["nr_topics"] == row["nr_topics"]))]
            table = pd.concat([old, table], ignore_index=True)
        f.seek(0)
        f.truncate()
        table.sort_values(["model", "nr_topics"]).to_csv(f, index=False)

    print(f"[info] wrote topics_{args.model}_{n_label}.csv, "
          f"doc_topics_{args.model}_{n_label}.csv and a row in {comparison}")
    print("[done]")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
