# Step 6 — Topic modeling

Turns `PSI_NC_affiliation.csv` (title + abstract) into a two-level topic hierarchy
using BERTopic on top of a choice of embedding models.

1. `extract_embeddings.py` (GPU): title/abstract → one embedding per publication.
2. `run_topic_model.py` (CPU): embeddings → **subtopics** (fine-grained) merged
   into **topics** (coarse-grained), plus quality metrics.

Everything runs on Hazel via `sbatch`. The login node is only used to
download model weights and to submit jobs.

## Where things live

| what | where |
|---|---|
| code, `sbatch/`, `logs/` | `/share/ftrscape/apethan/06_topic_modeling/` |
| input `PSI_NC_affiliation.csv` and all outputs (`embeddings/`, `topics/`) | `/rs1/researchers/a/amallav/IDR/` |
| Hugging Face model cache | `/share/ftrscape/apethan/hf_cache/` |
| conda env | `topicmodel` |

The scripts read the data directory from `IDR_DATA_DIR` (set in both sbatch
files, default `/rs1/researchers/a/amallav/IDR`). `PSI_NC_affiliation.csv`
(2,235 publications) needs the columns `Unnamed: 0` (the publication id,
written to the outputs as `pub_id`), `title` and `abstract`. If you rename
that column, pass `--id-col <name>` to both scripts.

## One-time setup

**1. Copy to Hazel** (from your laptop):
```bash
scp -r 06_topic_modeling apethan@login.hpc.ncsu.edu:/share/ftrscape/apethan/
scp PSI_NC_affiliation.csv apethan@login.hpc.ncsu.edu:/rs1/researchers/a/amallav/IDR/
```

**2. Create the conda env** (on Hazel):
```bash
conda deactivate                       # repeat if still inside another env
conda create -n topicmodel -c conda-forge python=3.11 -y
conda activate topicmodel
conda install -c conda-forge pip
module load cuda/12.1
python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
cd /share/ftrscape/apethan/06_topic_modeling
python -m pip install --no-cache-dir -r requirements.txt
```
If the env already exists, just add the new dependencies:
`python -m pip install --no-cache-dir gensim sentence-transformers`.

**3. Download the model weights** (login node; compute nodes are offline):
```bash
conda activate topicmodel
mkdir -p /share/ftrscape/apethan/hf_cache
cd /share/ftrscape/apethan/06_topic_modeling
python download_models.py --all --cache-dir /share/ftrscape/apethan/hf_cache

python download_models.py --model specter2 --cache-dir /share/ftrscape/apethan/hf_cache
python download_models.py --model scincl   --cache-dir /share/ftrscape/apethan/hf_cache
python download_models.py --model scibert  --cache-dir /share/ftrscape/apethan/hf_cache
python download_models.py --model gtr_t5   --cache-dir /share/ftrscape/apethan/hf_cache
python download_models.py --model mxbai    --cache-dir /share/ftrscape/apethan/hf_cache

```
It should end with `[done] weights cached`.

**4. Create the logs folder** (SLURM needs it before the job starts):
```bash
mkdir -p /share/ftrscape/apethan/06_topic_modeling/logs
```

**Old outputs** (from earlier versions and from the previous, larger input
file) no longer match and must be deleted before running:
```bash
rm -rf /rs1/researchers/a/amallav/IDR/embeddings /rs1/researchers/a/amallav/IDR/topics
```

## Run

Always submit from inside `06_topic_modeling/`.

```bash
cd /share/ftrscape/apethan/06_topic_modeling

# 1. Embeddings, one job per model (GPU)
sbatch -J emb_scibert  --export=MODEL=scibert  sbatch/extract_embeddings.sbatch
sbatch -J emb_scincl   --export=MODEL=scincl   sbatch/extract_embeddings.sbatch
sbatch -J emb_specter2 --export=MODEL=specter2 sbatch/extract_embeddings.sbatch
sbatch -J emb_gtr_t5   --export=MODEL=gtr_t5   sbatch/extract_embeddings.sbatch
sbatch -J emb_mxbai    --export=MODEL=mxbai    sbatch/extract_embeddings.sbatch

  ## Use to specify which GPU to be used
  sbatch --gres=gpu:rtx_2080:1 -J emb_<model> --export=MODEL=<model> sbatch/extract_embeddings.sbatch
    
    ### Use this to find the gpu:
      sinfo -p gpu -N -o "%N %G %T"


# 2. Topics, after that model's embeddings exist (CPU): 15 topics, and auto

  ## Let BERTopic select number of topics (max limit is set to 15)
  sbatch -J topic_scibert_auto  --export=MODEL=scibert,NR_TOPICS=auto  sbatch/run_topic_model.sbatch
  sbatch -J topic_scincl_auto   --export=MODEL=scincl,NR_TOPICS=auto   sbatch/run_topic_model.sbatch
  sbatch -J topic_specter2_auto --export=MODEL=specter2,NR_TOPICS=auto sbatch/run_topic_model.sbatch
  sbatch -J topic_gtr_t5_auto   --export=MODEL=gtr_t5,NR_TOPICS=auto   sbatch/run_topic_model.sbatch
  sbatch -J topic_mxbai_auto    --export=MODEL=mxbai,NR_TOPICS=auto    sbatch/run_topic_model.sbatch

  ## Hardcode number of topics as 15
  sbatch -J topic_scibert_15    --export=MODEL=scibert,NR_TOPICS=15    sbatch/run_topic_model.sbatch
  sbatch -J topic_scincl_15     --export=MODEL=scincl,NR_TOPICS=15     sbatch/run_topic_model.sbatch
  sbatch -J topic_specter2_15   --export=MODEL=specter2,NR_TOPICS=15   sbatch/run_topic_model.sbatch
  sbatch -J topic_gtr_t5_15     --export=MODEL=gtr_t5,NR_TOPICS=15     sbatch/run_topic_model.sbatch
  sbatch -J topic_mxbai_15      --export=MODEL=mxbai,NR_TOPICS=15      sbatch/run_topic_model.sbatch
```

| variable | default | meaning |
|---|---|---|
| `MODEL` | – | `specter2`, `scincl`, `scibert`, `gtr_t5` or `mxbai` (see Models) |
| `NR_TOPICS` | `15` | number of topics, or `auto` to let BERTopic decide |
| `MAX_TOPICS` | `15` | hard limit on the topic count when `NR_TOPICS=auto` |
| `MIN_TOPIC_SIZE` | `12` | smallest allowed subtopic, in publications (smaller = more subtopics). Rarely needs changing. |

**Models**

| `MODEL` | model | type | dim | notes |
|---|---|---|---|---|
| `specter2` | `allenai/specter2_base` + proximity adapter | scientific | 768 | |
| `scincl` | `malteos/scincl` | scientific | 768 | |
| `scibert` | `allenai/scibert_scivocab_uncased` | scientific, mean pooling | 768 | weaker baseline |
| `gtr_t5` | `sentence-transformers/gtr-t5-large` | general purpose | 768 | runs in fp32 |
| `mxbai` | `mixedbread-ai/mxbai-embed-large-v1` | general purpose | 1024 | runs in fp32 |

`-J` names the job and its logs: `logs/<jobname>_<jobid>.out` / `.err`.

**Check a run:**
```bash
squeue -u $USER                                    # empty = finished
sacct -u $USER -S today --format=JobID,JobName%22,State,ExitCode,Elapsed
cat logs/topic_scibert_auto_*.err                    # Python errors, if any
cat logs/topic_scibert_15_*.out                    # progress + resource summary
ls -la /rs1/researchers/a/amallav/IDR/embeddings/ /rs1/researchers/a/amallav/IDR/topics/
```
A good embeddings log ends with `[done] run run_topic_model.py next`; a good
topics log ends with `[done]`.

## Outputs

**`$IDR_DATA_DIR/embeddings/`** (per model)

| file | contents |
|---|---|
| `embeddings_<model>.npy` | one embedding vector per publication, in input-file row order |
| `ids_<model>.csv` | `pub_id` of each row of the `.npy` |

**`$IDR_DATA_DIR/topics/`** (`<n>` is `15` or `auto`)

| file | columns |
|---|---|
| `topics_<model>_<n>.csv` | `topic_id`, `size`, `topic_label`, `subtopics` |
| `doc_topics_<model>_<n>.csv` | `pub_id`, `topic_id`, `topic_label`, `subtopic_id`, `subtopic` |
| `model_comparison.csv` | `model`, `nr_topics`, `n_subtopics`, `n_topics`, `coherence_cv`, `topic_diversity` |

- **Subtopics** are BERTopic's natural clusters. Publications HDBSCAN could
  not place are moved to their nearest cluster (`reduce_outliers`), so every
  publication has exactly one subtopic and one topic; there is no `-1`.
  A subtopic label is its top 4 keywords.
- **Topics** are the subtopics merged by BERTopic's topic reduction. A topic
  label is a comma-separated **keyword list**, e.g. `data analytics,
  modeling, artificial intelligence`: the 4 highest-scoring keywords among its
  subtopics' keywords, so every word in a topic label also appears in one of
  its `subtopics` labels. It is a list of keywords, not a sentence; that is
  intended.
- `topics_*.csv`: `size` = publications in the topic; `subtopics` = the
  labels of its child subtopics, separated by `;`.
- `doc_topics_*.csv`: one row per publication; `pub_id` is the same id as in
  `faculty_authorship_long.csv`.
- `model_comparison.csv`: one row per (`model`, `nr_topics`) run, where
  `nr_topics` is `15` or `auto`, so all models and settings are in one table.
  Re-running a job replaces its row instead of adding a duplicate.
  `coherence_cv` is topic coherence (C_v, gensim, top 10 words per topic;
  higher is better). `topic_diversity` is the share of unique words among each
  topic's top 25 words (Dieng et al. 2020; 1.0 = no repeated words, low =
  near-duplicate topics).

## Copy outputs to your laptop

Run on your laptop, not on Hazel:
```bash
mkdir -p ~/Desktop/Aditi/IDR/step6_outputs
scp -r apethan@login.hpc.ncsu.edu:/rs1/researchers/a/amallav/IDR/topics     ~/Desktop/Aditi/IDR/06_topic_modeling/
scp -r apethan@login.hpc.ncsu.edu:/rs1/researchers/a/amallav/IDR/embeddings ~/Desktop/Aditi/IDR/06_topic_modeling/
scp -r apethan@login.hpc.ncsu.edu:/share/ftrscape/apethan/06_topic_modeling/logs ~/Desktop/Aditi/IDR/06_topic_modeling/
```
Use `rsync -avz` in place of `scp -r` for repeat copies.

## Adding an embedding model

Write one class in `embedders.py`; nothing else changes.
```python
@register_embedder("your-key")
class YourEmbedder(BaseEmbedder):
    def __init__(self, model_name="...", **kwargs):
        super().__init__(model_name, **kwargs)
        # load the model; set self.dim

    def prepare_input(self, title, abstract) -> str: ...
    def encode_batch(self, texts: list[str]) -> np.ndarray: ...   # (len(texts), self.dim) float32
```
Then run `download_models.py --model your-key ...`, and use `MODEL=your-key`
in the sbatch commands.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `sbatch: error: --gres=gpu:<count> is not allowed. You must specify a GPU type` | Hazel needs `gpu:<type>:<count>`; the script uses `gpu:a30:1`. If the GPU queue is busy, check `sinfo -p gpu -o "%N|%40G|%T"` and switch to an idle type. |
| `python: can't open file '/var/spool/slurmd/...py'` | Submitted from the wrong directory. Submit from `06_topic_modeling/`. |
| `.err`: `LocalEntryNotFoundError` / `couldn't connect to 'https://huggingface.co'` | Weights are not in the cache and jobs run `--offline`. Run `download_models.py` on the login node. |
| No log files / job fails instantly with no output | `logs/` did not exist. `mkdir -p logs` and resubmit. |
| `.../ids_<model>.csv not found — run extract_embeddings.py` | Run the embeddings job for that model first (and delete old outputs, see setup). |
| `Column 'Unnamed: 0' not found` | The input file has no such id column (the error lists the columns it found). Pass `--id-col <name>` if it was renamed. |
| specter2 `.err`: `There are adapters available but none are activated for the forward pass.` (one line) | Harmless. It is printed while the model loads, before the adapter is switched on. The embeddings are fine as long as the `.out` log has `[info] active adapter(s): Stack[proximity]`. |
| `ModuleNotFoundError: gensim` | `python -m pip install --no-cache-dir gensim` in the `topicmodel` env. |
| `Disk quota exceeded` during `pip install` | Always use `--no-cache-dir`; clear with `python -m pip cache purge`. |
| `NoChannelsConfiguredError` | Add `-c conda-forge` to `conda create`/`install`, or `conda config --add channels conda-forge`. |
| `conda: command not found` inside a job | Uncomment `# module load anaconda3` in the `.sbatch` file. |
