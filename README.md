# Neural Face Identifier

Authors: Richard Kocián (xkocia19), Karel Srna (xsrnak00), Tomáš Zgút (xzgutt00)

## uv workspace

The project is set up as an `uv` workspace with tree packages:
- `datasets` - utilities and datasets
- `evaluation` - model evaluation
- `traning` - model training

Basic commands:

```bash
uv sync
uv run --package datasets people-gator
uv run --package evaluation run-wikiface
uv run --package evaluation run-people-gator-embeddings --help
uv run --package evaluation run-people-gator-retrieval --help
uv run --package evaluation run-people-gator-retrieval-evaluate --help
uv run --package evaluation run-people-gator-retrieval-det --help
uv run --package evaluation run-people-gator-retrieval-boxplot --help
uv run --package evaluation run-people-gator-retrieval-gt --help
uv run --package training run-training
```

## People Gator CLI

Use the `people-gator` script from the `datasets` package:

```bash
uv run --package datasets people-gator <script-name> <args>
```

Examples:

```bash
# Find conflicting annotations
uv run --package datasets people-gator find-conflicts \
  --input-jsonl /path/to/people_gator__corresponding_faces__2026-02-11.test.jsonl \
  --output-csv /path/to/people_gator__corresponding_faces__2026-02-11.test__name_conflicts.csv

# Clean dataset using a conflicts CSV
uv run --package datasets people-gator clean-from-conflicts \
  --input-jsonl /path/to/people_gator__corresponding_faces__2026-02-11.test.jsonl \
  --conflicts-csv /path/to/people_gator__corresponding_faces__2026-02-11.test__name_conflicts.csv \
  --output-jsonl /path/to/people_gator__corresponding_faces__2026-02-11.test.cleaned.jsonl \
  --report-csv /path/to/people_gator__corresponding_faces__2026-02-11.test.cleanup_report.csv \
  --log-file /path/to/people_gator__corresponding_faces__2026-02-11.test.cleanup.log

# Split dataset by identity
uv run --package datasets people-gator split-dataset \
  --input-jsonl /path/to/people_gator__corresponding_faces__2026-02-11.test.jsonl \
  --seed 1 \
  --splits 0.8 \
  --split-names train test
```

## Evaluation defaults and examples:

### WikiFace evaluation

- `run-wikiface` is a dataset-specific script (WikiFace only).
- It always generates:
  - Top-1 cosine boxplot (`top1_cosine_boxplot_wikiface.png`)
  - DET plot + CSV (`top1_det_wikiface.png`, `top1_det_wikiface.csv`)
- default Top-1 miss previews: up to 10 files named `top1_miss_XXX_wiki_face.jpg`

```bash
# WikiFace (default)
uv run --package evaluation run-wikiface
```

### People Gator retrieval evaluation

```bash
# PeopleGator embeddings for retrieval

uv run --package evaluation run-people-gator-embeddings \
  --jsonl-path people_gator/people_gator__corresponding_faces__2026-02-11.test.cleaned.jsonl \
  --images-root people_gator/people_gator__data \
  --output-dir .embeddings/vit_small_patch8_gap_112_cosface_ms1mv3

# 1) Update dataset config paths if needed:
#    evaluation/src/peoplegator_namedfaces/retrieval/configs/dataset.template.json
# 2) Engine config:
#    evaluation/src/peoplegator_namedfaces/retrieval/configs/engine.image_embedding.json
uv run --package evaluation run-people-gator-retrieval \
  --dataset evaluation/src/peoplegator_namedfaces/retrieval/configs/dataset.template.json \
  --queries evaluation/src/peoplegator_namedfaces/retrieval/configs/image_queries.union.tst.jsonl \
  --engine evaluation/src/peoplegator_namedfaces/retrieval/configs/engine.image_embedding.json \
  --output evaluation_artifacts/retrieval.union.tst.pkl

# Build retrieval ground truth JSONL in the format expected by evaluate.py
uv run --package evaluation run-people-gator-retrieval-gt \
  --queries evaluation/src/peoplegator_namedfaces/retrieval/configs/image_queries.union.tst.jsonl \
  --annotations people_gator/people_gator__corresponding_faces__2026-02-11.test.cleaned.jsonl \
  --output evaluation_artifacts/retrieval.union.tst.ground_truth.jsonl

# Evaluate retrieval predictions
uv run --package evaluation run-people-gator-retrieval-evaluate \
  --predictions evaluation_artifacts/retrieval.union.tst.pkl \
  --ground-truth evaluation_artifacts/retrieval.union.tst.ground_truth.jsonl \
  --dataset evaluation/src/peoplegator_namedfaces/retrieval/configs/dataset.template.json \
  --top-k 1 5 10 \
  --ignore-index -1 \
  --output-file evaluation_artifacts/retrieval.union.tst.metrics.csv

# Bootstrap confidence retrieval predictions
uv run --package evaluation run-people-gator-retrieval-evaluate \
  --predictions evaluation_artifacts/retrieval.union.tst.pkl \
  --ground-truth evaluation_artifacts/retrieval.union.tst.ground_truth.jsonl \
  --dataset evaluation/src/peoplegator_namedfaces/retrieval/configs/dataset.template.json \
  --top-k 1 5 10 \
  --ignore-index -1 \
  --bootstrap-iters 10000 \
  --output-file evaluation_artifacts/retrieval.union.tst.metrics.bootstrap.csv

# DET curve (FPR vs FNR across score thresholds)
uv run --package evaluation run-people-gator-retrieval-det \
  --predictions evaluation_artifacts/retrieval.union.tst.pkl \
  --ground-truth evaluation_artifacts/retrieval.union.tst.ground_truth.jsonl \
  --dataset evaluation/src/peoplegator_namedfaces/retrieval/configs/dataset.template.json \
  --ignore-index -1 \
  --output-image evaluation_artifacts/retrieval.union.tst.det.png \
  --output-csv evaluation_artifacts/retrieval.union.tst.det.csv

# Top-1 cosine boxplot from retrieval predictions
# Also saves Top-1 wrong previews by default (top 10): query + predicted gallery + correct gallery (+ .txt metadata)
# Person names in preview labels are loaded from --annotations-jsonl.
uv run --package evaluation run-people-gator-retrieval-boxplot \
  --predictions evaluation_artifacts/retrieval.union.tst.pkl \
  --ground-truth evaluation_artifacts/retrieval.union.tst.ground_truth.jsonl \
  --dataset evaluation/src/peoplegator_namedfaces/retrieval/configs/dataset.template.json \
  --images-root people_gator/people_gator__data \
  --annotations-jsonl people_gator/people_gator__corresponding_faces__2026-02-11.test.cleaned.jsonl \
  --ignore-index -1 \
  --output-image evaluation_artifacts/retrieval.union.tst.top1_cosine_boxplot.png
```

`run-people-gator-retrieval-evaluate` loads prediction scores for each query, compares them against ground-truth relevant faces, and writes retrieval metrics (`precision`, `recall`, `f1`, `hitrate`, `map`, `mrr`, `ndcg`, `rprecision`, `auroc`) per `top-k` into CSV.
`run-people-gator-retrieval-det` builds a DET curve from the same predictions + ground truth and reports `EER` (equal error rate).
`run-people-gator-retrieval-boxplot` takes the Top-1 score per query, splits it into correct vs wrong retrievals, and saves a cosine-score boxplot (same style as legacy evaluation). It also saves Top-1 misclassified previews with person names for query/predicted/correct images.

## Retrieval metrics explained

All metrics are computed over ranked retrieval results for each query and then aggregated across queries.
`top-k` means we only look at the first `k` retrieved faces (for example `k=1`, `k=5`, `k=10`).

- `precision@k`: How many of the first `k` results are correct (relevant). Higher is better.
- `recall@k`: How many of all relevant faces were found in the first `k` results. Higher is better.
- `f1@k`: Harmonic mean of `precision@k` and `recall@k`. High only when both are high.
- `fallout@k`: Fraction of non-relevant items that were incorrectly retrieved in top-`k`. Lower is better.
- `hitrate@k`: Fraction of queries where at least one relevant face appears in top-`k`. Higher is better.
- `map@k`: Mean Average Precision at `k`; rewards ranking relevant results early, not only finding them. Higher is better.
- `mrr@k`: Mean Reciprocal Rank; focuses on position of the first correct result (earlier is better). Higher is better.
- `ndcg@k`: Normalized Discounted Cumulative Gain; rewards correct results near the top with position discounting. Higher is better.
- `rprecision`: Precision at `R`, where `R` is number of relevant items for that query. Higher is better.
- `auroc`: Area under ROC curve from positive vs negative scores. `0.5` is random, `1.0` is perfect.

Practical reading tip:
- For face retrieval, start with `hitrate@1`, `precision@1`, and `mrr` (how often and how early the first correct face appears), then use `map`/`ndcg` for overall ranking quality.

## What retrieval evaluation means here

In this project, retrieval evaluation answers this question:
"For each query face, does the system rank faces of the same person near the top?"

This is different from simple classification accuracy. We do not ask for one fixed class label only. Instead, we evaluate a ranked list of candidates.

### Step-by-step pipeline

1. `run.py` (`run-people-gator-retrieval`) loads:
   - dataset config (`dataset.template.json`) with face IDs and embedding locations
   - query list (`image_queries.union.tst.jsonl`)
   - retrieval engine config (`engine.image_embedding.json`)
2. For each query, the engine computes similarity scores against all gallery faces.
3. It stores ranked retrieval predictions into a single file (`.pkl`), e.g. `evaluation_artifacts/retrieval.union.tst.pkl`.
4. `evaluate.py` (`run-people-gator-retrieval-evaluate`) loads:
   - predictions from step 3
   - ground truth JSONL (which faces are relevant for each query)
   - top-k settings (`--top-k 1 5 10`)
5. For each query and each `k`, evaluation marks retrieved faces as relevant/non-relevant and computes metrics.
6. It aggregates across all queries and writes CSV with one row per metric and `top_k`.

## Bootstrap statistics

Without bootstrap, each metric has one deterministic value (same inputs -> same result).
With bootstrap enabled (`--bootstrap-iters N`), we estimate uncertainty of that metric.

### What bootstrap does

1. Keep the original query set of size `Q`.
2. Sample `Q` queries **with replacement** (some queries repeat, some are missing).
3. Recompute metric on this resampled query set.
4. Repeat `N` times (e.g. 1000).
5. Summarize the distribution of metric values.

Because each iteration uses a slightly different resampled set, metric values vary. This is expected and useful.

### How to read bootstrap output

Your bootstrap CSV stores summary statistics in a compact form such as:
`0.0644|0.0653|0.0408|0.1020|0.0204|0.0531|0.0735|0.1102`

   Interpretation (left -> right):
- mean
- median
- lower CI bound
- upper CI bound
- min
- Q1 (25%)
- Q3 (75%)
- max


## DET curve

DET curve plots:
- `x-axis`: `FPR` (false positive rate)
- `y-axis`: `FNR` (false negative rate)
- each point corresponds to one score threshold
- the saved figure is rendered in **probit (normal deviate) scale** on both axes (standard DET view)

## people gator dataset conflicts report
1. file `people_gator__corresponding_faces__2026-02-11.test.jsonl`:
- 78 conflicting annotations (more than 1 name per face):
    - 53 conflicts (completely diffent person)
    - 25 alias conflits (different name spelling)
        - 2 aliased names
- 4453 duplicate rows (same name, same face)
- after conflict resolution: 1399 unique name face pairs found

2. file `people_gator__corresponding_faces__2026-02-11.dev.jsonl`:
- 107 conflicting annotations (more than 1 name per face):
    - 30 conflicts (completely diffent person)
    - 77 alias conflits (different name spelling)
        - 7 aliased names
- 1103 duplicate rows (same name, same face)
- after conflict resolution: 1999 unique name face pairs found
