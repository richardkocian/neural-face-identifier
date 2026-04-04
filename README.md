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
uv run --package datasets make-wiki-face-split
uv run --package evaluation run-wikiface-evaluation
uv run --package evaluation run-people-gator-evaluation
uv run --package evaluation run-people-gator-embeddings --help
uv run --package evaluation run-people-gator-retrieval --help
uv run --package evaluation run-people-gator-retrieval-evaluate --help
uv run --package training run-training
```

Evaluation defaults and examples:

- `run-wikiface-evaluation` defaults to WikiFace paths
- `run-people-gator-evaluation` defaults to PeopleGator paths
- default boxplot names: `top1_cosine_boxplot_wikiface.png` / `top1_cosine_boxplot_people_gator.png`
- default Top-1 miss previews: up to 10 files named `top1_miss_XXX_wikiface.jpg` or `top1_miss_XXX_people_gator.jpg`

```bash
# WikiFace (default)
uv run --package evaluation run-wikiface-evaluation

# PeopleGator (default)
uv run --package evaluation run-people-gator-evaluation

# Explicit PeopleGator paths
uv run --package evaluation run-people-gator-evaluation \
  --jsonl-path people_gator__corresponding_faces__2026-02-11.test.jsonl \
  --images-root people_gator__data

# Retrieval (image queries)
# Generate embeddings from PeopleGatorDataset + timm model
uv run --package evaluation run-people-gator-embeddings \
  --jsonl-path people_gator/people_gator__corresponding_faces__2026-02-11.test.jsonl \
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
uv run python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

repo = Path(".")
queries_path = repo / "evaluation/src/peoplegator_namedfaces/retrieval/configs/image_queries.union.tst.jsonl"
ann_path = repo / "people_gator/people_gator__corresponding_faces__2026-02-11.test.jsonl"
out_path = repo / "evaluation_artifacts/retrieval.union.tst.ground_truth.jsonl"

face_to_name = {}
name_to_faces = defaultdict(set)
with ann_path.open("r") as f:
    for line in f:
        row = json.loads(line)
        face = row["face"]
        name = row["person_name"]
        face_to_name[face] = name
        name_to_faces[name].add(face)

with queries_path.open("r") as f_in, out_path.open("w") as f_out:
    for line in f_in:
        q = json.loads(line)
        person = face_to_name[q["query"]]
        record = {
            "query": q["query"],
            "query_type": q["query_type"],
            "faces": sorted(name_to_faces[person]),
        }
        f_out.write(json.dumps(record) + "\n")

print(out_path)
PY

# Evaluate retrieval predictions (torchmetrics retrieval metrics -> CSV)
uv run --package evaluation run-people-gator-retrieval-evaluate \
  --predictions evaluation_artifacts/retrieval.union.tst.pkl \
  --ground-truth evaluation_artifacts/retrieval.union.tst.ground_truth.jsonl \
  --dataset evaluation/src/peoplegator_namedfaces/retrieval/configs/dataset.template.json \
  --top-k 1 5 10 \
  --ignore-index -1 \
  --output-file evaluation_artifacts/retrieval.union.tst.metrics.csv

# Optional: add bootstrap confidence intervals (slower)
uv run --package evaluation run-people-gator-retrieval-evaluate \
  --predictions evaluation_artifacts/retrieval.union.tst.pkl \
  --ground-truth evaluation_artifacts/retrieval.union.tst.ground_truth.jsonl \
  --dataset evaluation/src/peoplegator_namedfaces/retrieval/configs/dataset.template.json \
  --top-k 1 5 10 \
  --ignore-index -1 \
  --bootstrap-iters 1000 \
  --output-file evaluation_artifacts/retrieval.union.tst.metrics.bootstrap.csv
```

`run-people-gator-retrieval-evaluate` loads prediction scores for each query, compares them against ground-truth relevant faces, and writes retrieval metrics (`precision`, `recall`, `f1`, `hitrate`, `map`, `mrr`, `ndcg`, `rprecision`, `auroc`) per `top-k` into CSV.

