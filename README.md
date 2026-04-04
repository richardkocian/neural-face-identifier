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
```

