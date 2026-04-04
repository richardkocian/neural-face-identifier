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
```

