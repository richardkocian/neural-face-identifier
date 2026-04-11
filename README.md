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
uv run --package evaluation run-wikiface-evaluation
uv run --package evaluation run-people-gator-evaluation
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

Evaluation defaults and examples:

- `run-wikiface-evaluation` defaults to WikiFace paths
- `run-people-gator-evaluation` defaults to PeopleGator paths
- default boxplot names: `top1_cosine_boxplot_wiki_face.png` / `top1_cosine_boxplot_people_gator.png`
- default Top-1 miss previews: up to 10 files named `top1_miss_XXX_wiki_face.jpg` or `top1_miss_XXX_people_gator.jpg`

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
