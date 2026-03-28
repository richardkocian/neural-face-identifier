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
uv run --package evaluation run-evaluation
uv run --package training run-training
```
