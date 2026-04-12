import pandas as pd
from pathlib import Path


def split_dataset(input_dataset: Path, seed: int, split: float, split_names: list[str]):
    df = pd.read_json(input_dataset, lines=True)

    identities = sorted(df["identity"].dropna().unique().tolist())
    shuffled_identities = (
        pd.Series(identities).sample(frac=1.0, random_state=seed).tolist()
    )

    train_count = int(len(shuffled_identities) * split)
    train_count = max(1, min(train_count, len(shuffled_identities) - 1))
    train_identities = set(shuffled_identities[:train_count])
    test_identities = set(shuffled_identities[train_count:])
    train_df = df[df["identity"].isin(train_identities)].reset_index(drop=True)
    test_df = df[df["identity"].isin(test_identities)].reset_index(drop=True)

    train_jonsl = input_dataset.with_name(
        f"{input_dataset.stem}_{split_names[0]}.jsonl"
    )
    test_jsonl = input_dataset.with_name(f"{input_dataset.stem}_{split_names[1]}.json")
    train_df.to_json(train_jonsl, lines=True, force_ascii=True, orient="records")
    test_df.to_json(test_jsonl, lines=True, force_ascii=True, orient="records")


def run(args):
    split_dataset(
        args.input_dataset,
        args.seed,
        args.splits,
        args.split_names,
    )
    return 0
