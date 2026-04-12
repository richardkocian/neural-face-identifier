import pandas as pd


def find_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Return faces that have conflicting identities across multiple annotators.

    A face is considered conflicting when it has more than one distinct
    person_name and these names come from more than one annotator.

    Args:
        df: Input annotations dataframe loaded from PeopleGator JSONL.

    Returns:
        A dataframe with one row per conflicting face and summary columns.

    Raises:
        ValueError: If required columns are missing in the input dataframe.
    """
    image_col = "face"
    label_col = "person_name"
    annotator_col = "annotator"

    required_cols = {image_col, label_col, annotator_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    pairs = df[[image_col, label_col, annotator_col]].dropna()

    stats = (
        pairs.groupby(image_col)
        .agg(
            name_count=(label_col, "nunique"),
            annotator_count=(annotator_col, "nunique"),
        )
        .reset_index()
    )

    conflicting_files = stats[
        (stats["name_count"] > 1) & (stats["annotator_count"] > 1)
    ][image_col]

    if conflicting_files.empty:
        return pd.DataFrame(
            columns=[image_col, "name_count", "annotator_count", "names", "annotators"]
        )

    conflicts = (
        pairs[pairs[image_col].isin(conflicting_files)]
        .groupby(image_col)
        .agg(
            names=(label_col, lambda values: sorted(set(values))),
            annotators=(annotator_col, lambda values: sorted(set(values))),
        )
        .reset_index()
    )
    conflicts["name_count"] = conflicts["names"].apply(len)
    conflicts["annotator_count"] = conflicts["annotators"].apply(len)

    return conflicts[
        [image_col, "name_count", "annotator_count", "names", "annotators"]
    ].sort_values(by=["name_count", "annotator_count"], ascending=False)


def run(args) -> int:
    df = pd.read_json(args.input_jsonl, lines=True)
    df = df.drop_duplicates().reset_index(drop=True)

    conflicts = find_conflicts(df)
    conflicts.to_csv(args.output_csv, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Conflicting files: {len(conflicts)}")
    print(f"Wrote conflict CSV: {args.output_csv}")
    return 0
