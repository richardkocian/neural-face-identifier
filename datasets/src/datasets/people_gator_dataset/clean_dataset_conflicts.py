import ast
import logging
from pathlib import Path

import pandas as pd

UNUSED_COLS: list[str] = [
    "library",
    "document",
    "page",
    "crop_name",
    "page_width",
    "page_height",
    "page_left",
    "page_top",
    "width",
    "height",
    "page_keypoints",
]


def _resolve_preferred_name_column(conflict_df: pd.DataFrame) -> str | None:
    """Return the preferred-name column name used in the conflicts CSV.

    Args:
        conflict_df: Dataframe loaded from the conflicts CSV.

    Returns:
        Name of the preferred-name column if present, otherwise None.
    """
    return "prefered_name" if "prefered_name" in conflict_df.columns else None


def _build_alias_mapping(
    conflict_df: pd.DataFrame,
    preferred_col: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str], dict[str, int]]:
    """Build alias and face-level preferred-name mappings from conflict rows.

    Args:
        conflict_df: Dataframe loaded from the conflicts CSV.
        preferred_col: Column name containing preferred identity labels.

    Returns:
        alias_to_preferred: Global alias -> preferred name mapping.
        alias_to_line: Source CSV line for each alias mapping.
        face_to_preferred: Face path -> preferred name mapping.
        face_to_line: Source CSV line for each face mapping.

    Raises:
        ValueError: If names format is invalid or mappings are inconsistent.
    """
    alias_to_preferred: dict[str, str] = {}
    alias_to_line: dict[str, int] = {}
    face_to_preferred: dict[str, str] = {}
    face_to_line: dict[str, int] = {}
    has_rename_col = "rename" in conflict_df.columns

    def should_apply_global_alias(row: pd.Series) -> bool:
        """Return True when this row should contribute to global alias mapping.

        Args:
            row: One row from the conflicts dataframe.

        Returns:
            True if the row should be used for global alias mapping.
        """
        if not has_rename_col:
            return False
        raw_flag = row.get("rename")
        if pd.isna(raw_flag):
            return False
        return True

    for row_idx, conflict_row in conflict_df.iterrows():
        csv_line = row_idx + 2
        face = str(conflict_row.get("face", "")).strip()
        raw_preferred = conflict_row.get(preferred_col)
        preferred = "" if pd.isna(raw_preferred) else str(raw_preferred).strip()
        if preferred == "":
            continue

        raw_names = conflict_row.get("names")
        try:
            parsed_names = (
                ast.literal_eval(raw_names) if isinstance(raw_names, str) else []
            )
        except (ValueError, SyntaxError) as exc:
            face = conflict_row.get("face", "<unknown-face>")
            raise ValueError(
                f"Invalid 'names' format at line {csv_line} for face '{face}'."
            ) from exc

        options = {
            str(name).strip() for name in parsed_names if str(name).strip() != ""
        }
        if preferred not in options:
            face_display = conflict_row.get("face", "<unknown-face>")
            raise ValueError(
                "Preferred name does not match options in conflicts CSV at "
                f"line {csv_line} for face '{face_display}'. "
                f"preferred='{preferred}', options={sorted(options)}"
            )

        existing_face_preferred = face_to_preferred.get(face)
        if existing_face_preferred is not None and existing_face_preferred != preferred:
            raise ValueError(
                "Conflicting preferred mappings for face "
                f"'{face}': '{existing_face_preferred}' (line {face_to_line[face]}) "
                f"vs '{preferred}' (line {csv_line})"
            )
        face_to_preferred[face] = preferred
        face_to_line[face] = csv_line

        if not should_apply_global_alias(conflict_row):
            continue

        for alias in options:
            if alias == preferred:
                continue

            existing_preferred = alias_to_preferred.get(alias)
            if existing_preferred is not None and existing_preferred != preferred:
                raise ValueError(
                    "Conflicting preferred mappings for alias "
                    f"'{alias}': '{existing_preferred}' (line {alias_to_line[alias]}) "
                    f"vs '{preferred}' (line {csv_line})"
                )

            alias_to_preferred[alias] = preferred
            alias_to_line[alias] = csv_line

    return alias_to_preferred, alias_to_line, face_to_preferred, face_to_line


def clean_dataset(
    input_jsonl_path: Path,
    conflicts_csv_path: Path,
    output_jsonl_path: Path,
    report_csv_path: Path,
    log_path: Path,
) -> None:
    """Clean dataset labels using conflict rules and write outputs.

    The function applies alias/conflict renaming rules, removes duplicates
    while ignoring annotator, writes cleaned JSONL, and writes a cleanup report.

    Args:
        input_jsonl_path: Path to source dataset JSONL.
        conflicts_csv_path: Path to conflicts CSV containing preferred names.
        output_jsonl_path: Path where cleaned JSONL will be written.
        report_csv_path: Path where cleanup event report CSV will be written.
        log_path: Path where execution log file will be written.

    Returns:
        None.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    logging.info("Starting cleanup")
    logging.info("Source JSONL: %s", input_jsonl_path)
    logging.info("Conflicts CSV: %s", conflicts_csv_path)

    source_df = pd.read_json(input_jsonl_path, lines=True)
    logging.info("Loaded source rows: %d", len(source_df))

    source_df = source_df.drop(columns=UNUSED_COLS, errors="ignore")
    logging.info(
        "Dropped unused columns. Remaining columns: %s", list(source_df.columns)
    )

    conflict_df = pd.read_csv(conflicts_csv_path)
    logging.info("Loaded conflict rows: %d", len(conflict_df))

    preferred_col = _resolve_preferred_name_column(conflict_df)
    if preferred_col is None:
        raise ValueError(
            "Could not resolve preferred-name column. Expected exact column 'prefered_name'."
        )

    global_alias_mode = "rename" in conflict_df.columns

    logging.info("Using preferred-name column: %s", preferred_col)
    if global_alias_mode:
        logging.info("'rename' column found: global alias mode is per-row.")
    else:
        logging.info("'rename' column not found: running in FACE-ONLY CONFLICT mode.")

    alias_to_preferred, alias_to_line, face_to_preferred, face_to_line = (
        _build_alias_mapping(conflict_df, preferred_col)
    )
    logging.info("Loaded alias mappings: %d", len(alias_to_preferred))
    logging.info("Loaded face preferred mappings: %d", len(face_to_preferred))

    cleanup_events: list[dict[str, str | int]] = []

    source_df["person_name"] = source_df["person_name"].astype("string")
    source_df["face"] = source_df["face"].astype("string")
    alias_resolution_count = 0
    conflict_resolution_count = 0

    if global_alias_mode and alias_to_preferred:
        stripped_names = source_df["person_name"].str.strip()
        alias_replacements = stripped_names.map(alias_to_preferred)
        alias_mask = alias_replacements.notna()
        alias_resolution_count = int(alias_mask.sum())
        renamed_alias_rows = source_df[alias_mask].copy()
        for _, row in renamed_alias_rows.iterrows():
            old_name = str(row.get("person_name", "")).strip()
            preferred_name = str(alias_to_preferred.get(old_name, ""))
            cleanup_events.append(
                {
                    "face": str(row.get("face", "")),
                    "person_name": str(row.get("person_name", "")),
                    "annotator": str(row.get("annotator", "")),
                    "preferred_name": preferred_name,
                    "action": "renamed_alias_global",
                    "reason": "Name globally normalized from conflict aliases table.",
                    "conflict_csv_line": (
                        int(alias_to_line[old_name])
                        if old_name in alias_to_line
                        else ""
                    ),
                }
            )
        source_df.loc[alias_mask, "person_name"] = alias_replacements[alias_mask]

    if face_to_preferred:
        preferred_for_face = source_df["face"].map(face_to_preferred)
        conflict_mask = preferred_for_face.notna() & (
            source_df["person_name"].str.strip()
            != preferred_for_face.astype("string").str.strip()
        )
        conflict_resolution_count = int(conflict_mask.sum())
        renamed_conflict_rows = source_df[conflict_mask].copy()
        for _, row in renamed_conflict_rows.iterrows():
            face = str(row.get("face", ""))
            preferred_name = str(face_to_preferred.get(face, ""))
            cleanup_events.append(
                {
                    "face": face,
                    "person_name": str(row.get("person_name", "")),
                    "annotator": str(row.get("annotator", "")),
                    "preferred_name": preferred_name,
                    "action": "renamed_conflict_face",
                    "reason": "Name replaced by preferred name for matching conflict face.",
                    "conflict_csv_line": (
                        int(face_to_line[face]) if face in face_to_line else ""
                    ),
                }
            )
        source_df.loc[conflict_mask, "person_name"] = preferred_for_face[conflict_mask]

    logging.info("Alias resolutions before dedup: %d", alias_resolution_count)
    logging.info("Conflict resolutions before dedup: %d", conflict_resolution_count)

    dedup_subset = [column for column in source_df.columns if column != "annotator"]
    duplicate_mask = source_df.duplicated(subset=dedup_subset, keep="first")
    duplicate_rows = source_df[duplicate_mask].copy()
    logging.info(
        "Rows removed as duplicates (ignoring annotator): %d", len(duplicate_rows)
    )
    for _, row in duplicate_rows.iterrows():
        cleanup_events.append(
            {
                "face": str(row.get("face", "")),
                "person_name": str(row.get("person_name", "")),
                "annotator": str(row.get("annotator", "")),
                "preferred_name": "",
                "action": "removed_duplicate",
                "reason": "Duplicate row when comparing all fields except annotator.",
                "conflict_csv_line": "",
            }
        )

    source_df = source_df[~duplicate_mask].reset_index(drop=True)

    source_df.to_json(output_jsonl_path, orient="records", lines=True, force_ascii=True)
    pd.DataFrame(cleanup_events).to_csv(report_csv_path, index=False)

    logging.info("Wrote cleaned JSONL: %s", output_jsonl_path)
    logging.info("Final cleaned rows: %d", len(source_df))
    logging.info("Wrote cleanup report CSV: %s", report_csv_path)
    logging.info("Cleanup events recorded: %d", len(cleanup_events))
    logging.info("Done")


def run(args) -> int:
    """Run the cleanup script from CLI arguments."""
    clean_dataset(
        input_jsonl_path=args.input_jsonl,
        conflicts_csv_path=args.conflicts_csv,
        output_jsonl_path=args.output_jsonl,
        report_csv_path=args.report_csv,
        log_path=args.log_file,
    )
    return 0
