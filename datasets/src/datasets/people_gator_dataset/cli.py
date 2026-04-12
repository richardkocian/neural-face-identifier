import argparse
import sys
from pathlib import Path

from .clean_dataset_conflicts import run as run_clean_dataset_conflicts
from .find_dataset_conflicts import run as run_find_dataset_conflicts


SCRIPT_DEFAULT_FIELDS: dict[str, tuple[str, ...]] = {
    "find-conflicts": ("output_csv",),
    "clean-from-conflicts": (
        "conflicts_csv",
        "output_jsonl",
        "report_csv",
        "log_file",
    ),
}


def _derive_default_paths(input_jsonl: Path) -> dict[str, Path]:
    """Build default output paths from input JSONL path and stem.

    Args:
        input_jsonl: Source PeopleGator JSONL path.

    Returns:
        Mapping of output argument names to derived default paths.
    """
    base = (input_jsonl.parent / input_jsonl.stem).resolve()
    return {
        "output_csv": Path(f"{base}_conflicts.csv"),
        "conflicts_csv": Path(f"{base}_conflicts.csv"),
        "output_jsonl": Path(f"{base}.cleaned.jsonl"),
        "report_csv": Path(f"{base}_clean_report.csv"),
        "log_file": Path(f"{base}_clean_log.log"),
    }


def _apply_script_defaults(args: argparse.Namespace, defaults: dict[str, Path]) -> None:
    """Apply derived defaults to missing args for the selected script.

    Args:
        args: Parsed CLI arguments.
        defaults: Derived default path values.
    """
    for field_name in SCRIPT_DEFAULT_FIELDS.get(args.script_name, ()):
        if getattr(args, field_name) is None:
            setattr(args, field_name, defaults[field_name])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for people-gator script delegation.

    Returns:
        Parsed argument namespace including selected script and script arguments.
    """
    # ----------------------------#
    #        common args         #
    # ----------------------------#
    parser = argparse.ArgumentParser(
        prog="people-gator",
        description="Run PeopleGator tooling scripts through one unified CLI.",
    )
    subparsers = parser.add_subparsers(dest="script_name", required=True)

    shared_subparser_args = argparse.ArgumentParser(add_help=False)
    shared_subparser_args.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Path to source PeopleGator JSONL used by commands.",
    )

    # ----------------------------#
    #     find conflicts args    #
    # ----------------------------#

    find_parser = subparsers.add_parser(
        "find-conflicts",
        help="Find conflicting identities in a PeopleGator JSONL file.",
        parents=[shared_subparser_args],
    )
    find_parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path where the conflicts CSV report will be written (default: {input-stem}_conflicts.csv).",
    )

    # ----------------------------#
    #      clean dataset args    #
    # ----------------------------#

    clean_parser = subparsers.add_parser(
        "clean-from-conflicts",
        help="Clean PeopleGator JSONL using a conflicts CSV.",
        parents=[shared_subparser_args],
    )

    clean_parser.add_argument(
        "--conflicts-csv",
        type=Path,
        default=None,
        help="Path to conflicts CSV containing preferred-name decisions (default: {input-stem}_conflicts.csv).",
    )
    clean_parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Path where cleaned JSONL will be written (default: {input-stem}.cleaned.jsonl).",
    )

    clean_parser.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Path where cleanup report CSV will be written (default: {input-stem}_clean_report.csv).",
    )

    clean_parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to cleanup log file (default: {input-stem}_clean_log.log).",
    )

    args = parser.parse_args()
    defaults = _derive_default_paths(args.input_jsonl)
    _apply_script_defaults(args, defaults)

    return args


def get_script_from_args(args: argparse.Namespace):
    """Map parsed args to the concrete script runner.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Callable script runner function.

    Raises:
        ValueError: If script_name is not recognized.
    """
    if args.script_name == "find-conflicts":
        return run_find_dataset_conflicts
    if args.script_name == "clean-from-conflicts":
        return run_clean_dataset_conflicts
    raise ValueError(f"Unknown script_name: {args.script_name}")


def main() -> int:
    """CLI entrypoint that parses args and delegates to selected script.

    Returns:
        Integer process exit code.
    """
    try:
        args = parse_args()
        script = get_script_from_args(args)
        return script(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
