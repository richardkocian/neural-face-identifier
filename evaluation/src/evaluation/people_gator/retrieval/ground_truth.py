from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build retrieval ground-truth JSONL expected by peoplegator_namedfaces.retrieval.evaluate."
        )
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("evaluation/src/peoplegator_namedfaces/retrieval/configs/image_queries.union.tst.jsonl"),
        help="Input query JSONL with fields: query, query_type.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("people_gator/people_gator__corresponding_faces__2026-02-11.test.cleaned.jsonl"),
        help="PeopleGator cleaned annotation JSONL with fields: face, person_name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_artifacts/retrieval.union.tst.ground_truth.jsonl"),
        help="Output ground-truth JSONL path.",
    )
    parser.add_argument(
        "--allow-missing-queries",
        action="store_true",
        help="Skip queries missing in annotations instead of failing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    face_to_name: dict[str, str] = {}
    name_to_faces: dict[str, set[str]] = defaultdict(set)

    with args.annotations.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            row = json.loads(line)
            try:
                face = str(row["face"])
                person_name = str(row["person_name"])
            except KeyError as exc:
                raise KeyError(f"Missing key in annotations line {line_num}: {exc}") from exc

            face_to_name[face] = person_name
            name_to_faces[person_name].add(face)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.queries.open("r", encoding="utf-8") as f_in, args.output.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line_num, line in enumerate(f_in, start=1):
            query_row = json.loads(line)
            query = str(query_row["query"])
            query_type = str(query_row["query_type"])

            person_name = face_to_name.get(query)
            if person_name is None:
                if args.allow_missing_queries:
                    skipped += 1
                    continue
                raise KeyError(
                    f"Query '{query}' from {args.queries} line {line_num} not found in annotations {args.annotations}."
                )

            record = {
                "query": query,
                "query_type": query_type,
                "faces": sorted(name_to_faces[person_name]),
            }
            f_out.write(json.dumps(record) + "\n")
            written += 1

    print("Retrieval ground truth build finished.")
    print(f"Queries input: {args.queries}")
    print(f"Annotations input: {args.annotations}")
    print(f"Written records: {written}")
    print(f"Skipped missing queries: {skipped}")
    print(f"Output: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

