from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image

from .augmenter import Augmenter, AugmentationType


def _resolve_augmentation_types(requested: Iterable[str]) -> list[AugmentationType]:
    normalized = {
        value.strip().lower()
        for value in requested
        if isinstance(value, str) and value.strip()
    }

    if not normalized or "none" in normalized:
        return []

    if "all" in normalized:
        return [
            AugmentationType.CropAug,
            AugmentationType.LowResAug,
            AugmentationType.PhotocentricAug,
        ]

    allowed = {
        AugmentationType.CropAug.value: AugmentationType.CropAug,
        AugmentationType.LowResAug.value: AugmentationType.LowResAug,
        AugmentationType.PhotocentricAug.value: AugmentationType.PhotocentricAug,
    }

    unknown = sorted(value for value in normalized if value not in allowed)
    if unknown:
        raise ValueError(
            "Unsupported augmentation type(s): "
            f"{unknown}. Allowed values are all, none, crop, low-res, photo."
        )

    return [allowed[value] for value in sorted(normalized)]


def _build_destination_face_path(face_path: Path, augmentation_type: AugmentationType) -> Path:
    return Path(augmentation_type.value) / face_path.with_name(
        f"{face_path.stem}__aug-{augmentation_type.value}{face_path.suffix}"
    )


def _validate_required_columns(dataset_df: pd.DataFrame) -> None:
    required_cols = {"face"}
    missing_cols = required_cols - set(dataset_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset JSONL: {sorted(missing_cols)}")


def make_augmentations(
    image_root: Path,
    input_jsonl: Path,
    destination_root: Path,
    augmentation_names: Iterable[str],
) -> int:
    selected_augmentations = _resolve_augmentation_types(augmentation_names)
    if not selected_augmentations:
        print("No augmentation selected. Nothing was generated.")
        return 0

    source_df = pd.read_json(input_jsonl, lines=True)
    _validate_required_columns(source_df)

    image_root = image_root.resolve()
    destination_root = destination_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    for augmentation_type in selected_augmentations:
        augmenter = Augmenter(augmentation_type)
        output_records: list[dict] = []

        for record in source_df.to_dict(orient="records"):
            source_face_rel = Path(str(record["face"]))
            source_image_path = image_root / source_face_rel
            if not source_image_path.exists():
                raise FileNotFoundError(f"Image not found: {source_image_path}")

            with Image.open(source_image_path) as source_image:
                augmented_image = augmenter.augment(source_image.convert("RGB"))

            output_face_rel = _build_destination_face_path(
                source_face_rel,
                augmentation_type,
            )
            output_image_path = destination_root / output_face_rel
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            augmented_image.save(output_image_path)

            updated_record = dict(record)
            updated_record["face"] = output_face_rel.as_posix()
            updated_record["augmentation"] = augmentation_type.value
            output_records.append(updated_record)

        output_df = pd.DataFrame(output_records)
        output_jsonl = destination_root / f"{input_jsonl.stem}.{augmentation_type.value}.jsonl"
        output_df.to_json(output_jsonl, orient="records", lines=True, force_ascii=True)
        print(
            f"Generated {len(output_records)} samples for '{augmentation_type.value}': {output_jsonl}"
        )

    return 0




def run(args) ->int:
    return make_augmentations(
        image_root=args.images_root,
        input_jsonl=args.input_jsonl,
        destination_root=args.destination_root,
        augmentation_names=args.augmentations,
    )