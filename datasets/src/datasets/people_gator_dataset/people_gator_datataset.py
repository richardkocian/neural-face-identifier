from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .augmenter import AugmentationType


class PeopleGatorDataset(Dataset):
    """PyTorch dataset for PeopleGator JSONL metadata."""

    def __init__(
        self,
        jsonl_path: Path,
        images_root: Path,
        transform=None,
    ):
        """
        Parameters
        ----------
        jsonl_path : Path
            Path to JSONL metadata file.
        images_root : Path
            Root directory where image files are stored.
        transform : callable, optional
            Additional transform applied on top of default preprocessing.
            Default preprocessing is resize to 112x112, tensor conversion,
            and normalization.
        """
        self.jsonl_path = Path(jsonl_path).resolve()
        self.images_root = Path(images_root).resolve()
        self.image_col = "face"
        self.label_col = "person_name"

        self.df = pd.read_json(self.jsonl_path, lines=True)

        classes = sorted(self.df[self.label_col].dropna().unique().tolist())
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        default_transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        if transform is None:
            self.transform = default_transform
        else:
            # self.transform = transforms.Compose([default_transform, transform])
            self.transform = transforms.Compose([transform, self.base_transform])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        rel_path = Path(str(row[self.image_col]))
        img_path = self.images_root / rel_path
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for sample {idx}: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label_name = row[self.label_col]
        label_idx = self.class_to_idx[label_name]
        label = torch.tensor(label_idx, dtype=torch.long)

        return image, label


class AugmentedPeopleGatorDataset(PeopleGatorDataset):
    """PeopleGator dataset with optional augmentation-based filtering."""
    
    def __init__(
        self,
        jsonl_path: Path,
        images_root: Path,
        transform=None,
        augmentation_filter: AugmentationType | None = None,
    ):
        super().__init__(jsonl_path=jsonl_path, images_root=images_root, transform=transform)
        self.augmentation_filter = augmentation_filter

        self._validate_augmentation_values()
        self._apply_augmentation_filter()
        self._rebuild_class_mappings()

    def _validate_augmentation_values(self) -> None:
        """Ensure augmentation column only contains values from AugmentationType."""
        if "augmentation" not in self.df.columns:
            return

        allowed_values = {augmentation.value for augmentation in AugmentationType}
        present_values = {
            value.strip().lower()
            for value in self.df["augmentation"].dropna().astype("string")
            if value.strip() != ""
        }
        invalid_values = sorted(value for value in present_values if value not in allowed_values)
        if invalid_values:
            raise ValueError(
                "Invalid values in 'augmentation' column: "
                f"{invalid_values}. Allowed values: {sorted(allowed_values)}"
            )

    def _apply_augmentation_filter(self) -> None:
        """Filter dataset by augmentation value when requested and column exists."""
        if self.augmentation_filter is None:
            return

        if "augmentation" not in self.df.columns:
            # If column is absent, treat every row as implicit 'none'.
            if self.augmentation_filter == AugmentationType.NoAug:
                return
            self.df = self.df.iloc[0:0].copy()
            return

        normalized_filter = self.augmentation_filter.value.strip().lower()
        matches = (
            self.df["augmentation"]
            .astype("string")
            .fillna(AugmentationType.NoAug.value)
            .str.strip()
            .str.lower()
            .eq(normalized_filter)
        )
        self.df = self.df[matches].reset_index(drop=True)

    def _rebuild_class_mappings(self) -> None:
        classes = sorted(self.df[self.label_col].dropna().unique().tolist())
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
