from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
            self.transform = transforms.Compose([default_transform, transform])


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
