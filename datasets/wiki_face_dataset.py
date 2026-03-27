from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class WikiFaceDataset(Dataset):
    """
    Class used to represent the WikiFaceDataset
    """
    def __init__(
        self,
        csv_path: Path,
        images_root: Path,
        image_col:str="aligned_image",
        label_col:str="identity",
        transform=None,
    ):
        """
        Initialize the WikiFace dataset.

        Parameters
        ----------
        csv_path : Path
            Path to the CSV metadata file. The file must contain at least
            `image_col` and `label_col` columns.
        images_root : Path
            Root directory containing image files referenced by `image_col`.
            Relative paths from the CSV are resolved against this directory.
        image_col : str, optional
            Name of the CSV column containing image paths.
            Defaults to "aligned_image".
        label_col : str, optional
            Name of the CSV column containing identity labels.
            Defaults to "identity".
        transform : callable, optional
            Additional transform applied on top of default preprocessing.
            Default preprocessing converts images to tensors and normalizes
            them to [-1, 1].
        """
        self.df = pd.read_csv(csv_path.resolve(), sep=";")
        self.image_col = image_col
        self.label_col = label_col
        self.images_root = images_root.resolve()
        classes = sorted(self.df[self.label_col].unique().tolist())
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}
        default_transform = transforms.Compose(
            [
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

    def __getitem__(self, idx:int)->tuple[Image.Image,torch.Tensor]:
        row = self.df.iloc[idx]

        rel_path = Path(row[self.image_col])
        img_path = self.images_root / rel_path
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label_name = row[self.label_col]
        label_idx = self.class_to_idx[label_name]
        label = torch.tensor(label_idx, dtype=torch.long)

        return image, label
