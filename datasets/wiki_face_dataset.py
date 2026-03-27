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


def make_dataset_split(csv_path: Path, seed: int, dataset_split: float = 80.0) -> None:
    """
    Split a WikiFace CSV into train and test CSVs with non-overlapping identities.

    The split is identity-based, meaning all images of one identity are placed
    entirely in either train or test. Output files are written next to the input
    file as `<stem>.train.csv` and `<stem>.test.csv`.

    Parameters
    ----------
    csv_path : Path
        Path to the source CSV file.
    seed : int
        Random seed used for deterministic identity shuffling.
    dataset_split : float, optional
        Percentage of identities assigned to train split. Defaults to 80.0.
        Values in (0, 1] are interpreted as a ratio and converted to percent.
    """
    csv_path = csv_path.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if dataset_split <= 0:
        raise ValueError("dataset_split must be > 0")

    # Accept both ratio style (0.8) and percent style (80.0).
    if dataset_split <= 1.0:
        dataset_split *= 100.0

    if dataset_split >= 100:
        raise ValueError("dataset_split must be < 100")

    df = pd.read_csv(csv_path, sep=";")
    if "identity" not in df.columns:
        raise ValueError("CSV must contain an 'identity' column")

    identities = sorted(df["identity"].dropna().unique().tolist())
    if len(identities) < 2:
        raise ValueError("Need at least 2 identities to create train/test split")

    shuffled_identities = (
        pd.Series(identities)
        .sample(frac=1.0, random_state=seed)
        .tolist()
    )

    train_count = int(len(shuffled_identities) * (dataset_split / 100.0))
    train_count = max(1, min(train_count, len(shuffled_identities) - 1))

    train_identities = set(shuffled_identities[:train_count])
    test_identities = set(shuffled_identities[train_count:])

    train_df = df[df["identity"].isin(train_identities)].reset_index(drop=True)
    test_df = df[df["identity"].isin(test_identities)].reset_index(drop=True)

    train_csv = csv_path.with_name(f"{csv_path.stem}.train.csv")
    test_csv = csv_path.with_name(f"{csv_path.stem}.test.csv")
    train_df.to_csv(train_csv, sep=";", index=False)
    test_df.to_csv(test_csv, sep=";", index=False)

