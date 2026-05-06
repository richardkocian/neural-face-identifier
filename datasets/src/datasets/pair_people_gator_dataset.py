import numpy as np
import torch
from torch.utils.data import Dataset

from .people_gator_dataset import PeopleGatorDataset

class PeopleGatorPairDataset(Dataset):
    """Wraps PeopleGatorDataset to generate image pairs for validation."""
    
    def __init__(self, peoplegator_ds: PeopleGatorDataset, seed: int = 42):
        self.ds = peoplegator_ds
        self.class_to_indices = self._build_class_indices()

        self.seed = seed
        # Pre-generate the randomness so it's the same every epoch
        self.rng = np.random.default_rng(seed)

        # Create a fixed list of "is_match" and "pair_index" for the whole dataset
        self.fixed_pairs = []
        for i in range(len(self.ds)):
            is_match = self.rng.random() > 0.5
            self.fixed_pairs.append(is_match)
    
    def _build_class_indices(self):
        """Map class indices to sample indices."""
        class_indices = {}
        for idx in range(len(self.ds.df)):
            label_name = self.ds.df.iloc[idx][self.ds.label_col]
            label_idx = self.ds.class_to_idx[label_name]
            if label_idx not in class_indices:
                class_indices[label_idx] = []
            class_indices[label_idx].append(idx)
        return class_indices
    
    def __len__(self):
        return len(self.ds)
    
    # def __getitem__(self, idx):
    #     img1, label1 = self.ds[idx]
    #     label1_idx = label1.item()
        
    #     # 50% chance: match (same class) or non-match (different class)
    #     if torch.rand(1) > 0.5 and len(self.class_to_indices[label1_idx]) > 1:
    #         # Match: get another image from same class
    #         idx2 = self.class_to_indices[label1_idx][torch.randint(0, len(self.class_to_indices[label1_idx]), (1,)).item()]
    #         img2, _ = self.ds[idx2]
    #         match = torch.tensor(1, dtype=torch.long)
    #     else:
    #         # Non-match: get image from different class
    #         other_labels = [l for l in self.class_to_indices.keys() if l != label1_idx]
    #         other_label = other_labels[torch.randint(0, len(other_labels), (1,)).item()]
    #         idx2 = self.class_to_indices[other_label][torch.randint(0, len(self.class_to_indices[other_label]), (1,)).item()]
    #         img2, _ = self.ds[idx2]
    #         match = torch.tensor(0, dtype=torch.long)
        
    #     return img1, img2, match
    def __getitem__(self, idx):
        # Use a deterministic seed based on the index and global seed
        # This ensures image 10 always pairs with the same image X
        local_rng = np.random.default_rng(self.seed + idx)
        
        img1, label1 = self.ds[idx]
        label1_idx = label1.item()
        is_match = self.fixed_pairs[idx]

        if is_match and len(self.class_to_indices[label1_idx]) > 1:
            # Pick a different index from the same class
            idx2 = local_rng.choice(self.class_to_indices[label1_idx])
            match = torch.tensor(1, dtype=torch.long)
        else:
            # Pick a random index from a different class
            other_labels = [l for l in self.class_to_indices.keys() if l != label1_idx]
            other_label = local_rng.choice(other_labels)
            idx2 = local_rng.choice(self.class_to_indices[other_label])
            match = torch.tensor(0, dtype=torch.long)

        img2, _ = self.ds[idx2]
        return img1, img2, match