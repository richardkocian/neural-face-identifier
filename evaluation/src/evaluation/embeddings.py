from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.wiki_face_dataset import WikiFaceDataset


def resolve_dataset_indices(dataset: WikiFaceDataset | Subset) -> list[int]:
    if isinstance(dataset, Subset):
        parent_indices = resolve_dataset_indices(dataset.dataset)
        return [parent_indices[int(i)] for i in dataset.indices]
    return list(range(len(dataset)))


def extract_embeddings(
    dataset: WikiFaceDataset | Subset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    sample_indices = resolve_dataset_indices(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    emb_list: list[torch.Tensor] = []
    label_list: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            embs = model(images)
            embs = F.normalize(embs, dim=1)
            emb_list.append(embs.cpu())
            label_list.append(labels.cpu())

    if not emb_list:
        raise ValueError("Dataset is empty, no embeddings were generated.")

    return torch.cat(emb_list, dim=0), torch.cat(label_list, dim=0), sample_indices

