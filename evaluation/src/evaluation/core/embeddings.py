from __future__ import annotations

from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


class EmbeddingDatasetProtocol(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...


def resolve_dataset_indices(dataset: EmbeddingDatasetProtocol | Subset) -> list[int]:
    if isinstance(dataset, Subset):
        parent_indices = resolve_dataset_indices(dataset.dataset)
        return [parent_indices[int(i)] for i in dataset.indices]
    return list(range(len(dataset)))


def extract_embeddings(
    dataset: EmbeddingDatasetProtocol | Subset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if max_samples > 0:
        dataset = Subset(cast(Any, dataset), range(min(max_samples, len(dataset))))

    sample_indices = resolve_dataset_indices(dataset)

    dataloader = DataLoader(
        cast(Any, dataset),
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

