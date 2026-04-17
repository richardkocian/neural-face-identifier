from __future__ import annotations

import statistics

import torch

MisclassifiedItem = dict[str, int | float]


def _gallery_query_indices(labels: torch.Tensor) -> tuple[list[int], list[int]]:
    unique_labels = labels.unique(sorted=True)
    gallery_idx: list[int] = []
    query_idx: list[int] = []

    for label_value in unique_labels.tolist():
        cls_idx = torch.nonzero(labels == label_value, as_tuple=False).flatten()
        if cls_idx.numel() < 2:
            continue
        gallery_idx.append(int(cls_idx[0].item()))
        query_idx.extend(int(x.item()) for x in cls_idx[1:])

    if not query_idx:
        raise ValueError("Need at least one identity with >=2 images to build query samples.")
    return gallery_idx, query_idx


def gallery_query_pair_labels_scores(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gallery_idx, query_idx = _gallery_query_indices(labels=labels)
    gallery_emb = embeddings[gallery_idx]
    gallery_labels = labels[gallery_idx]
    query_emb = embeddings[query_idx]
    query_labels = labels[query_idx]

    sims = query_emb @ gallery_emb.T
    y_true = (query_labels.unsqueeze(1) == gallery_labels.unsqueeze(0)).to(torch.int8).reshape(-1)
    y_score = sims.reshape(-1)
    return y_true, y_score


def gallery_query_topk(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    ks: tuple[int, ...] = (1, 5),
) -> tuple[dict[int, float], list[MisclassifiedItem], list[float], list[float]]:
    gallery_idx, query_idx = _gallery_query_indices(labels=labels)

    gallery_emb = embeddings[gallery_idx]
    gallery_labels = labels[gallery_idx]
    query_emb = embeddings[query_idx]
    query_labels = labels[query_idx]

    sims = query_emb @ gallery_emb.T
    max_k = min(max(ks), sims.shape[1])
    topk_indices = sims.topk(k=max_k, dim=1).indices
    topk_labels = gallery_labels[topk_indices]

    metrics: dict[int, float] = {}
    for k in tuple(ks):
        k_eff = min(k, topk_labels.shape[1])
        hits = (topk_labels[:, :k_eff] == query_labels.unsqueeze(1)).any(dim=1)
        metrics[k] = float(hits.float().mean().item())

    label_to_gallery_embedding_idx = {
        int(label.item()): gallery_idx[pos] for pos, label in enumerate(gallery_labels)
    }

    top1_gallery_idx = topk_indices[:, 0]
    top1_pred_labels = gallery_labels[top1_gallery_idx]
    top1_scores = sims.gather(1, top1_gallery_idx.unsqueeze(1)).squeeze(1)
    top1_correct_mask = top1_pred_labels == query_labels
    correct_scores = [float(x) for x in top1_scores[top1_correct_mask].tolist()]
    wrong_scores = [float(x) for x in top1_scores[~top1_correct_mask].tolist()]

    wrong_query_local_idx = torch.nonzero(top1_pred_labels != query_labels, as_tuple=False).flatten()

    misclassified: list[MisclassifiedItem] = []
    for wrong_pos in wrong_query_local_idx:
        wrong_idx = int(wrong_pos.item())
        gallery_local_idx = int(top1_gallery_idx[wrong_idx].item())
        query_embedding_idx = query_idx[wrong_idx]
        gallery_embedding_idx = int(gallery_idx[gallery_local_idx])
        true_label = int(query_labels[wrong_idx].item())
        true_gallery_embedding_idx = int(label_to_gallery_embedding_idx[true_label])
        misclassified.append(
            {
                "query_embedding_idx": query_embedding_idx,
                "gallery_embedding_idx": gallery_embedding_idx,
                "true_gallery_embedding_idx": true_gallery_embedding_idx,
                "true_label": true_label,
                "pred_label": int(top1_pred_labels[wrong_idx].item()),
                "score": float(sims[wrong_idx, gallery_local_idx].item()),
            }
        )
    return metrics, misclassified, correct_scores, wrong_scores


def describe_scores(scores: list[float]) -> tuple[int, float, float]:
    if not scores:
        return 0, float("nan"), float("nan")
    return len(scores), float(statistics.mean(scores)), float(statistics.median(scores))


def first_quartile(scores: list[float]) -> float:
    if not scores:
        return float("nan")
    if len(scores) == 1:
        return float(scores[0])
    quartiles = statistics.quantiles(scores, n=4, method="inclusive")
    return float(quartiles[0])

