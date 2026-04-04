from __future__ import annotations

import timm
import torch

from datasets.wiki_face_dataset import WikiFaceDataset

from .artifacts import save_top1_misclassified_previews, save_top1_score_boxplot
from .cli import build_parser
from .embeddings import extract_embeddings
from .metrics import describe_scores, first_quartile, gallery_query_topk


def main() -> int:
    args = build_parser().parse_args()

    # Dataset returns (image_tensor, class_index) where image tensor is normalized
    # to [-1, 1] in the dataset preprocessing pipeline.
    dataset = WikiFaceDataset(csv_path=args.csv_path, images_root=args.images_root)

    if len(dataset) == 0:
        print("Dataset is empty, nothing to evaluate.")
        return 1

    device = torch.device(args.device)
    # num_classes=0 removes the classification head and returns embedding vectors.
    model = timm.create_model(args.model_id, pretrained=True, num_classes=0).to(device)
    model.eval()

    embeddings, labels, sample_indices = extract_embeddings(
        dataset=dataset,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    metrics, misclassified, correct_scores, wrong_scores = gallery_query_topk(
        embeddings, labels, ks=(1, 5)
    )

    boxplot_path = save_top1_score_boxplot(
        correct_scores=correct_scores,
        wrong_scores=wrong_scores,
        output_path=args.top1_boxplot_path,
    )

    saved_misses = 0

    print("WikiFace evaluation finished.")
    print(f"CSV: {args.csv_path.resolve()}")
    print(f"Images root: {args.images_root.resolve()}")
    print(f"Samples used: {embeddings.shape[0]}")
    print(f"Classes in samples: {labels.unique().numel()}")
    print(f"Device: {device}")
    print(f"Model: {args.model_id}")
    print(f"Top-1 (gallery/query): {metrics[1] * 100:.2f}%")
    print(f"Top-5 (gallery/query): {metrics[5] * 100:.2f}%")

    correct_n, correct_mean, correct_median = describe_scores(correct_scores)
    wrong_n, wrong_mean, wrong_median = describe_scores(wrong_scores)
    correct_q1 = first_quartile(correct_scores)
    high_score_misclassified = [
        item for item in misclassified if float(item["score"]) > correct_q1
    ]

    if args.show_misclassified_top1 > 0:
        saved_misses = save_top1_misclassified_previews(
            dataset=dataset,
            sample_indices=sample_indices,
            misclassified=high_score_misclassified,
            limit=args.show_misclassified_top1,
            output_dir=args.misclassified_output_dir,
        )

    print(
        "Top-1 cosine (correct) -> "
        f"count: {correct_n}, mean: {correct_mean:.4f}, median: {correct_median:.4f}"
    )
    print(f"Top-1 cosine (correct) -> Q1: {correct_q1:.4f}")
    print(
        "Top-1 cosine (wrong) -> "
        f"count: {wrong_n}, mean: {wrong_mean:.4f}, median: {wrong_median:.4f}"
    )
    print(
        "Top-1 wrong with score > Q1(correct) -> "
        f"count: {len(high_score_misclassified)}"
    )
    print(f"Saved Top-1 cosine boxplot: {boxplot_path}")

    if args.show_misclassified_top1 > 0:
        print(
            f"Saved Top-1 misclassified previews (score > Q1(correct)): {saved_misses} -> "
            f"{args.misclassified_output_dir.resolve()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

