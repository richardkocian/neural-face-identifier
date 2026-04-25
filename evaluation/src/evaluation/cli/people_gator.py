from __future__ import annotations

from evaluation.people_gator.checkpoint_automation import main as checkpoint_automation_main
from evaluation.people_gator.embeddings import main as embeddings_main
from evaluation.people_gator.retrieval.boxplot import main as retrieval_boxplot_main
from evaluation.people_gator.retrieval.det_curve import main as retrieval_det_main
from evaluation.people_gator.retrieval.ground_truth import main as retrieval_ground_truth_main


def main_embeddings() -> int:
    return embeddings_main()


def main_retrieval_gt() -> int:
    return retrieval_ground_truth_main()


def main_retrieval_boxplot() -> int:
    return retrieval_boxplot_main()


def main_retrieval_det() -> int:
    return retrieval_det_main()


def main_retrieval_checkpoints() -> int:
    return checkpoint_automation_main()
