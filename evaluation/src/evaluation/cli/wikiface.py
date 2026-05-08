from __future__ import annotations

from evaluation.wikiface.checkpoint_automation import main as checkpoint_automation_main
from evaluation.wikiface.checkpoint_metrics_summary_pdf import (
    main as checkpoint_metrics_summary_pdf_main,
)
from evaluation.wikiface.evaluate import main as wikiface_main


def main() -> int:
    return wikiface_main()


def main_retrieval_checkpoints() -> int:
    return checkpoint_automation_main()


def main_retrieval_metrics_summary_pdf() -> int:
    return checkpoint_metrics_summary_pdf_main()

