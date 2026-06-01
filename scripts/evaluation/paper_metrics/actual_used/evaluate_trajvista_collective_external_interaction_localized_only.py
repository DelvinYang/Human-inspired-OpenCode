from __future__ import annotations

from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dataset_building import evaluate_trajvista_collective_loro_localized_only as localized
from tools.dataset_building.evaluate_trajvista_per_metric_filtered_external_interaction import EXPERIMENTS


localized.EXPERIMENTS = EXPERIMENTS
localized.LOCALIZED_CHECKPOINT = "runs/revision_external_interaction_transfer/localized/{experiment}/f005/transfer/best_model.pt"


if __name__ == "__main__":
    localized.main()
