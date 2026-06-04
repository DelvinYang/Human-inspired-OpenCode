from __future__ import annotations

from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dataset_building import evaluate_trajvista_complete_trajectory_rollout as complete
from tools.dataset_building.evaluate_trajvista_individual_loro_rollout import main


LOCALIZED_SPEC = {
    "model_name": "Localized",
    "model_type": "localized",
    "checkpoint": "runs/baselines/localized_loro_target_only/{experiment}/f005/transfer/best_model.pt",
}


if not any(spec["model_name"] == "Localized" for spec in complete.MODEL_SPECS):
    complete.MODEL_SPECS.append(LOCALIZED_SPEC)


if __name__ == "__main__":
    main()
