from __future__ import annotations

from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dataset_building import evaluate_trajvista_complete_trajectory_rollout as complete
from tools.dataset_building import evaluate_trajvista_individual_loro_rollout as rollout
from tools.dataset_building.evaluate_trajvista_per_metric_filtered_external_interaction import EXPERIMENTS, FINAL_MODELS


EXTERNAL_EXPERIMENTS = [
    {"experiment": experiment, "target_datasets": exp["target_datasets"]}
    for experiment, exp in EXPERIMENTS.items()
]

EXTERNAL_MODEL_SPECS = [
    {
        "model_name": spec["model"],
        "model_type": spec["model_type"],
        "checkpoint": spec["checkpoint"],
    }
    for spec in FINAL_MODELS
]

complete.EXPERIMENTS = EXTERNAL_EXPERIMENTS
complete.MODEL_SPECS = EXTERNAL_MODEL_SPECS
rollout.EXPERIMENTS = EXTERNAL_EXPERIMENTS


if __name__ == "__main__":
    rollout.main()
