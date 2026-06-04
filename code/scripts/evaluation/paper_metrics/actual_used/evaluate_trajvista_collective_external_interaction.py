from __future__ import annotations

from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dataset_building import evaluate_trajvista_collective_loro as collective
from tools.dataset_building.evaluate_trajvista_per_metric_filtered_external_interaction import EXPERIMENTS, FINAL_MODELS


collective.EXPERIMENTS = EXPERIMENTS
collective.FINAL_MODELS = tuple(spec for spec in FINAL_MODELS if spec["model"] != "Localized")
collective.MODEL_ORDER = ("Ours", "GAN-TL", "FD-Align", "GT-MMD")


if __name__ == "__main__":
    collective.main()
