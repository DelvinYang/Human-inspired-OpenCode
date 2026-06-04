from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import torch

from cultural_align.models.trajvista import FEATURE_DIM, HIDDEN_DIM, make_model
from cultural_align.training.data import NormStats


CODE_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = CODE_ROOT / "pretrained" / "models" / "ind_5k_best"


def iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_strings(key)
            yield from iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_strings(item)


class IndPretrainedReleaseTest(unittest.TestCase):
    def test_checkpoint_loads_with_released_model(self) -> None:
        payload = torch.load(CHECKPOINT_ROOT / "best_model.pt", map_location="cpu", weights_only=True)
        self.assertEqual(sorted(payload.keys()), ["args", "model_state_dict", "release_metadata", "stats"])

        metadata = json.loads((CHECKPOINT_ROOT / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["dataset"], "inD")
        self.assertEqual(metadata["model_dimensions"], {"hidden_dim": HIDDEN_DIM, "feature_dim": FEATURE_DIM})

        stats = NormStats(**payload["stats"])
        model = make_model(stats, input_dim=12, dropout=payload["args"]["dropout"])
        report = model.load_state_dict(payload["model_state_dict"], strict=False)
        self.assertEqual(list(report.missing_keys), [])
        self.assertEqual(list(report.unexpected_keys), [])

    def test_checkpoint_metadata_has_no_local_paths(self) -> None:
        payload = torch.load(CHECKPOINT_ROOT / "best_model.pt", map_location="cpu", weights_only=True)
        metadata = json.loads((CHECKPOINT_ROOT / "metadata.json").read_text(encoding="utf-8"))
        blocked = ("/" + "Volumes/", "/" + "Users/", "/" + "home/", "crossculture_" + "second_round")
        for text in list(iter_strings(payload)) + list(iter_strings(metadata)):
            self.assertFalse(any(needle in text for needle in blocked), text)


if __name__ == "__main__":
    unittest.main()
