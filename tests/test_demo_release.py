from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from cultural_align.models.trajvista import FEATURE_DIM, HIDDEN_DIM
from cultural_align.training.data import stable_unit_hash
from cultural_align.training.engine import to_namespace


ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "data" / "demo" / "ind"


class DemoReleaseTest(unittest.TestCase):
    def test_demo_summary_matches_npz_shapes(self) -> None:
        summary_path = DEMO_ROOT / "expected_outputs" / "ind_demo_expected_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["samples"], 153)
        self.assertEqual(summary["split_counts"], {"train": 106, "val": 26, "test": 21})

        for item in summary["files"]:
            with np.load(DEMO_ROOT / item["file"]) as shard:
                self.assertEqual(list(shard["state"].shape), item["state_shape"])
                self.assertEqual(list(shard["action"].shape), item["action_shape"])
                self.assertEqual(list(shard["next_state"].shape), item["state_shape"])
                self.assertEqual(list(shard["next_action"].shape), item["action_shape"])

    def test_model_dimensions_are_fixed_for_reproducibility(self) -> None:
        self.assertEqual(HIDDEN_DIM, 64)
        self.assertEqual(FEATURE_DIM, 64)
        with self.assertRaises(ValueError):
            to_namespace({"datasets": ["inD"], "dataset_dir": "x", "out_dir": "y", "hidden_dim": 32})

    def test_target_fraction_hash_matches_paper_runs(self) -> None:
        self.assertAlmostEqual(stable_unit_hash("inD/inD_04/1"), 0.11720937341144194)
        self.assertAlmostEqual(stable_unit_hash("HighD/HighD_01/42"), 0.6614286443341199)


if __name__ == "__main__":
    unittest.main()
