from __future__ import annotations

import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
METRIC_ROOT = CODE_ROOT / "scripts" / "evaluation" / "paper_metrics" / "actual_used"


class PaperMetricSourceTest(unittest.TestCase):
    def test_revision_metric_scripts_are_released(self) -> None:
        expected = {
            "evaluate_trajvista_rbfmmd_r2.py": ("def compute_r2_old", "def compute_rbf_mmd_old"),
            "evaluate_trajvista_individual_loro_rollout.py": ("def rollout_model_trajectory_metrics", "def aggregate_traj_rows"),
            "evaluate_trajvista_collective_loro.py": ("def ax_rmse", "def ttc_peak_error"),
            "plot_longtail_complete_track_rollout_methods.py": ("ade_m", "fde_m"),
            "select_individual_loro_localized_only.py": ("CR_PROTOCOL", "def metrics"),
        }
        for filename, needles in expected.items():
            text = (METRIC_ROOT / filename).read_text(encoding="utf-8")
            for needle in needles:
                self.assertIn(needle, text, filename)

    def test_legacy_metric_helpers_are_released(self) -> None:
        legacy = METRIC_ROOT / "legacy_trajvista_common"
        expected = {
            "evaluate_va95_rpa.py": ("def compute_va95", "def compute_rpa"),
            "evaluate_ttc.py": ("def _compute_frame_ttc",),
            "evaluate_fde.py": ("def _true_displacement", "def _predict_displacement"),
            "evaluate_acc.py": ("def run_acc_evaluation",),
        }
        for filename, needles in expected.items():
            text = (legacy / filename).read_text(encoding="utf-8")
            for needle in needles:
                self.assertIn(needle, text, filename)

    def test_no_absolute_local_paths_in_released_metric_sources(self) -> None:
        blocked = ("/" + "Volumes/", "/" + "Users/", "/" + "home/yangjj", "crossculture_" + "second_round")
        for path in METRIC_ROOT.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertFalse(any(needle in text for needle in blocked), str(path))


if __name__ == "__main__":
    unittest.main()
