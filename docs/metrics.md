# Metric Reproduction Notes

The metric code released with this repository is the code used during the
revision experiments, not a clean-room reimplementation. The audit copy lives
under `scripts/evaluation/paper_metrics/actual_used/`.

| Metric | Actual code location |
| --- | --- |
| `RBF-MMD^2`, `sqrt RBF-MMD` | `scripts/evaluation/paper_metrics/actual_used/evaluate_trajvista_rbfmmd_r2.py` |
| `R2` | `scripts/evaluation/paper_metrics/actual_used/evaluate_trajvista_rbfmmd_r2.py` |
| filtered/order-selected DataLevel tables | `scripts/evaluation/paper_metrics/actual_used/evaluate_trajvista_per_metric_filtered_loro.py` and external-INTERACTION variant |
| `ADE`, `FDE`, main `CR` | `scripts/evaluation/paper_metrics/actual_used/evaluate_trajvista_individual_loro_rollout.py` |
| complete-trajectory FDE diagnostics | `scripts/evaluation/paper_metrics/actual_used/evaluate_trajvista_complete_trajectory_rollout.py` |
| `Ax-RMSE`, `TTC-PE` | `scripts/evaluation/paper_metrics/actual_used/evaluate_trajvista_collective_loro.py` |
| Localized subset-selection CR surrogate | `scripts/evaluation/paper_metrics/actual_used/select_individual_loro_localized_only.py` and external-INTERACTION variant |
| long-tail complete-track case `ADE`/`FDE` | `scripts/evaluation/paper_metrics/actual_used/plot_longtail_complete_track_rollout_methods.py` |
| `VA95`, `RPA` | `scripts/evaluation/paper_metrics/actual_used/legacy_trajvista_common/evaluate_va95_rpa.py` |
| frame-level TTC series | `scripts/evaluation/paper_metrics/actual_used/legacy_trajvista_common/evaluate_ttc.py` |
| acceleration-sequence diagnostics | `scripts/evaluation/paper_metrics/actual_used/legacy_trajvista_common/evaluate_acc.py` |

The scripts retain the original revision experiment structure and may reference
paper-scale checkpoints or baseline model classes that are not redistributed in
this public repository. They are included so reviewers can inspect the exact
metric definitions and reproduce the calculations after supplying their own
permitted data/checkpoint paths.

`MidRatio` appears in earlier project audit notes as a desired metric, but no
actual executed implementation was found in the revision server workspace or in
the local first-round TrajVista code. It is not fabricated in this release.
