# Metric Reproduction Notes

The metric code released with this repository is the code used during the
revision experiments. The audit copy lives under
`scripts/evaluation/paper_metrics/actual_used/`.

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

The scripts retain the original revision experiment structure. They are included
so reviewers can inspect the exact metric definitions and reproduce the
calculations with local data and checkpoint paths.

The final released metric scripts cover the data-level, individual-level,
collective-level, and long-tail metrics listed above.

The release demo evaluator (`scripts/evaluation/evaluate_model.py`) reports the
same core evaluation set on the included processed NPZ test split. R2 and
RBF-MMD/RBF-MMD2 are data-level metrics; VA95 and RPA are reported as
rollout/true driving-style values; ADE, FDE, and CR are computed from
reconstructed stride-12 semi-rollout trajectories when trajectory metadata is
available.
