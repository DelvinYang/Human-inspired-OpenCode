# Paper Metric Code

This directory contains the evaluation scripts actually used during the paper
revision experiments. They are included for auditability of the reported metric
definitions and subset-selection protocols.

## Actual Revision Scripts

`actual_used/` was copied from the revision workspace on the experiment server
(`tools/dataset_building/`). These files contain the code paths used for:

- Data-level `RBF-MMD`, `RBF-MMD^2`, and `R2`:
  `evaluate_trajvista_rbfmmd_r2.py`,
  `evaluate_trajvista_filtered_rbfmmd_r2.py`,
  `evaluate_trajvista_per_metric_filtered_loro.py`,
  `evaluate_trajvista_per_metric_filtered_external_interaction.py`.
- Individual-level rollout `ADE`, `FDE`, acceleration RMSE/MAE, and `CR`:
  `evaluate_trajvista_complete_trajectory_rollout.py`,
  `evaluate_trajvista_individual_loro_rollout.py`,
  `evaluate_trajvista_individual_external_interaction_rollout.py`.
- Collective-level `Ax-RMSE` and `TTC-PE`:
  `evaluate_trajvista_collective_loro.py`,
  `evaluate_trajvista_collective_external_interaction.py`.
- Localized subset-selection protocols used to append the Localized baseline:
  `select_individual_loro_localized_only.py`,
  `select_individual_external_interaction_common_subset.py`,
  `select_individual_external_interaction_localized_only.py`.
- Long-tail qualitative complete-track rollout `ADE` and `FDE`:
  `plot_longtail_complete_track_rollout_methods.py`.

`actual_used/legacy_trajvista_common/` contains the first-round TrajVista common
metric helpers that were available locally and are relevant to paper-described
legacy metrics:

- `evaluate_va95_rpa.py`: `VA95` and `RPA`.
- `evaluate_ttc.py`: frame-level TTC series.
- `evaluate_fde.py`: open-loop/semi-rollout FDE.
- `evaluate_acc.py`: acceleration-sequence extraction.

## Metric Definitions In The Actual Code

- `R2`: `evaluate_trajvista_rbfmmd_r2.py::compute_r2_old`, computed per
  acceleration dimension and then averaged.
- `RBF-MMD^2`: `evaluate_trajvista_rbfmmd_r2.py::compute_rbf_mmd_old`, using
  the multi-scale RBF bandwidths `(1, 2, 4, 8)`. Reported tables use
  `sqrt(RBF-MMD^2)` unless the table explicitly says `RBF-MMD^2`.
- `ADE`: mean Euclidean displacement error over semi-rollout steps in
  `evaluate_trajvista_individual_loro_rollout.py::rollout_model_trajectory_metrics`.
- `FDE`: final Euclidean displacement error after the same semi-rollout.
- `CR`: fraction of trajectories with
  `FDE <= max(cr_abs_threshold, cr_rel_threshold * true_path_len)` in the main
  individual rollout script. Localized-only selection scripts additionally
  record their selection surrogate as
  `mean clipped 1 - FDE/(3*true_path_len)`.
- `Ax-RMSE`: x-acceleration RMSE in
  `evaluate_trajvista_collective_loro.py::ax_rmse`.
- `TTC-PE`: absolute error between smoothed TTC density peaks in
  `evaluate_trajvista_collective_loro.py::ttc_peak_error`; when the peak is
  underdetermined, the revision script falls back to the quantile error from
  `ttc_quantile_error`.
- `VA95`: 95th percentile of `|v| * |a|` in
  `legacy_trajvista_common/evaluate_va95_rpa.py::compute_va95`.
- `RPA`: `sum(|v| * max(|a|, 0) * dt) / sum(|v| * dt)` in
  `legacy_trajvista_common/evaluate_va95_rpa.py::compute_rpa`.

## Known Gap

The project audit notes mention `MidRatio`, but the revision workspace and the
local first-round TrajVista code do not contain an actual executed MidRatio
implementation. It is therefore not reimplemented here from scratch.
