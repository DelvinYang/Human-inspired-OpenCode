# Model Card

## Model Details

- Model family: Psi/Phi successor-feature model for human-inspired data-light
  cultural alignment.
- Architecture: temporal encoder with fixed hidden dimension `64`, fixed
  Psi/Phi feature dimension `64`, and a learned preference vector `w`.
- Inputs: trajectory state windows with shape `[N, 12, 12]` produced by the
  released preprocessing pipeline.
- Outputs: next-step action predictions with shape `[N, 2]`, corresponding to
  longitudinal and lateral acceleration components in the processed schema.

## Intended Use

This code is intended for research reproduction and method inspection. The
included inD demo subset is a smoke test for installation, preprocessing,
training, checkpoint loading, and evaluation. Paper-scale experiments require
users to obtain the original datasets and run the released preprocessing scripts
locally under the dataset providers' terms.

## Out-of-Scope Use

This release is not intended for safety-critical vehicle control, operational
autonomous-driving deployment, or redistribution of full third-party datasets
and checkpoints. The demo subset is too small to support claims about
paper-level accuracy or deployment performance.

## Evaluation

On the included inD demo test split, the current smoke-test run evaluates 21
state-transition samples and reports RMSE `[0.0205, 0.0196]` for the two action
dimensions, with mean RMSE `0.0200`, mean R2 `0.9951`, and RBF-MMD `0.000417`.
These values are expected for the small smooth demo excerpt and should be used
only to check that the released code path is functioning.

The corresponding previous-action reference RMSE is `[0.0236, 0.0216]`, so the
demo model is modestly better than the local persistence reference on this tiny
split. Full quantitative claims should be taken from the paper-scale
experiments, not from this demo.

## Data and License Notes

Full raw datasets, full processed tensors, and checkpoints trained on restricted
third-party data are not redistributed. Users are responsible for obtaining
datasets from their original providers and complying with the applicable
licenses and citation requirements.
