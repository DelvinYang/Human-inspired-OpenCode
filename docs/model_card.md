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
users to configure local dataset paths and run the released preprocessing
scripts.

The included DE-source metatype checkpoint is intended to verify the
proposed-model checkpoint-loading and data-light transfer path. The full
paper-scale suite is covered by the reproduction pipeline.
The included inD checkpoint is intended to verify the released evaluation path
on the small inD demo/test split.

## Scope

This release is scoped to research reproduction and method inspection. The demo
subset checks repository commands; paper-level results are reported in the
manuscript experiments.

## Evaluation

On the included inD demo test split, the current smoke-test run evaluates 21
state-transition samples and reports RMSE `[0.0205, 0.0196]` for the two action
dimensions, with mean RMSE `0.0200`, mean R2 `0.9951`, and RBF-MMD `0.000417`.
These values check that the released code path is functioning.

Full quantitative claims are reported by the paper-scale experiments.

The released DE-source metatype metadata records source-domain test RMSE
`[0.0040, 0.0031]` on 334,636 Germany-domain samples before demo fine-tuning.
These numbers document the included checkpoint; the inD demo transfer command
checks the transfer path on the repository demo subset.

The released metatype transfer smoke test reports inD demo RMSE
`[0.0149, 0.0041]` on 21 samples.

The released inD checkpoint evaluation demo reports RMSE `[0.0089, 0.0038]`,
mean R2 `0.9995`, and RBF-MMD `0.000212` on the same 21-sample demo test split.

## Data Notes

Paper-scale experiments use local dataset paths configured under
`configs/datasets/`. Generated tensors, caches, and paper-scale outputs are
written under local artifact directories.
