# Pretrained Artifacts

This directory contains the trained model artifacts redistributed with this
release:

- `metatypes/de_source_metatype/`: compact Germany-source metatype checkpoint
  for the proposed-method transfer demo.
- `models/ind_5k_best/`: sanitized inD checkpoint for running the full released
  evaluation output on the included inD demo/test subset.

These checkpoints are included to let reviewers verify checkpoint loading,
data-light transfer, and evaluation without receiving the full third-party
training datasets. They contain trained parameters and normalization statistics
only.
