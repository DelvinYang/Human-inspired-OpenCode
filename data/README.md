# Data Directory

This repository does not redistribute full driving datasets.

Only `data/demo/ind/` and publication-safe derived reference-path artifacts
under `data/reference_paths/` are intended to be versioned. Full raw data,
generated state/action tensors, caches, checkpoints, and paper-scale processed
tensors are ignored by git.

Users must obtain full datasets from the original providers and configure local
paths under `configs/datasets/`.
