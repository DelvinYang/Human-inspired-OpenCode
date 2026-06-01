# inD Demo/Test Subset

This directory is reserved for a small inD subset used to demonstrate
installation, preprocessing, training, and evaluation.

Planned contents:

- `raw/`: a minimal provider-compliant inD excerpt or synthetic inD-format files.
- `processed/`: demo tensors generated from `raw/`.
- `expected_outputs/`: small reference outputs for smoke-test comparison.

Full inD data is not included. Users who need full-scale experiments should
download inD from the original provider and update `configs/datasets/ind_demo.yaml`
or a separate local config file.
