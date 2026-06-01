# Reproducibility Plan

The repository supports two reproducibility levels.

## Level 1: Demo Reproduction

This level uses only the small inD demo/test subset included in the repository.
It is intended for reviewers and users to verify that installation, data
loading, training, and evaluation work end to end.

Planned commands:

```bash
python scripts/datasets/ind/build_ind_demo.py --config configs/datasets/ind_demo.yaml
python scripts/experiments/train_demo.py --config configs/experiments/demo_ind.yaml
python scripts/evaluation/evaluate_demo.py --config configs/experiments/demo_ind.yaml
```

Expected outputs and tolerances should be recorded in `examples/README.md`.

## Level 2: Paper-Scale Reproduction

This level requires users to obtain the full datasets from their original
providers and configure local paths under `configs/datasets/`.

Planned pipeline:

1. Build dataset-specific trajectory states.
2. Build the unified PsiPhi training dataset.
3. Train the proposed model.
4. Run baseline scripts.
5. Evaluate RMSE, MAE, RBF-MMD, R2, and long-tail subsets.
6. Regenerate paper tables and supplementary figures.

The implementation scripts will be added under `scripts/` and the importable
method code under `src/cultural_align/`.
