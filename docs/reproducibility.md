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

Reference-path pipeline:

1. Build `SceneRaw` caches with `scripts/reference_paths/precache.py`.
2. Mine cluster-average candidate centerlines with
   `scripts/reference_paths/mine_candidates.py`.
3. Convert accepted candidates to reference paths and QC artifacts with
   `scripts/reference_paths/finalize_candidates.py`.
4. Assign raw tracks to saved reference paths with
   `scripts/reference_paths/assign_tracks.py`.

Dataset preprocessing scripts:

- `scripts/datasets/highd/build_highd.py`
- `scripts/datasets/ind/build_ind.py`
- `scripts/datasets/citysim/build_citysim.py`
- `scripts/datasets/sind/build_sind.py`
- `scripts/datasets/ngsim/build_ngsim.py`
- `scripts/datasets/dji/build_dji.py`
- `scripts/datasets/interaction/build_interaction.py`

Paper-scale pipeline:

1. Build dataset-specific trajectory states.
2. Train the proposed model.
4. Run baseline scripts.
5. Evaluate RMSE, MAE, RBF-MMD, R2, and long-tail subsets.
6. Regenerate paper tables and supplementary figures.

All command examples should use local paths outside this repository for full
raw datasets and provider-restricted processed outputs.
