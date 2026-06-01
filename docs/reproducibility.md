# Reproducibility Plan

The repository supports two reproducibility levels.

## Level 1: Demo Reproduction

This level uses only the small inD demo/test subset included in the repository.
It is intended for reviewers and users to verify that installation, data
loading, training, and evaluation work before full-dataset access is configured.

Implemented preprocessing command:

```bash
python scripts/datasets/ind/build_ind_demo.py --config configs/datasets/ind_demo.yaml --clean
```

Expected result:

```text
samples=153
split_counts={"train": 106, "val": 26, "test": 21}
```

Demo training and evaluation:

```bash
python scripts/experiments/train_demo.py --config configs/experiments/demo_ind.yaml --clean
python scripts/evaluation/evaluate_demo.py --config configs/experiments/demo_ind.yaml
```

The proposed model architecture fixes the temporal hidden dimension at `64` and
the Psi/Phi feature dimension at `64`. These dimensions are not exposed as
configuration knobs, to keep released checkpoints and reproduced runs
architecture-compatible.

The repository also includes one proposed-method DE-source metatype checkpoint.
Reviewers can use it to exercise the paper's data-light transfer path on the
small inD demo subset. Because the included subset has only 106 training
samples, the demo config uses `target_fraction: 1.0`; paper-scale runs should
set the intended target fraction explicitly.

```bash
python scripts/experiments/train_demo_transfer.py \
  --config configs/experiments/demo_transfer_from_de_metatype.yaml \
  --clean
python scripts/evaluation/evaluate_demo_transfer.py \
  --config configs/experiments/demo_transfer_from_de_metatype.yaml
```

## Level 2: Paper-Scale Reproduction

This level requires users to obtain the full datasets from their original
providers and configure local paths under `configs/datasets/`.

Reference-path pipeline:

1. Build `SceneRaw` caches with `scripts/reference_paths/precache.py`.
2. Mine cluster-average candidate centerlines with
   `scripts/reference_paths/mine_candidates.py`.
3. Convert accepted candidates to reference paths and QC artifacts with
   `scripts/reference_paths/finalize_candidates.py`. When a matching
   `reference_paths/manual_selection/<dataset>/<scene_id>.json` file exists,
   this step applies the manually accepted, deleted, and rejected candidate IDs.
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
2. Train source-domain or pooled proposed models with
   `scripts/experiments/train_domain.py`.
3. Run data-light transfer with `scripts/experiments/train_transfer.py`.
4. Evaluate proposed-model checkpoints with `scripts/evaluation/evaluate_model.py`.
5. Regenerate paper tables and supplementary figures.

The exact revision metric scripts used for data-level, individual-level,
collective-level, and long-tail evaluations are included under
`scripts/evaluation/paper_metrics/actual_used/`. See `docs/metrics.md` for the
metric-to-script mapping.

All command examples should use local paths outside this repository for full
raw datasets and provider-restricted processed outputs.

The release includes the reviewed reference-path metadata and final assignment
filters under `data/reference_paths/`; raw trajectory caches remain excluded.
