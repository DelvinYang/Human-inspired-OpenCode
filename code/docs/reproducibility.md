# Reproducibility Plan

The repository supports two reproducibility levels.

## Level 1: Demo Reproduction

This level uses only the small inD demo/test subset included in the repository.
It is intended for reviewers and users to verify that installation, data
loading, training, and evaluation work before full-dataset access is configured.

Implemented preprocessing command:

```bash
python code/scripts/datasets/ind/build_ind_demo.py --config code/configs/datasets/ind_demo.yaml --clean
```

Expected result:

```text
samples=153
split_counts={"train": 106, "val": 26, "test": 21}
```

Demo training and evaluation:

```bash
python code/scripts/experiments/train_demo.py --config code/configs/experiments/demo_ind.yaml --clean
python code/scripts/evaluation/evaluate_demo.py --config code/configs/experiments/demo_ind.yaml
```

The proposed model architecture fixes the temporal hidden dimension at `64` and
the Psi/Phi feature dimension at `64`. These dimensions are kept as architecture
constants, matching the released checkpoints and paper runs.

The repository also includes one proposed-method CN+US-source metatype
checkpoint. Reviewers can use it to exercise the paper's data-light transfer
path on the small inD demo subset. The demo runs `calibrate_w` followed by
`finetune_with_target_w`; paper-scale runs set the target fraction explicitly.

```bash
python code/scripts/experiments/train_demo_transfer.py \
  --config code/configs/experiments/demo_transfer_from_cn_us_metatype.yaml \
  --clean
python code/scripts/evaluation/evaluate_demo_transfer.py \
  --config code/configs/experiments/demo_transfer_from_cn_us_metatype.yaml
```

The same demo sequence can be run with:

```bash
bash code/scripts/run_release_demo.sh
```

The repository also includes one inD checkpoint for running the full released
evaluation output on the included inD demo/test split:

```bash
python code/scripts/evaluation/evaluate_ind_pretrained_demo.py \
  --config code/configs/experiments/demo_evaluate_ind_pretrained.yaml
```

## Level 2: Paper-Scale Reproduction

This level uses local dataset paths configured under `code/configs/datasets/`.

Reference-path pipeline:

1. Build `SceneRaw` caches with `code/scripts/reference_paths/precache.py`.
2. Mine cluster-average candidate centerlines with
   `code/scripts/reference_paths/mine_candidates.py`.
3. Convert accepted candidates to reference paths and QC artifacts with
   `code/scripts/reference_paths/finalize_candidates.py`. When a matching
   `reference_paths/manual_selection/<dataset>/<scene_id>.json` file exists,
   this step applies the manually accepted, deleted, and rejected candidate IDs.
4. Assign raw tracks to final reference paths with
   `code/scripts/reference_paths/assign_tracks.py`.

Dataset preprocessing scripts:

- `code/scripts/datasets/highd/build_highd.py`
- `code/scripts/datasets/ind/build_ind.py`
- `code/scripts/datasets/citysim/build_citysim.py`
- `code/scripts/datasets/sind/build_sind.py`
- `code/scripts/datasets/ngsim/build_ngsim.py`
- `code/scripts/datasets/dji/build_dji.py` (AD4CHE)
- `code/scripts/datasets/interaction/build_interaction.py`

Paper-scale pipeline:

1. Build dataset-specific trajectory states.
2. Train source-domain or pooled proposed models with
   `code/scripts/experiments/train_domain.py`.
3. Run data-light transfer with `code/scripts/experiments/train_transfer.py` using
   `calibrate_w` followed by `finetune_with_target_w`.
4. Evaluate proposed-model checkpoints with `code/scripts/evaluation/evaluate_model.py`.
5. Regenerate paper tables and supplementary figures.

The exact revision metric scripts used for data-level, individual-level,
collective-level, and long-tail evaluations are included under
`code/scripts/evaluation/paper_metrics/actual_used/`. See `code/docs/metrics.md` for the
metric-to-script mapping.

Full-data command examples use local paths outside this repository.

Paper-scale reference-path metadata and final assignment filters are generated
locally under the configured `work_dir`, typically
`results/reference_paths_work/`.
