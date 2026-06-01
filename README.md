# Human-inspired Data-light Cultural Alignment

This repository is the open-source release scaffold for the paper
"Human-inspired Data-light Cultural Alignment for Cross-regional Deployment of
Autonomous Vehicles".

The repository is organized to make the reviewed method installable, auditable,
and reproducible without redistributing third-party driving datasets. Full raw
datasets are not included. A small inD demo/test subset is the only data planned
for publication in this repository; scripts for all other datasets are provided
so users with proper dataset access can reproduce the pipeline locally.

## Release Scope

- Included: source code, configuration templates, documentation, and a small inD
  demo/test subset.
- Not included: full raw datasets, full processed datasets, model checkpoints
  trained on third-party data, or any artifacts whose redistribution is
  restricted by dataset providers.
- Supported datasets by script: inD, highD, NGSIM, sinD, CitySim, DJI, and
  INTERACTION.

## Repository Layout

```text
configs/                 Dataset and experiment configuration templates.
data/demo/ind/           Small inD demo/test subset and expected demo outputs.
docs/                    Data policy, reproducibility, and checklist mapping.
examples/                End-to-end demo commands.
scripts/                 Command-line entry points for data, training, and evaluation.
src/cultural_align/      Importable Python package for the method implementation.
tests/                   Lightweight tests using the inD demo subset.
artifacts/               Local run outputs; ignored by git except for .gitkeep.
```

## System Requirements

The final release is intended for Linux or macOS with Python 3.10 or 3.11.
GPU acceleration is recommended for full training, but the inD demo should run
on a normal desktop CPU once the implementation is added.

Exact tested operating systems, CUDA versions, and package versions should be
recorded in `docs/software_checklist.md` before publication.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Typical installation time on a normal desktop machine should be recorded after
the implementation and dependency set are finalized.

## Demo

The demo uses only the small inD recording-04 excerpt under `data/demo/ind/`.

```bash
python scripts/datasets/ind/build_ind_demo.py \
  --config configs/datasets/ind_demo.yaml \
  --clean

python scripts/experiments/train_demo.py \
  --config configs/experiments/demo_ind.yaml

python scripts/evaluation/evaluate_demo.py \
  --config configs/experiments/demo_ind.yaml
```

The preprocessing step is fully implemented and writes compressed NPZ shards
under `data/demo/ind/processed/`. The training and evaluation demo entry points
remain placeholders until the model release code is added.

## Running on User Data

Users should obtain each raw dataset from the original provider and comply with
that provider's license and access terms. After placing local raw data outside
this repository, update the corresponding file in `configs/datasets/` and run
the matching script under `scripts/datasets/`.

Example full-data preprocessing command:

```bash
python scripts/datasets/highd/build_highd.py \
  --raw-root /path/to/highD/data \
  --assignment-dir data/reference_paths/assignments \
  --out-dir artifacts/datasets/trajvista_xy_psiphi_v1
```

Reference-path preparation is split into explicit steps:

```bash
python scripts/reference_paths/precache.py \
  --dataset-root /path/to/raw_dataset_root \
  --cache-root artifacts/reference_paths_work/cache/scenes \
  --dataset HighD

python scripts/reference_paths/mine_candidates.py \
  --cache-root artifacts/reference_paths_work/cache/scenes \
  --work-dir artifacts/reference_paths_work \
  --dataset HighD

python scripts/reference_paths/finalize_candidates.py \
  --candidate-file artifacts/reference_paths_work/reference_paths/candidates/HighD/HighD_01.json \
  --cache-root artifacts/reference_paths_work/cache/scenes \
  --work-dir artifacts/reference_paths_work

python scripts/reference_paths/assign_tracks.py \
  --cache-root artifacts/reference_paths_work/cache/scenes \
  --work-dir artifacts/reference_paths_work \
  --dataset HighD
```

The manually reviewed reference paths and track-assignment filters used by the
release are included under `data/reference_paths/`.

Full data preparation and paper-level reproduction notes are maintained in
`docs/reproducibility.md`.

## License

The code is released under the MIT License. Dataset files remain governed by
their original providers' licenses and are not sublicensed by this repository.

## Citation

Please cite the paper if you use this code. A machine-readable citation template
is provided in `CITATION.cff`.
