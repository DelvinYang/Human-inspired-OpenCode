# Human-inspired Data-light Cultural Alignment

This repository is the open-source release for the paper
"Human-inspired Data-light Cultural Alignment for Cross-regional Deployment of
Autonomous Vehicles".

The repository is organized to make the reviewed method installable, auditable,
and reproducible without redistributing third-party driving datasets. Full raw
datasets are not included. A small inD demo/test subset is the only dataset
excerpt redistributed in this repository; scripts for all other datasets are
provided so users with proper dataset access can reproduce the pipeline locally.

## Release Scope

- Included: source code, configuration templates, documentation, and a small inD
  demo/test subset, plus one compact DE-source metatype checkpoint for reviewer
  reproduction of the proposed-model transfer path.
- Not included: full raw datasets, full processed datasets, model checkpoints
  other than the single released reviewer metatype, or any artifacts whose
  redistribution is restricted by dataset providers.
- Supported datasets by script: inD, highD, NGSIM, sinD, CitySim, DJI, and
  INTERACTION.

## Repository Layout

```text
configs/                 Dataset and experiment configuration templates.
data/demo/ind/           Small inD demo/test subset and expected demo outputs.
docs/                    Data policy, reproducibility, and checklist mapping.
examples/                End-to-end demo commands.
pretrained/              Single released reviewer metatype checkpoint.
scripts/                 Command-line entry points for data, training, and evaluation.
scripts/evaluation/paper_metrics/
                         Actual revision metric scripts and metric notes.
src/cultural_align/      Importable Python package for the method implementation.
tests/                   Lightweight tests using the inD demo subset.
artifacts/               Local run outputs; ignored by git except for .gitkeep.
```

## System Requirements

The release is intended for Linux or macOS with Python 3.10 or newer. The demo
does not require non-standard hardware and runs on a normal desktop CPU. GPU
acceleration is recommended only for full paper-scale training.

Tested environment:

```text
Operating system: macOS 26.5 arm64
Python: 3.12.7
PyTorch: 2.10.0
CUDA: not used for the demo
Hardware: Apple Silicon CPU; no GPU required for the demo
```

Software dependencies are listed in `requirements.txt` and `pyproject.toml`.
The minimum supported dependency versions are:

```text
numpy>=1.24
pandas>=1.5
scipy>=1.10
scikit-learn>=1.2
hdbscan>=0.8.33
torch>=2.0
tqdm>=4.66
loguru>=0.7
matplotlib>=3.7
plotly>=5.20
pyyaml>=6.0
pillow>=10.0
pyproj>=3.5
```

Exact tested software versions and demo runtimes are recorded in
`docs/software_checklist.md`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Typical installation time on a normal desktop is about 5--15 minutes for a
clean environment. The wall time is mainly determined by the PyTorch wheel
download and the local package cache; when wheels are already cached, the
install is typically shorter. The install command only installs Python package
dependencies and does not download full driving datasets.

## Demo

The demo uses only the small inD recording-04 excerpt under `data/demo/ind/`.
The included processed test dataset is:

```text
data/demo/ind/processed/test/inD/inD_psiphi_xy_00000.npz
```

```bash
python scripts/datasets/ind/build_ind_demo.py \
  --config configs/datasets/ind_demo.yaml \
  --clean

python scripts/experiments/train_demo.py \
  --config configs/experiments/demo_ind.yaml

python scripts/evaluation/evaluate_demo.py \
  --config configs/experiments/demo_ind.yaml
```

Expected preprocessing output:

```text
raw trajectory rows: 1,929
retained tracks: 13
processed samples: 153
split_counts={"train": 106, "val": 26, "test": 21}
state shape: [N, 12, 12]
action shape: [N, 2]
```

Expected demo artifacts:

```text
data/demo/ind/processed/train/inD/inD_psiphi_xy_00000.npz
data/demo/ind/processed/val/inD/inD_psiphi_xy_00000.npz
data/demo/ind/processed/test/inD/inD_psiphi_xy_00000.npz
artifacts/demo_ind/ours/best_model.pt
artifacts/demo_ind/ours/summary.json
artifacts/demo_ind/ours/evaluation_summary.json
```

On the tested local machine, the demo reports test RMSE
`[0.0205, 0.0196]` on 21 inD demo test samples. The corresponding R2 mean is
`0.9951`. Runtime on the tested normal desktop was approximately:

```text
preprocessing: 0.83 s
training: 2.36 s
evaluation: 0.90 s
```

`artifacts/` is ignored by git.

The repository also includes a compact proposed-method metatype checkpoint
trained on the Germany source domain (`HighD` and `inD`). It is provided so
reviewers can exercise the transfer code path without access to the full
training datasets. The demo transfer config uses the full tiny demo training
split to keep the smoke test stable; paper-scale data-light runs should set the
target fraction through `scripts/experiments/train_transfer.py`.

```bash
python scripts/experiments/train_demo_transfer.py \
  --config configs/experiments/demo_transfer_from_de_metatype.yaml \
  --clean

python scripts/evaluation/evaluate_demo_transfer.py \
  --config configs/experiments/demo_transfer_from_de_metatype.yaml
```

Expected transfer artifacts:

```text
artifacts/demo_ind_transfer_from_de_metatype/ours_transfer/best_model.pt
artifacts/demo_ind_transfer_from_de_metatype/ours_transfer/summary.json
artifacts/demo_ind_transfer_from_de_metatype/ours_transfer/evaluation_summary.json
```

On the tested local machine, this transfer demo reports test RMSE
`[0.0150, 0.0041]` on 21 inD demo test samples. Runtime was approximately
1.93 s for transfer training and 0.91 s for transfer evaluation.

Paper-scale source-domain and data-light transfer runs use
`scripts/experiments/train_domain.py` and `scripts/experiments/train_transfer.py`.
Comparison-method implementations are not redistributed in this code release.
The released model fixes the temporal hidden dimension at `64` and the Psi/Phi
feature dimension at `64`, matching the paper runs.

## Instructions for Use

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

## Reproduction Instructions

Full data preparation and paper-level quantitative reproduction notes are
maintained in `docs/reproducibility.md`. The repository includes scripts for
the seven supported datasets (`inD`, `highD`, `NGSIM`, `sinD`, `CitySim`,
`DJI`, and `INTERACTION`) and the actual revision metric scripts under
`scripts/evaluation/paper_metrics/actual_used/`. Full raw datasets must be
obtained from their original providers before paper-scale reproduction.

## License

The code is released under the MIT License. Dataset files remain governed by
their original providers' licenses and are not sublicensed by this repository.

## Citation

Please cite the paper if you use this code. A machine-readable citation template
is provided in `CITATION.cff`.
