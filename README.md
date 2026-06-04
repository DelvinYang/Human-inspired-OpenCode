# Human-inspired Data-light Cultural Alignment

This repository is the open-source release for the paper
"Human-inspired Data-light Cultural Alignment for Cross-regional Deployment of
Autonomous Vehicles".

The repository is organized to make the reviewed method installable, auditable,
and reproducible with runnable demos, released checkpoints, and scripts for the
datasets used in the paper.

Anonymous review repository:
https://anonymous.4open.science/r/Human-inspired-OpenCode-7D46/

## Release Scope

- Included: source code, configuration templates, documentation, a small inD
  demo/test subset, long-tail figure demo inputs, and compact
  reviewer checkpoints for the proposed-method transfer and inD evaluation
  paths.
- Paper-scale dataset paths are configured locally through `configs/datasets/`.
- Supported datasets by script: inD, highD, NGSIM, sinD, CitySim, DJI, and
  INTERACTION.

## Repository Layout

```text
configs/                 Dataset and experiment configuration templates.
data/demo/ind/           Small inD demo/test subset and expected demo outputs.
data/demo/longtail_cases/
                         Inputs for the Appendix long-tail case figure demo.
docs/                    Data policy, reproducibility, and checklist mapping.
examples/                End-to-end demo commands.
pretrained/              Released reviewer checkpoints.
scripts/                 Command-line entry points for data, training, and evaluation.
scripts/visualization/   Figure reproduction scripts.
scripts/evaluation/paper_metrics/
                         Actual revision metric scripts and metric notes.
src/cultural_align/      Importable Python package for the method implementation.
tests/                   Lightweight tests using the inD demo subset.
artifacts/               Local run outputs; ignored by git except for .gitkeep.
```

## System Requirements

The release is intended for Linux or macOS with Python 3.10 or newer. The demo
runs on a normal desktop CPU. GPU acceleration is recommended for full
paper-scale training.

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

The Appendix long-tail figure demo additionally uses R with the packages
`ggplot2`, `png`, and `scales`.

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
dependencies. Dataset paths are configured separately for paper-scale runs.

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
trained on the China and U.S. source regions (`DJI`, `sinD`, `NGSIM`, and
`CitySim`). It is provided so reviewers can exercise the transfer code path.
Paper-scale data-light runs set the target fraction through
`scripts/experiments/train_transfer.py`.

```bash
python scripts/experiments/train_demo_transfer.py \
  --config configs/experiments/demo_transfer_from_cn_us_metatype.yaml \
  --clean

python scripts/evaluation/evaluate_demo_transfer.py \
  --config configs/experiments/demo_transfer_from_cn_us_metatype.yaml
```

Expected transfer artifacts:

```text
artifacts/demo_ind_transfer_from_cn_us_metatype/calibrate_w/best_model.pt
artifacts/demo_ind_transfer_from_cn_us_metatype/calibrate_w/w_best.npy
artifacts/demo_ind_transfer_from_cn_us_metatype/calibrate_w/summary.json
artifacts/demo_ind_transfer_from_cn_us_metatype/ours_transfer/best_model.pt
artifacts/demo_ind_transfer_from_cn_us_metatype/ours_transfer/w_best.npy
artifacts/demo_ind_transfer_from_cn_us_metatype/ours_transfer/summary.json
artifacts/demo_ind_transfer_from_cn_us_metatype/ours_transfer/evaluation_summary.json
```

On the tested local machine, this transfer demo reports test RMSE
`[0.0106, 0.0100]` on 21 inD demo test samples. Runtime was approximately
2.27 s for transfer training and 0.85 s for transfer evaluation.

To run the release demo sequence in one command:

```bash
bash scripts/run_release_demo.sh
```

The repository also includes an inD checkpoint from a prior
`trajvista_ind_xy_psiphi_5k` experiment. It runs the released evaluation output
on the included inD demo/test split.

```bash
python scripts/evaluation/evaluate_ind_pretrained_demo.py \
  --config configs/experiments/demo_evaluate_ind_pretrained.yaml
```

Expected output:

```text
artifacts/demo_ind_pretrained_eval/evaluation_summary.json
```

On the tested local machine, this evaluation reports RMSE `[0.0089, 0.0038]`,
MAE `[0.0075, 0.0031]`, R2 mean `0.9995`, RBF-MMD `0.000212`, and standardized
loss `0.000111` on 21 inD demo test samples. Runtime was approximately 1.6 s.

The Appendix long-tail qualitative case figure demo runs the R plotting script
and writes the three case PDFs used in the Appendix.

```bash
Rscript scripts/visualization/plot_longtail_case_demo.R
```

Expected output:

```text
artifacts/longtail_case_demo/longtail_case_main_figure_01_inD_02_track12_frame409.pdf
artifacts/longtail_case_demo/longtail_case_main_figure_02_inD_18_track94_frame5270.pdf
artifacts/longtail_case_demo/longtail_case_main_figure_03_inD_17_track301_frame20516.pdf
```

Paper-scale source-domain and data-light transfer runs use
`scripts/experiments/train_domain.py` and `scripts/experiments/train_transfer.py`.
The released model fixes the temporal hidden dimension at `64` and the Psi/Phi
feature dimension at `64`, matching the paper runs.

For paper-scale data-light transfer, run `train_transfer.py` first with
`--phase calibrate_w`, then run it with `--phase finetune_with_target_w` and
`--target-w-path` pointing to the first stage `w_best.npy`.

## Instructions for Use

For full-data runs, place local dataset copies outside this repository, update
the corresponding file in `configs/datasets/`, and run the matching script under
`scripts/datasets/`.

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
configured locally before paper-scale reproduction.

## License

The code is released under the MIT License. Dataset files follow their source
dataset terms.

## Citation

Please cite the paper if you use this code. A machine-readable citation template
is provided in `CITATION.cff`.
