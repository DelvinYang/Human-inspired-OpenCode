# Human-inspired Data-light Cultural Alignment

This repository is the open-source release for the paper
"Human-inspired Data-light Cultural Alignment for Cross-regional Deployment of
Autonomous Vehicles".

The repository is organized to make the reviewed method installable, auditable,
and reproducible with runnable demos, released checkpoints, and scripts for the
datasets used in the paper.

GitHub repository:
https://github.com/DelvinYang/Human-inspired-OpenCode.git

## Release Scope

- Included: source code, configuration templates, documentation, a small inD
  demo/test subset, long-tail figure demo inputs, and compact
  reviewer checkpoints for the proposed-method transfer and inD evaluation
  paths.
- Paper-scale dataset paths are configured locally through `code/configs/datasets/`.
- Supported datasets by script: inD, highD, NGSIM, sinD, CitySim, AD4CHE, and
  INTERACTION.

## Repository Layout

```text
code/configs/            Dataset and experiment configuration templates.
data/demo/ind/           Small inD demo/test subset and expected demo outputs.
data/demo/longtail_cases/
                         Inputs for the Appendix long-tail case figure demo.
code/docs/               Data policy, reproducibility, and checklist mapping.
code/examples/           End-to-end demo commands.
code/pretrained/         Released reviewer checkpoints.
code/scripts/            Command-line entry points for data, training, and evaluation.
code/scripts/visualization/ Figure reproduction scripts.
code/scripts/evaluation/paper_metrics/
                         Actual revision metric scripts and metric notes.
code/src/cultural_align/ Importable Python package for the method implementation.
code/tests/              Lightweight tests using the inD demo subset.
results/                 Local run outputs; ignored by git except for .gitkeep.
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

Software dependencies are listed in `code/requirements.txt` and `code/pyproject.toml`.
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
`code/docs/software_checklist.md`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
python -m pip install -e code
```

Typical installation time on a normal desktop is about 5--15 minutes for a
clean environment. The wall time is mainly determined by the PyTorch wheel
download and the local package cache; when wheels are already cached, the
install is typically shorter. The install command only installs Python package
dependencies. Dataset paths are configured separately for paper-scale runs.

## Demo

The demo uses only the small inD recording-04 excerpt under `data/demo/ind/`.
The included processed test split is:

```text
data/demo/ind/processed/test/inD/inD_psiphi_xy_00000.npz
```

```bash
python code/scripts/datasets/ind/build_ind_demo.py \
  --config code/configs/datasets/ind_demo.yaml \
  --clean

python code/scripts/experiments/train_demo.py \
  --config code/configs/experiments/demo_ind.yaml

python code/scripts/evaluation/evaluate_demo.py \
  --config code/configs/experiments/demo_ind.yaml
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
results/demo/ind/processed/train/inD/inD_psiphi_xy_00000.npz
results/demo/ind/processed/val/inD/inD_psiphi_xy_00000.npz
results/demo/ind/processed/test/inD/inD_psiphi_xy_00000.npz
results/demo_ind/ours/best_model.pt
results/demo_ind/ours/summary.json
results/demo_ind/ours/evaluation_summary.json
```

On the tested local machine, the demo reports test RMSE
`[0.0205, 0.0196]` on 21 inD demo test samples. The corresponding R2 mean is
`0.9951`; the complete evaluation output also reports RBF-MMD `0.005683`,
VA95 pred/true `12.4494/11.6891`, RPA pred/true `0.4623/0.4561`, ADE
`0.1596`, FDE `0.3410`, and CR `1.000` on 2 reconstructed demo trajectories.
Runtime on the tested normal desktop was approximately:

```text
preprocessing: 0.83 s
training: 2.36 s
evaluation: 0.90 s
```

All generated demo outputs are written under the repository-root `results/`
directory for CodeOcean reproducible-run snapshots.

The repository also includes a compact proposed-method metatype checkpoint
trained on the China and U.S. source regions (`AD4CHE`, `sinD`, `NGSIM`, and
`CitySim`). It is provided so reviewers can exercise the transfer code path.
Paper-scale data-light runs set the target fraction through
`code/scripts/experiments/train_transfer.py`.

```bash
python code/scripts/experiments/train_demo_transfer.py \
  --config code/configs/experiments/demo_transfer_from_cn_us_metatype.yaml \
  --clean

python code/scripts/evaluation/evaluate_demo_transfer.py \
  --config code/configs/experiments/demo_transfer_from_cn_us_metatype.yaml
```

Expected transfer artifacts:

```text
results/demo_ind_transfer_from_cn_us_metatype/calibrate_w/best_model.pt
results/demo_ind_transfer_from_cn_us_metatype/calibrate_w/w_best.npy
results/demo_ind_transfer_from_cn_us_metatype/calibrate_w/summary.json
results/demo_ind_transfer_from_cn_us_metatype/ours_transfer/best_model.pt
results/demo_ind_transfer_from_cn_us_metatype/ours_transfer/w_best.npy
results/demo_ind_transfer_from_cn_us_metatype/ours_transfer/summary.json
results/demo_ind_transfer_from_cn_us_metatype/ours_transfer/evaluation_summary.json
```

On the tested local machine, this transfer demo reports test RMSE
`[0.0106, 0.0100]` on 21 inD demo test samples. The complete evaluation output
reports R2 mean `0.9987`, RBF-MMD `0.006032`, VA95 pred/true
`11.8533/11.6891`, RPA pred/true `0.4484/0.4561`, ADE `0.1073`, FDE `0.2505`,
and CR `1.000` on 2 reconstructed demo trajectories. Runtime was approximately
2.27 s for transfer training and 0.85 s for transfer evaluation. On this
compact demo test split, the transfer run is better than the scratch run on
most reported metrics.

To run the release demo sequence in one command:

```bash
bash code/scripts/run_release_demo.sh
```

The repository also includes an inD checkpoint from a prior
`trajvista_ind_xy_psiphi_5k` experiment. It runs the released evaluation output
on the included inD demo/test split.

```bash
python code/scripts/evaluation/evaluate_ind_pretrained_demo.py \
  --config code/configs/experiments/demo_evaluate_ind_pretrained.yaml
```

Expected output:

```text
results/demo_ind_pretrained_eval/evaluation_summary.json
```

On the tested local machine, this evaluation reports RMSE `[0.0089, 0.0038]`,
MAE `[0.0075, 0.0031]`, R2 mean `0.9995`, RBF-MMD `0.003154`, VA95 pred/true
`11.4421/11.6891`, RPA pred/true `0.4484/0.4561`, ADE `0.0845`, FDE `0.1836`,
CR `1.000`, and standardized loss `0.000111` on 21 inD demo test samples.
Runtime was approximately 1.6 s.

The Appendix long-tail qualitative case figure demo runs the R plotting script
and writes the three case PDFs used in the Appendix.

```bash
Rscript code/scripts/visualization/plot_longtail_case_demo.R
```

Expected output:

```text
results/longtail_case_demo/longtail_case_main_figure_01_inD_02_track12_frame409.pdf
results/longtail_case_demo/longtail_case_main_figure_02_inD_18_track94_frame5270.pdf
results/longtail_case_demo/longtail_case_main_figure_03_inD_17_track301_frame20516.pdf
```

Paper-scale source-domain and data-light transfer runs use
`code/scripts/experiments/train_domain.py` and `code/scripts/experiments/train_transfer.py`.
The released model fixes the temporal hidden dimension at `64` and the Psi/Phi
feature dimension at `64`, matching the paper runs.

For paper-scale data-light transfer, run `train_transfer.py` first with
`--phase calibrate_w`, then run it with `--phase finetune_with_target_w` and
`--target-w-path` pointing to the first stage `w_best.npy`.

## Instructions for Use

For full-data runs, place local dataset copies outside this repository, update
the corresponding file in `code/configs/datasets/`, and run the matching script under
`code/scripts/datasets/`.

Example full-data preprocessing command:

```bash
python code/scripts/datasets/highd/build_highd.py \
  --raw-root /path/to/highD/data \
  --assignment-dir results/reference_paths_work/assignments \
  --out-dir results/datasets/trajvista_xy_psiphi_v1
```

Reference-path preparation is split into explicit steps:

```bash
python code/scripts/reference_paths/precache.py \
  --dataset-root /path/to/raw_dataset_root \
  --cache-root results/reference_paths_work/cache/scenes \
  --dataset HighD

python code/scripts/reference_paths/mine_candidates.py \
  --cache-root results/reference_paths_work/cache/scenes \
  --work-dir results/reference_paths_work \
  --dataset HighD

python code/scripts/reference_paths/finalize_candidates.py \
  --candidate-file results/reference_paths_work/reference_paths/candidates/HighD/HighD_01.json \
  --cache-root results/reference_paths_work/cache/scenes \
  --work-dir results/reference_paths_work

python code/scripts/reference_paths/assign_tracks.py \
  --cache-root results/reference_paths_work/cache/scenes \
  --work-dir results/reference_paths_work \
  --dataset HighD
```

Reference-path outputs and track-assignment filters for full-data runs are
generated under `results/reference_paths_work/` by the commands above.

## Reproduction Instructions

Full data preparation and paper-level quantitative reproduction notes are
maintained in `code/docs/reproducibility.md`. The repository includes scripts for
the seven supported datasets (`inD`, `highD`, `NGSIM`, `sinD`, `CitySim`,
`AD4CHE`, and `INTERACTION`) and the actual revision metric scripts under
`code/scripts/evaluation/paper_metrics/actual_used/`. Full raw datasets and
paper-scale reference-path outputs must be configured or generated locally
before paper-scale reproduction.

## License

The code is released under the MIT License. Dataset files follow their source
dataset terms. The license file is `code/LICENSE`.

## Citation

Please cite the paper if you use this code. A machine-readable citation template
is provided in `code/CITATION.cff`.
