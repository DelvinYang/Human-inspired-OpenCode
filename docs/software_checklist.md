# Software and ML Checklist Mapping

This file maps the Nature Research software checklist and the Machine Learning
checklist to repository content. Keep this file current before review or public
release.

| Checklist item | Repository location | Current status |
| --- | --- | --- |
| Source code | `src/cultural_align/`, `scripts/` | Dataset, reference-path preprocessing, training, transfer, and evaluation code added |
| Small demo/test dataset | `data/demo/ind/` | inD recording-04 excerpt added |
| System requirements | `README.md`, this file | Scaffolded |
| Dependencies and OS versions | `requirements.txt`, this file | Dependencies listed; preprocessing smoke-test environment recorded below |
| Tested software versions | this file | Preprocessing smoke-test versions recorded below |
| Installation instructions | `README.md` | Scaffolded |
| Typical install time | `README.md`, this file | TBD after validation |
| Demo instructions | `README.md`, `examples/README.md` | Preprocessing, training, and evaluation demo implemented |
| Expected demo output | `data/demo/ind/README.md`, `examples/README.md` | 153 samples across train/val/test |
| Expected demo run time | `examples/README.md` | Preprocessing smoke run is under one second on the tested local machine |
| How to run on user data | `README.md`, `docs/reproducibility.md` | Scaffolded |
| Quantitative reproduction instructions | `docs/reproducibility.md` | Scaffolded |
| License | `LICENSE`, `README.md` | MIT for code; datasets retain original licenses |
| Open-source repository link | `CITATION.cff`, `README.md` | TBD after final remote is set |
| Data availability policy | `docs/data_policy.md` | Scaffolded |

## Tested Environment

- Operating system: macOS 26.5 arm64
- Python version: 3.12.7
- PyTorch version: 2.10.0
- CUDA version, if applicable: not used by preprocessing demo
- CPU/GPU: local Apple Silicon CPU for preprocessing smoke test
- Install time: TBD after final dependency lock
- Demo preprocessing run time: 0.83 s wall time for
  `scripts/datasets/ind/build_ind_demo.py --clean`
- Demo training run time: 2.36 s wall time for
  `scripts/experiments/train_demo.py --config configs/experiments/demo_ind.yaml --clean`
- Demo evaluation run time: 0.90 s wall time for
  `scripts/evaluation/evaluate_demo.py --config configs/experiments/demo_ind.yaml`
