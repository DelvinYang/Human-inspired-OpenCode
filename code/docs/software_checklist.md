# Software and ML Checklist Mapping

This file maps the Nature Research software checklist and the Machine Learning
checklist to repository content. Keep this file current before review or public
release.

| Checklist item | Repository location | Current status |
| --- | --- | --- |
| Source code | `code/src/cultural_align/`, `code/scripts/` | Dataset, reference-path preprocessing, training, transfer, demo-metatype, and evaluation code added |
| Small demo/test dataset | `data/demo/ind/` | inD recording-04 excerpt added |
| Reviewer metatype checkpoint | `code/pretrained/metatypes/cn_us_source_metatype/` | One CN+US-source proposed-method checkpoint added for transfer-path reproduction |
| Reviewer inD evaluation checkpoint | `code/pretrained/models/ind_5k_best/` | One inD checkpoint added for full demo evaluation output |
| Appendix long-tail figure demo | `data/demo/longtail_cases/`, `code/scripts/visualization/plot_longtail_case_demo.R` | Complete R plotting demo added |
| Paper metric code | `code/scripts/evaluation/paper_metrics/actual_used/`, `code/docs/metrics.md` | Actual revision metric scripts and metric mapping added |
| System requirements | `code/INSTRUCTIONS.md`, this file | Recorded for the tested demo environment |
| Dependencies and OS versions | `code/requirements.txt`, this file | Dependencies listed; preprocessing smoke-test environment recorded below |
| Tested software versions | this file | Preprocessing smoke-test versions recorded below |
| Installation instructions | `code/INSTRUCTIONS.md` | Provided |
| Typical install time | `code/INSTRUCTIONS.md`, this file | Environment-dependent; PyTorch wheel download dominates clean installs |
| Demo instructions | `code/INSTRUCTIONS.md`, `code/examples/NOTES.md`, `code/scripts/run_release_demo.sh` | Preprocessing, training, evaluation, transfer, pretrained-evaluation, and long-tail figure demos implemented |
| Expected demo output | `data/demo/ind/NOTES.md`, `data/demo/longtail_cases/NOTES.md`, `code/examples/NOTES.md` | 153 samples across train/val/test plus demo-transfer, pretrained-evaluation, and three long-tail figure PDFs |
| Expected demo run time | `code/examples/NOTES.md` | Preprocessing smoke run is under one second on the tested local machine |
| How to run on user data | `code/INSTRUCTIONS.md`, `code/docs/reproducibility.md` | Provided |
| Quantitative reproduction instructions | `code/docs/reproducibility.md` | Provided |
| License | `code/LICENSE`, `code/INSTRUCTIONS.md` | MIT for code; datasets retain original licenses |
| Open-source repository link | `README.md`, `code/INSTRUCTIONS.md` | GitHub repository available |
| Data availability policy | `code/docs/data_policy.md` | Provided |

## Tested Environment

- Operating system: macOS 26.5 arm64
- Python version: 3.12.7
- PyTorch version: 2.10.0
- CUDA version, if applicable: not used by preprocessing demo
- CPU/GPU: local Apple Silicon CPU for preprocessing smoke test
- Install time: environment-dependent; PyTorch wheel download and hardware
  decide the clean-install wall time
- Demo preprocessing run time: 0.83 s wall time for
  `code/scripts/datasets/ind/build_ind_demo.py --clean`
- Demo training run time: 2.36 s wall time for
  `code/scripts/experiments/train_demo.py --config code/configs/experiments/demo_ind.yaml --clean`
- Demo evaluation run time: 0.90 s wall time for
  `code/scripts/evaluation/evaluate_demo.py --config code/configs/experiments/demo_ind.yaml`
- Demo transfer run time: 2.27 s wall time for
  `code/scripts/experiments/train_demo_transfer.py --config code/configs/experiments/demo_transfer_from_cn_us_metatype.yaml --clean`
- Demo transfer evaluation run time: 0.85 s wall time for
  `code/scripts/evaluation/evaluate_demo_transfer.py --config code/configs/experiments/demo_transfer_from_cn_us_metatype.yaml`
- Demo inD pretrained evaluation run time: 1.47 s wall time for
  `code/scripts/evaluation/evaluate_ind_pretrained_demo.py --config code/configs/experiments/demo_evaluate_ind_pretrained.yaml`
- Appendix long-tail figure demo run time: 6.85 s wall time for
  `Rscript code/scripts/visualization/plot_longtail_case_demo.R`
