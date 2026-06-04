# Scripts

Script directories are organized by role and dataset.

- `datasets/highd/build_highd.py`: highD state/action shard builder.
- `datasets/ind/build_ind.py`: full inD state/action shard builder.
- `datasets/citysim/build_citysim.py`: CitySim ABDE state/action shard builder.
- `datasets/sind/build_sind.py`: sinD state/action shard builder.
- `datasets/ngsim/build_ngsim.py`: NGSIM state/action shard builder.
- `datasets/dji/build_dji.py`: DJI state/action shard builder.
- `datasets/interaction/build_interaction.py`: INTERACTION CHN/DEU/USA shard
  builder.
- `reference_paths/`: raw-scene caching, candidate mining, candidate
  finalization/QC, and raw-track assignment scripts.
- `experiments/train_demo.py`: CPU smoke training on the included inD demo set.
- `experiments/train_domain.py`: source-domain or pooled training for the
  proposed model.
- `experiments/train_transfer.py`: data-light target adaptation for the
  proposed model.
- `evaluation/evaluate_demo.py`: demo checkpoint evaluation.
- `evaluation/evaluate_ind_pretrained_demo.py`: full released evaluation output
  for the included inD pretrained checkpoint on the demo/test split.
- `evaluation/evaluate_model.py`: paper-scale checkpoint evaluation on NPZ test
  splits.
- `evaluation/paper_metrics/actual_used/`: actual metric and table-generation
  scripts used in the paper revision experiments.
- `visualization/plot_longtail_case_demo.R`: R demo that writes the three
  Appendix long-tail qualitative case PDFs.

Dataset scripts read local dataset paths configured under `configs/datasets/`.
Generated paper-scale outputs should be written under local artifact
directories.
