# Examples

This directory documents runnable examples.

## Demo inD Example

The preprocessing demo uses the small subset under `data/demo/ind/`.

```bash
python code/scripts/datasets/ind/build_ind_demo.py --config code/configs/datasets/ind_demo.yaml --clean
```

Expected preprocessing output:

```text
samples=153
split_counts={"train": 106, "val": 26, "test": 21}
wall_time_local=0.83 s
```

The generated shards are:

```text
results/demo/ind/processed/train/inD/inD_psiphi_xy_00000.npz
results/demo/ind/processed/val/inD/inD_psiphi_xy_00000.npz
results/demo/ind/processed/test/inD/inD_psiphi_xy_00000.npz
```

Train and evaluate the demo model:

```bash
python code/scripts/experiments/train_demo.py --config code/configs/experiments/demo_ind.yaml --clean
python code/scripts/evaluation/evaluate_demo.py --config code/configs/experiments/demo_ind.yaml
```

Expected training artifacts:

```text
results/demo_ind/ours/best_model.pt
results/demo_ind/ours/summary.json
results/demo_ind/ours/evaluation_summary.json
```

## Demo Transfer From Released Metatype

This example uses the released CN+US-source proposed-method metatype under
`code/pretrained/metatypes/cn_us_source_metatype/` and fine-tunes it on the same
small inD demo subset.

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

On the tested local machine, this smoke run reports transfer-demo RMSE
`[0.0106, 0.0100]` on 21 inD demo test samples.

## Demo Evaluation From Released inD Checkpoint

This example evaluates the released inD checkpoint under
`code/pretrained/models/ind_5k_best/` on the included inD demo/test split.

```bash
python code/scripts/evaluation/evaluate_ind_pretrained_demo.py \
  --config code/configs/experiments/demo_evaluate_ind_pretrained.yaml
```

Expected artifact:

```text
results/demo_ind_pretrained_eval/evaluation_summary.json
```

On the tested local machine, this evaluation reports RMSE `[0.0089, 0.0038]`,
R2 mean `0.9995`, and RBF-MMD `0.000212` on 21 inD demo test samples. Runtime
was approximately 1.6 s.

## Appendix Long-Tail Case Figure Demo

This example runs the R plotting script and writes the three Appendix long-tail
qualitative case PDFs.

```bash
Rscript code/scripts/visualization/plot_longtail_case_demo.R
```

Expected artifacts:

```text
results/longtail_case_demo/longtail_case_main_figure_01_inD_02_track12_frame409.pdf
results/longtail_case_demo/longtail_case_main_figure_02_inD_18_track94_frame5270.pdf
results/longtail_case_demo/longtail_case_main_figure_03_inD_17_track301_frame20516.pdf
```
