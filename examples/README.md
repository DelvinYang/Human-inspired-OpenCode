# Examples

This directory documents runnable examples.

## Demo inD Example

The preprocessing demo uses the small subset under `data/demo/ind/`.

```bash
python scripts/datasets/ind/build_ind_demo.py --config configs/datasets/ind_demo.yaml --clean
```

Expected preprocessing output:

```text
samples=153
split_counts={"train": 106, "val": 26, "test": 21}
wall_time_local=0.83 s
```

The generated shards are:

```text
data/demo/ind/processed/train/inD/inD_psiphi_xy_00000.npz
data/demo/ind/processed/val/inD/inD_psiphi_xy_00000.npz
data/demo/ind/processed/test/inD/inD_psiphi_xy_00000.npz
```

The model training and evaluation demo commands are placeholders until the
public model code is added:

```bash
python scripts/experiments/train_demo.py --config configs/experiments/demo_ind.yaml
python scripts/evaluation/evaluate_demo.py --config configs/experiments/demo_ind.yaml
```
