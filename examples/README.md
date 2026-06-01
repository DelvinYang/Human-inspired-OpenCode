# Examples

This directory documents runnable examples.

## Demo inD Example

The demo will use the small subset under `data/demo/ind/`.

```bash
python scripts/datasets/ind/build_ind_demo.py --config configs/datasets/ind_demo.yaml
python scripts/experiments/train_demo.py --config configs/experiments/demo_ind.yaml
python scripts/evaluation/evaluate_demo.py --config configs/experiments/demo_ind.yaml
```

Record the final expected output file list, metric range, and run time here
after the demo data and code are added.
