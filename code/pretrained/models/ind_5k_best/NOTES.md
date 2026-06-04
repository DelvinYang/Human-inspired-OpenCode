# inD 5k Best Checkpoint

This directory contains a proposed-method checkpoint from the
`trajvista_ind_xy_psiphi_5k` inD experiment. It is used by the released inD
evaluation demo.

Contents:

- `best_model.pt`: checkpoint compatible with the released model.
- `best_model.sha256`: SHA-256 checksum for the checkpoint.
- `metadata.json`: source and validation metadata.

The checkpoint loads with the released model keys.

Demo evaluation command:

```bash
python code/scripts/evaluation/evaluate_ind_pretrained_demo.py \
  --config code/configs/experiments/demo_evaluate_ind_pretrained.yaml
```

Expected output is written to
`results/demo_ind_pretrained_eval/evaluation_summary.json` and includes RMSE,
MAE, R2, RBF-MMD, VA95, RPA, ADE, FDE, CR, and standardized loss.
