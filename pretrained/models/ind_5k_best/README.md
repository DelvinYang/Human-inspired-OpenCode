# inD 5k Best Checkpoint

This directory contains a proposed-method checkpoint from the
`trajvista_ind_xy_psiphi_5k` inD experiment. It is used by the released inD
evaluation demo.

Contents:

- `best_model.pt`: checkpoint compatible with the released model.
- `best_model.sha256`: SHA-256 checksum for the checkpoint.
- `metadata.json`: source, conversion, and validation metadata.

The checkpoint was converted before release by adding the current `backbone.`
prefix to recurrent encoder keys, dropping legacy decoder keys that are not used
by the released model, and removing local path values from checkpoint arguments.

Demo evaluation command:

```bash
python scripts/evaluation/evaluate_ind_pretrained_demo.py \
  --config configs/experiments/demo_evaluate_ind_pretrained.yaml
```

Expected output is written to
`artifacts/demo_ind_pretrained_eval/evaluation_summary.json` and includes RMSE,
MAE, R2, RBF-MMD, and standardized loss.
