# CN+US Source Metatype

`best_model.pt` is a proposed-method source metatype trained on the China and
U.S. source regions from the revision pooled-source experiment. The source
datasets are `DJI`, `sinD`, `NGSIM`, and `CitySim`.

- temporal hidden dimension: `64`
- Psi/Phi feature dimension: `64`
- state window shape: `[N, 12, 12]`
- action shape: `[N, 2]`

The checkpoint loads with PyTorch `weights_only=True` and matches the released
model keys.

Use it with:

```bash
python scripts/experiments/train_demo_transfer.py \
  --config configs/experiments/demo_transfer_from_cn_us_metatype.yaml \
  --clean

python scripts/evaluation/evaluate_demo_transfer.py \
  --config configs/experiments/demo_transfer_from_cn_us_metatype.yaml
```
