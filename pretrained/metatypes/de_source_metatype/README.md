# DE Source Metatype

`best_model.pt` is a proposed-method source metatype trained on the Germany
source domain (`HighD` and `inD`) with the paper architecture:

- temporal hidden dimension: `64`
- Psi/Phi feature dimension: `64`
- state window shape: `[N, 12, 12]`
- action shape: `[N, 2]`

The checkpoint was converted for release by removing local filesystem paths,
renaming legacy recurrent-layer keys to the public `backbone.*` module layout,
and dropping unused legacy `decoder.*` tensors. The resulting file loads with
PyTorch `weights_only=True` and matches the released model keys.

Use it with:

```bash
python scripts/experiments/train_demo_transfer.py \
  --config configs/experiments/demo_transfer_from_de_metatype.yaml \
  --clean

python scripts/evaluation/evaluate_demo_transfer.py \
  --config configs/experiments/demo_transfer_from_de_metatype.yaml
```
