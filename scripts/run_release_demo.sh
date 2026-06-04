#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
R_BIN="${R_BIN:-Rscript}"

cd "$ROOT_DIR"

SCRATCH_CONFIG="configs/experiments/demo_ind.yaml"
TRANSFER_CONFIG="configs/experiments/demo_transfer_from_cn_us_metatype.yaml"
DATASET_CONFIG="configs/datasets/ind_demo.yaml"
DATASET_DIR="data/demo/ind/processed"
SCRATCH_CHECKPOINT="artifacts/demo_ind/ours/best_model.pt"
TRANSFER_CHECKPOINT="artifacts/demo_ind_transfer_from_cn_us_metatype/ours_transfer/best_model.pt"
EVAL_ROOT="artifacts/release_demo_eval"

run_step() {
  echo
  echo "== $*"
  "$@"
}

run_step "$PYTHON_BIN" scripts/datasets/ind/build_ind_demo.py --config "$DATASET_CONFIG" --clean
run_step "$PYTHON_BIN" scripts/experiments/train_demo.py --config "$SCRATCH_CONFIG" --clean
run_step "$PYTHON_BIN" scripts/evaluation/evaluate_demo.py --config "$SCRATCH_CONFIG"

run_step "$PYTHON_BIN" scripts/experiments/train_demo_transfer.py --config "$TRANSFER_CONFIG" --clean
run_step "$PYTHON_BIN" scripts/evaluation/evaluate_demo_transfer.py --config "$TRANSFER_CONFIG"

rm -rf "$EVAL_ROOT"
mkdir -p "$EVAL_ROOT"
run_step "$PYTHON_BIN" scripts/evaluation/evaluate_model.py \
  --checkpoint "$SCRATCH_CHECKPOINT" \
  --dataset-dir "$DATASET_DIR" \
  --datasets inD \
  --out-dir "$EVAL_ROOT/scratch"
run_step "$PYTHON_BIN" scripts/evaluation/evaluate_model.py \
  --checkpoint "$TRANSFER_CHECKPOINT" \
  --dataset-dir "$DATASET_DIR" \
  --datasets inD \
  --out-dir "$EVAL_ROOT/transfer"

run_step "$R_BIN" scripts/visualization/plot_longtail_case_demo.R

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

rows = [
    ("Scratch demo", Path("artifacts/demo_ind/ours/evaluation_summary.json")),
    ("CN+US metatype transfer", Path("artifacts/demo_ind_transfer_from_cn_us_metatype/ours_transfer/evaluation_summary.json")),
    ("Scratch full evaluation", Path("artifacts/release_demo_eval/scratch/evaluation_summary.json")),
    ("Transfer full evaluation", Path("artifacts/release_demo_eval/transfer/evaluation_summary.json")),
]

def metrics(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("metrics") or payload
    rmse = data["raw_rmse"]
    return {
        "rmse": rmse,
        "mean_rmse": sum(rmse) / len(rmse),
        "r2_mean": data["r2_mean"],
        "rbf_mmd": data["rbf_mmd"],
        "va95_pred": data.get("va95_pred_mean", data.get("va95", float("nan"))),
        "va95_true": data.get("va95_true_mean", float("nan")),
        "rpa_pred": data.get("rpa_pred_mean", data.get("rpa", float("nan"))),
        "rpa_true": data.get("rpa_true_mean", float("nan")),
        "ade": data.get("ade", float("nan")),
        "fde": data.get("fde", float("nan")),
        "cr": data.get("cr", float("nan")),
        "samples": data["samples"],
        "trajectories": data.get("trajectories", 0),
    }

print("\n== Demo metric summary")
print(f"{'Run':<28} {'RMSE [ax, ay]':<24} {'Mean RMSE':>10} {'R2 mean':>10} {'RBF-MMD':>12} {'VA95 p/t':>17} {'RPA p/t':>17} {'ADE':>10} {'FDE':>10} {'CR':>8} {'N/T':>9}")
print("-" * 161)
values = []
for label, path in rows:
    item = metrics(path)
    values.append((label, item))
    rmse = "[" + ", ".join(f"{v:.4f}" for v in item["rmse"]) + "]"
    nt = f"{item['samples']}/{item['trajectories']}"
    va95 = f"{item['va95_pred']:.4f}/{item['va95_true']:.4f}"
    rpa = f"{item['rpa_pred']:.4f}/{item['rpa_true']:.4f}"
    print(f"{label:<28} {rmse:<24} {item['mean_rmse']:>10.4f} {item['r2_mean']:>10.4f} {item['rbf_mmd']:>12.6f} {va95:>17} {rpa:>17} {item['ade']:>10.4f} {item['fde']:>10.4f} {item['cr']:>8.3f} {nt:>9}")

scratch = values[0][1]["mean_rmse"]
transfer = values[1][1]["mean_rmse"]
print(f"\nTransfer mean RMSE change vs scratch: {(scratch - transfer) / scratch * 100:.1f}%")
print("\nLong-tail PDFs:")
for path in sorted(Path("artifacts/longtail_case_demo").glob("*.pdf")):
    print(path)
PY
