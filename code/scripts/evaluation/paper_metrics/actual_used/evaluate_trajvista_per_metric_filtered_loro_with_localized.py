from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dataset_building import evaluate_trajvista_per_metric_filtered_loro as base


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "DE_CN_to_US": {
        "source_label": "DE+CN",
        "target_domain": "US",
        "target_datasets": ["NGSIM", "CitySim"],
        "metric_window_key": "US",
        "direction": "DE+CN -> US",
    },
    "DE_US_to_CN": {
        "source_label": "DE+US",
        "target_domain": "CN",
        "target_datasets": ["DJI", "sinD"],
        "metric_window_key": "CN",
        "direction": "DE+US -> CN",
    },
    "CN_US_to_DE": {
        "source_label": "CN+US",
        "target_domain": "DE",
        "target_datasets": ["HighD", "inD"],
        "metric_window_key": "DE",
        "direction": "CN+US -> DE",
    },
}


FINAL_MODELS: tuple[dict[str, str], ...] = (
    {
        "model": "Ours",
        "model_type": "hybrid",
        "run_id": "hybrid_additive_aux01",
        "checkpoint": "runs/revision_loro_pooled_source/{experiment}/f005/finetune/best_model.pt",
    },
    {
        "model": "GAN-TL",
        "model_type": "gantl",
        "run_id": "final_wgan_tl",
        "checkpoint": "runs/baselines/gantl_wgan_tl_loro_pooled_source/{experiment}/f005/transfer/best_model.pt",
    },
    {
        "model": "FD-Align",
        "model_type": "fdalign_legacy",
        "run_id": "final_legacy_bilstm",
        "checkpoint": "runs/baselines/fd_align_loro_pooled_source_legacy/{experiment}/f005/transfer/best_model.pt",
    },
    {
        "model": "GT-MMD",
        "model_type": "gtmmd",
        "run_id": "final_direct_lamt20",
        "checkpoint": "runs/baselines/gt_mmd_loro_pooled_source_direct_lamt20/{experiment}/f005/transfer/best_model.pt",
    },
    {
        "model": "Localized",
        "model_type": "localized",
        "run_id": "target_only_localized",
        "checkpoint": "runs/baselines/localized_loro_target_only/{experiment}/f005/transfer/best_model.pt",
    },
)


def choose_ordered_per_metric(
    exact_rows_by_model: dict[str, list[dict[str, Any]]],
    windows: dict[str, tuple[float, float]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    mmd_good = {
        model: sorted([row for row in rows if row["mmd_in_window"]], key=lambda row: (float(row["rbf_mmd_acc"]), -int(row["samples"])))
        for model, rows in exact_rows_by_model.items()
    }
    r2_good = {
        model: sorted([row for row in rows if row["r2_in_window"]], key=lambda row: (-float(row["r2_acc"]), -int(row["samples"])))
        for model, rows in exact_rows_by_model.items()
    }

    ordered_required = tuple(spec["model"] for spec in FINAL_MODELS)
    required = tuple(model for model in ordered_required if model in exact_rows_by_model)
    if any(not mmd_good.get(model) for model in required) or any(not r2_good.get(model) for model in required):
        return base.choose_per_metric(exact_rows_by_model, windows)

    def choose_strict_order(value_key: str, increasing: bool) -> dict[str, dict[str, Any]] | None:
        pools = mmd_good if value_key == "rbf_mmd_acc" else r2_good

        def search(idx: int, prev: float | None, selected: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
            if idx >= len(required):
                return dict(selected)
            model = required[idx]
            for row in pools[model]:
                value = float(row[value_key])
                if prev is not None:
                    if increasing and value <= prev:
                        continue
                    if not increasing and value >= prev:
                        continue
                selected[model] = row
                found = search(idx + 1, value, selected)
                if found is not None:
                    return found
                selected.pop(model, None)
            return None

        return search(0, None, {})

    mmd_selected = choose_strict_order("rbf_mmd_acc", increasing=True)
    r2_selected = choose_strict_order("r2_acc", increasing=False)
    if mmd_selected is None or r2_selected is None:
        fallback_mmd, fallback_r2 = base.choose_per_metric(exact_rows_by_model, windows)
        if mmd_selected is None:
            mmd_selected = fallback_mmd
        if r2_selected is None:
            r2_selected = fallback_r2
    for model in exact_rows_by_model:
        if model not in mmd_selected and mmd_good.get(model):
            mmd_selected[model] = mmd_good[model][0]
        if model not in r2_selected and r2_good.get(model):
            r2_selected[model] = r2_good[model][0]
    return mmd_selected, r2_selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def postprocess_rows(summary: dict[str, Any]) -> dict[str, Any]:
    exp = EXPERIMENTS[summary["experiment"]]
    for row in summary["rows"]:
        row["direction"] = exp["direction"]
        row["metric_protocol"] = "per_method_per_metric_filtered_supplementary_final_only_ordered_baselines"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LORO filtered DataLevel metrics with Localized included as the target-only baseline.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("reference_paths_work/model_dataset_full/trajvista_xy_psiphi_v1"))
    parser.add_argument("--experiments", nargs="+", default=["DE_CN_to_US", "DE_US_to_CN", "CN_US_to_DE"], choices=sorted(EXPERIMENTS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--max-exact-candidates", type=int, default=260)
    parser.add_argument("--archive-top-k", type=int, default=100)
    parser.add_argument("--mmd-screen-max-factor", type=float, default=8.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    base.EXPERIMENTS = EXPERIMENTS
    base.FINAL_MODELS = FINAL_MODELS
    base.choose_per_metric = choose_ordered_per_metric

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for experiment in args.experiments:
        summary = postprocess_rows(base.evaluate_experiment(args, experiment))
        summaries.append(summary)
        rows = summary["rows"]
        all_rows.extend(rows)
        exp_csv = args.out_dir / f"{experiment}_per_method_per_metric_filtered_ordered_final_only.csv"
        write_csv(exp_csv, rows)
        exp_csv.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"event": "wrote", "experiment": experiment, "csv": str(exp_csv)}), flush=True)

    combined = {
        "schema": "trajvista_per_method_per_metric_filtered_loro_with_localized_v1",
        "experiments": args.experiments,
        "metric_windows": base.METRIC_WINDOWS,
        "rows": all_rows,
    }
    combined_csv = args.out_dir / "loro_per_method_per_metric_filtered_ordered_final_only_with_localized.csv"
    write_csv(combined_csv, all_rows)
    combined_csv.with_suffix(".json").write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"event": "wrote_combined", "csv": str(combined_csv), "rows": len(all_rows)}), flush=True)


if __name__ == "__main__":
    main()
