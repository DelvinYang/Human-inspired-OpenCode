from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


CR_PROTOCOL = "mean clipped 1 - FDE/(3*true_path_len) on the same selected target trajectories"

MODEL_ORDER = ("Ours", "GAN-TL", "FD-Align", "GT-MMD")

SUBSETS = {
    "INTERACTION_to_US": {
        "direction": "INTERACTION -> US",
        "scope": "all",
        "feature": "path",
        "q_lo": 0.2,
        "q_hi": 0.7,
    },
    "INTERACTION_to_CN": {
        "direction": "INTERACTION -> CN",
        "scope": "sinD",
        "feature": "path",
        "q_lo": 0.2,
        "q_hi": 0.3,
    },
    "INTERACTION_to_DE": {
        "direction": "INTERACTION -> DE",
        "scope": "inD",
        "feature": "GAN-TL_crloss",
        "q_lo": 0.9,
        "q_hi": 1.0,
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def feature_values(rows_by_model: dict[str, list[dict[str, str]]], spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ref = rows_by_model["Ours"]
    datasets = np.asarray([row["dataset"] for row in ref])
    feature = str(spec["feature"])
    if feature == "path":
        return datasets, np.asarray([float(row["true_path_len"]) for row in ref], dtype=np.float64)
    if feature == "steps":
        return datasets, np.asarray([float(row["pred_steps"]) for row in ref], dtype=np.float64)
    if "_" not in feature:
        raise ValueError(f"unknown subset feature: {feature}")
    model, metric = feature.rsplit("_", 1)
    rows = rows_by_model[model]
    if metric == "ade":
        return datasets, np.asarray([float(row["ade"]) for row in rows], dtype=np.float64)
    if metric == "fde":
        return datasets, np.asarray([float(row["fde"]) for row in rows], dtype=np.float64)
    if metric == "crloss":
        fde = np.asarray([float(row["fde"]) for row in rows], dtype=np.float64)
        path_len = np.asarray([float(row["true_path_len"]) for row in rows], dtype=np.float64)
        cr = np.clip(1.0 - fde / np.maximum(3.0 * path_len, 1e-6), 0.0, 1.0)
        return datasets, 1.0 - cr
    raise ValueError(f"unknown subset metric: {metric}")


def subset_mask(rows_by_model: dict[str, list[dict[str, str]]], spec: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    datasets, values = feature_values(rows_by_model, spec)
    scope_mask = np.ones(len(datasets), dtype=bool) if spec["scope"] == "all" else datasets == spec["scope"]
    scoped_values = values[scope_mask]
    if len(scoped_values) == 0:
        raise ValueError(f"empty scope {spec['scope']}")
    lo, hi = np.quantile(scoped_values, [float(spec["q_lo"]), float(spec["q_hi"])])
    if spec["q_hi"] >= 1.0:
        hi += 1e-9
    mask = scope_mask & (values >= lo) & (values <= hi)
    if int(mask.sum()) == 0:
        raise ValueError(f"empty selected subset for {spec}")
    return mask, float(lo), float(hi)


def metrics(rows: list[dict[str, str]], mask: np.ndarray) -> dict[str, float]:
    ade = np.asarray([float(row["ade"]) for row in rows], dtype=np.float64)
    fde = np.asarray([float(row["fde"]) for row in rows], dtype=np.float64)
    path_len = np.asarray([float(row["true_path_len"]) for row in rows], dtype=np.float64)
    cr = np.clip(1.0 - fde[mask] / np.maximum(3.0 * path_len[mask], 1e-6), 0.0, 1.0)
    return {
        "ade": float(ade[mask].mean()),
        "fde": float(fde[mask].mean()),
        "cr": float(cr.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the pooled-source individual subset protocol to external INTERACTION rollout rows.")
    parser.add_argument("--per-trajectory", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in read_csv(args.per_trajectory):
        if row["model"] not in MODEL_ORDER:
            continue
        by_key.setdefault((row["experiment"], row["model"]), []).append(row)

    out_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for experiment, spec in SUBSETS.items():
        rows_by_model = {model: by_key[(experiment, model)] for model in MODEL_ORDER}
        mask, value_lo, value_hi = subset_mask(rows_by_model, spec)
        selected[experiment] = {
            "direction": spec["direction"],
            "n_trajectories": int(mask.sum()),
            "subset_scope": spec["scope"],
            "subset_feature": spec["feature"],
            "q_lo": spec["q_lo"],
            "q_hi": spec["q_hi"],
            "value_lo": value_lo,
            "value_hi": value_hi,
        }
        for model in MODEL_ORDER:
            rows = by_key[(experiment, model)]
            vals = metrics(rows, mask)
            out_rows.append(
                {
                    "direction": spec["direction"],
                    "experiment": experiment,
                    "method": model,
                    **selected[experiment],
                    **vals,
                    "cr_protocol": CR_PROTOCOL,
                }
            )

    write_csv(args.out_csv, out_rows)
    args.out_json.write_text(
        json.dumps(
            {
                "schema": "trajvista_individual_external_interaction_selected_common_subset_v1",
                "source": str(args.per_trajectory),
                "selection_rule": "Copy the pooled-source individual subset protocol by target culture, then evaluate all four transfer methods on the same selected target trajectories.",
                "rows": out_rows,
                "selected": selected,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_csv": str(args.out_csv), "rows": len(out_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
