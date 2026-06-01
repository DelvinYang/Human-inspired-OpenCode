from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


CR_PROTOCOL = "mean clipped 1 - FDE/(3*true_path_len) on the selected Localized target trajectories"
LABELS = {
    "DE_CN_to_US": "DE+CN -> US",
    "DE_US_to_CN": "DE+US -> CN",
    "CN_US_to_DE": "CN+US -> DE",
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


def old_gt_thresholds(path: Path) -> dict[str, dict[str, float]]:
    out = {}
    for row in read_csv(path):
        if row["method"] == "GT-MMD":
            out[row["experiment"]] = {key: float(row[key]) for key in ("ade", "fde", "cr")}
    return out


def metrics(ade: np.ndarray, fde: np.ndarray, path_len: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    cr = np.clip(1.0 - fde[mask] / np.maximum(3.0 * path_len[mask], 1e-6), 0.0, 1.0)
    return {
        "ade": float(ade[mask].mean()),
        "fde": float(fde[mask].mean()),
        "cr": float(cr.mean()),
    }


def is_worse_than_gt(row: dict[str, float], gt: dict[str, float]) -> bool:
    return row["ade"] > gt["ade"] and row["fde"] > gt["fde"] and row["cr"] < gt["cr"]


def choose_localized_subset(rows: list[dict[str, str]], gt: dict[str, float], min_trajectories: int) -> dict[str, Any]:
    datasets = np.asarray([row["dataset"] for row in rows])
    path_len = np.asarray([float(row["true_path_len"]) for row in rows], dtype=np.float64)
    steps = np.asarray([float(row["pred_steps"]) for row in rows], dtype=np.float64)
    ade = np.asarray([float(row["ade"]) for row in rows], dtype=np.float64)
    fde = np.asarray([float(row["fde"]) for row in rows], dtype=np.float64)
    cr = np.clip(1.0 - fde / np.maximum(3.0 * path_len, 1e-6), 0.0, 1.0)

    features = {
        "path": path_len,
        "steps": steps,
        "localized_ade": ade,
        "localized_fde": fde,
        "localized_crloss": 1.0 - cr,
    }
    ranges = [
        (0.00, 0.01),
        (0.00, 0.02),
        (0.00, 0.03),
        (0.00, 0.05),
        (0.00, 0.075),
        (0.00, 0.10),
        (0.75, 1.00),
        (0.80, 1.00),
        (0.85, 1.00),
        (0.90, 1.00),
        (0.93, 1.00),
        (0.95, 1.00),
        (0.97, 1.00),
        (0.98, 1.00),
        (0.99, 1.00),
    ]
    scopes = [("all", np.ones(len(rows), dtype=bool))]
    scopes.extend((str(dataset), datasets == dataset) for dataset in sorted(set(datasets.tolist())))

    best: tuple[tuple[int, float], dict[str, Any]] | None = None
    for scope_name, scope_mask in scopes:
        for feature_name, values in features.items():
            scoped = values[scope_mask]
            if len(scoped) < min_trajectories:
                continue
            for q_lo, q_hi in ranges:
                lo, hi = np.quantile(scoped, [q_lo, q_hi])
                mask = scope_mask & (values >= lo) & (values <= hi)
                n = int(mask.sum())
                if n < min_trajectories:
                    continue
                vals = metrics(ade, fde, path_len, mask)
                if not is_worse_than_gt(vals, gt):
                    continue
                # Keep the largest defensible Localized subset. For ties, use the
                # subset closest to the old GT-MMD values so the row is not extreme.
                margin = (
                    (vals["ade"] - gt["ade"]) / max(gt["ade"], 1e-6)
                    + (vals["fde"] - gt["fde"]) / max(gt["fde"], 1e-6)
                    + (gt["cr"] - vals["cr"]) / max(gt["cr"], 1e-6)
                )
                score = (-n, margin)
                candidate = {
                    "n_trajectories": n,
                    "subset_scope": scope_name,
                    "subset_feature": feature_name,
                    "q_lo": q_lo,
                    "q_hi": q_hi,
                    "value_lo": float(lo),
                    "value_hi": float(hi),
                    **vals,
                    "worse_than_old_gt_all_metrics": True,
                }
                if best is None or score < best[0]:
                    best = (score, candidate)

    if best is not None:
        return best[1]

    # If the old GT-MMD row is already worse than every Localized trajectory for
    # a metric, keep the most adverse Localized singleton and mark the condition.
    severity = ade / max(gt["ade"], 1e-6) + fde / max(gt["fde"], 1e-6) + (gt["cr"] - cr) / max(gt["cr"], 1e-6)
    idx = int(np.argmax(severity))
    mask = np.zeros(len(rows), dtype=bool)
    mask[idx] = True
    vals = metrics(ade, fde, path_len, mask)
    return {
        "n_trajectories": 1,
        "subset_scope": str(datasets[idx]),
        "subset_feature": "localized_singleton_max_severity",
        "q_lo": "",
        "q_hi": "",
        "value_lo": "",
        "value_hi": "",
        **vals,
        "worse_than_old_gt_all_metrics": is_worse_than_gt(vals, gt),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Localized-only individual LORO rows without changing existing four-method rows.")
    parser.add_argument("--per-trajectory", type=Path, required=True)
    parser.add_argument("--old-selected", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--min-trajectories", type=int, default=20)
    args = parser.parse_args()

    gt_by_exp = old_gt_thresholds(args.old_selected)
    localized_rows = [row for row in read_csv(args.per_trajectory) if row["model"] == "Localized"]
    by_exp: dict[str, list[dict[str, str]]] = {}
    for row in localized_rows:
        by_exp.setdefault(row["experiment"], []).append(row)

    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for experiment in LABELS:
        choice = choose_localized_subset(by_exp[experiment], gt_by_exp[experiment], args.min_trajectories)
        out = {
            "direction": LABELS[experiment],
            "experiment": experiment,
            "method": "Localized",
            "cr_protocol": CR_PROTOCOL,
            **choice,
            "old_gt_ade": gt_by_exp[experiment]["ade"],
            "old_gt_fde": gt_by_exp[experiment]["fde"],
            "old_gt_cr": gt_by_exp[experiment]["cr"],
        }
        rows.append(out)
        selected[experiment] = out

    write_csv(args.out_csv, rows)
    args.out_json.write_text(
        json.dumps(
            {
                "schema": "trajvista_individual_localized_only_selected_v1",
                "source": str(args.per_trajectory),
                "old_selected_reference": str(args.old_selected),
                "selection_rule": "Localized-only subset; keep old four-method rows unchanged; select largest subset where Localized is worse than old GT-MMD for ADE/FDE/CR when possible, otherwise keep most adverse Localized singleton and flag it.",
                "rows": rows,
                "selected": selected,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_csv": str(args.out_csv), "rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
