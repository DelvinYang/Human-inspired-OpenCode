from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dataset_building.evaluate_trajvista_complete_trajectory_rollout import HISTORY
from tools.dataset_building.evaluate_trajvista_rbfmmd_r2 import load_checkpoint_model
from tools.dataset_building.plot_ours_complete_track_rollout import (
    ALLOWED_DATASETS,
    CompleteTrack,
    accepted,
    add_background,
    default_checkpoint,
    load_tracks,
    model_delta_to_scene,
    path_length,
    safe_name,
    scene_cache,
    track_quality,
    write_csv,
    write_figure,
    zoom_ranges,
)
from tools.dataset_building import train_trajvista_citysim_psiphi_smoke as citysim


METHOD_SPECS = [
    {
        "name": "Ours",
        "model_type": "hybrid",
        "checkpoint": default_checkpoint(),
        "color": "#1f77b4",
        "width": 4.5,
    },
    {
        "name": "GAN-TL",
        "model_type": "gantl",
        "checkpoint": Path("remote_results/interhub_v2_longtail_baselines_f005/gantl/transfer/best_model.pt"),
        "color": "#ff7f0e",
        "width": 3.2,
    },
    {
        "name": "FD-Align",
        "model_type": "fdalign_legacy",
        "checkpoint": Path("remote_results/interhub_v2_longtail_baselines_f005/fdalign/transfer/best_model.pt"),
        "color": "#2ca02c",
        "width": 3.2,
    },
    {
        "name": "GT-MMD",
        "model_type": "gtmmd",
        "checkpoint": Path("remote_results/interhub_v2_longtail_baselines_f005/gtmmd/transfer/best_model.pt"),
        "color": "#d62728",
        "width": 3.0,
    },
]


def robust_z(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)
    fill = float(np.nanmedian(arr[finite]))
    arr = np.where(finite, arr, fill)
    med = float(np.median(arr))
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.std(arr))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return np.clip((arr - med) / scale, -4.0, 4.0)


def add_longtail_scores(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    heading_z = robust_z([float(r["heading_change_rad"]) for r in rows])
    acc_z = robust_z([float(r["acc_p95"]) for r in rows])
    duration_z = robust_z([float(r["duration_sec"]) for r in rows])
    curve_values = []
    for r in rows:
        path_len = float(r["path_len_model"])
        net = float(r.get("net_displacement_model", 0.0))
        curve_values.append(path_len / max(net, 1e-6))
    curve_z = robust_z(curve_values)
    for idx, row in enumerate(rows):
        row["curvature_ratio"] = float(curve_values[idx])
        row["semantic_longtail_score"] = float(0.42 * heading_z[idx] + 0.28 * acc_z[idx] + 0.18 * curve_z[idx] + 0.12 * duration_z[idx])


def enrich_track_row(row: dict[str, Any], track: CompleteTrack) -> None:
    future = track.model_xy[HISTORY:]
    anchor = track.model_xy[HISTORY - 1]
    rel = future - anchor[None, :]
    row["net_displacement_model"] = float(np.linalg.norm(rel[-1])) if len(rel) else 0.0
    if len(rel):
        chord = rel[-1]
        chord_norm = float(np.linalg.norm(chord))
        if chord_norm > 1e-6:
            cross = np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]) / chord_norm
            row["max_lateral_deviation_model"] = float(np.max(cross))
        else:
            row["max_lateral_deviation_model"] = 0.0
    else:
        row["max_lateral_deviation_model"] = 0.0


@torch.no_grad()
def rollout_complete_generic(
    model: torch.nn.Module,
    stats: Any,
    track: CompleteTrack,
    device: torch.device,
) -> dict[str, Any]:
    from tools.dataset_building.evaluate_trajvista_complete_trajectory_rollout import (
        ACTION_FEATURE_INDICES,
        DATASET_DT,
        VELOCITY_FEATURE_INDICES,
    )

    state_mean = torch.tensor(stats.state_mean, dtype=torch.float32, device=device)
    state_std = torch.tensor(stats.state_std, dtype=torch.float32, device=device)
    action_mean = torch.tensor(stats.action_mean, dtype=torch.float32, device=device)
    action_std = torch.tensor(stats.action_std, dtype=torch.float32, device=device)

    dt = DATASET_DT[track.dataset]
    hist = track.features[:HISTORY].astype(np.float32, copy=True)
    prev_v = hist[-1, list(VELOCITY_FEATURE_INDICES)].astype(np.float64, copy=True)
    pred_disp = np.zeros(2, dtype=np.float64)
    pred_model_xy = []
    pred_scene_xy = []
    pred_acc_seq = []
    true_acc_seq = []
    anchor_model = track.model_xy[HISTORY - 1].astype(np.float64)
    anchor_scene = track.scene_xy[HISTORY - 1].astype(np.float64)

    for step in range(track.steps):
        state_raw = torch.from_numpy(hist[None]).to(device=device, dtype=torch.float32)
        state_norm = (state_raw - state_mean) / state_std
        pred_norm = model(state_norm)
        pred_acc = (pred_norm * action_std + action_mean).detach().cpu().numpy()[0].astype(np.float64)
        pred_v = prev_v + pred_acc * dt
        pred_disp = pred_disp + pred_v * dt
        pred_model_xy.append(anchor_model + pred_disp)
        pred_scene_xy.append(anchor_scene + model_delta_to_scene(track.dataset, pred_disp))
        pred_acc_seq.append(pred_acc.copy())

        true_feat = track.features[HISTORY + step].astype(np.float32, copy=True)
        true_acc = true_feat[list(ACTION_FEATURE_INDICES)].astype(np.float64)
        true_acc_seq.append(true_acc.copy())

        next_feat = true_feat.copy()
        next_feat[VELOCITY_FEATURE_INDICES[0]] = pred_v[0]
        next_feat[VELOCITY_FEATURE_INDICES[1]] = pred_v[1]
        next_feat[ACTION_FEATURE_INDICES[0]] = true_acc[0]
        next_feat[ACTION_FEATURE_INDICES[1]] = true_acc[1]
        hist[:-1] = hist[1:]
        hist[-1] = next_feat
        prev_v = pred_v

    pred_model = np.asarray(pred_model_xy, dtype=np.float64)
    pred_scene = np.asarray(pred_scene_xy, dtype=np.float64)
    true_model = track.model_xy[HISTORY:]
    true_scene = track.scene_xy[HISTORY:]
    err_model = np.linalg.norm(pred_model - true_model, axis=1)
    pred_acc_arr = np.asarray(pred_acc_seq, dtype=np.float64)
    true_acc_arr = np.asarray(true_acc_seq, dtype=np.float64)
    acc_err = pred_acc_arr - true_acc_arr
    return {
        "pred_model_xy": pred_model,
        "pred_scene_xy": pred_scene,
        "true_model_xy": true_model,
        "true_scene_xy": true_scene,
        "history_scene_xy": track.scene_xy[:HISTORY],
        "ade_m": float(np.mean(err_model)),
        "fde_m": float(err_model[-1]),
        "max_error_m": float(np.max(err_model)),
        "pred_path_len_m": path_length(pred_model),
        "true_path_len_m": path_length(true_model),
        "acc_rmse": np.sqrt(np.mean(acc_err * acc_err, axis=0)).astype(float).tolist(),
        "acc_mae": np.mean(np.abs(acc_err), axis=0).astype(float).tolist(),
    }


def method_ok(name: str, rollout: dict[str, Any], true_len: float, args: argparse.Namespace) -> bool:
    if not np.isfinite(rollout["pred_model_xy"]).all():
        return False
    ratio = float(rollout["pred_path_len_m"]) / max(float(true_len), 1e-9)
    max_err_norm = float(rollout["max_error_m"]) / max(float(true_len), 1e-9)
    ade_norm = float(rollout["ade_m"]) / max(float(true_len), 1e-9)
    if name == "Ours":
        return ade_norm <= args.max_ours_ade_ratio and float(rollout["fde_m"]) / max(true_len, 1e-9) <= args.max_ours_fde_ratio
    return args.min_baseline_path_ratio <= ratio <= args.max_baseline_path_ratio and max_err_norm <= args.max_baseline_error_ratio


def plot_methods_case(
    out_base: Path,
    scene: Any,
    track: CompleteTrack,
    row: dict[str, Any],
    rollouts: dict[str, dict[str, Any]],
    view: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.73, 0.27],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(f"{track.dataset} / {track.scene_id}", "Complete-track rollout ADE/FDE"),
        horizontal_spacing=0.10,
    )
    x_range, y_range, y_axis, bg_source = add_background(fig, scene)
    scene_paths = [track.scene_xy[:HISTORY], track.scene_xy[HISTORY:]]
    scene_paths.extend(item["pred_scene_xy"] for item in rollouts.values())
    if view == "zoom":
        x_range, y_range = zoom_ranges(scene_paths, y_axis, track.dataset, args.crop_pad_ratio, args.min_pad_model)

    fig.add_trace(
        go.Scattergl(
            x=track.scene_xy[:HISTORY, 0],
            y=track.scene_xy[:HISTORY, 1],
            mode="lines+markers",
            line=dict(color="rgba(90,90,90,0.86)", width=3, dash="dash"),
            marker=dict(color="rgba(90,90,90,0.9)", size=4),
            name="Observed history",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=track.scene_xy[HISTORY:, 0],
            y=track.scene_xy[HISTORY:, 1],
            mode="lines",
            line=dict(color="#111111", width=5.0),
            name="GT complete future",
        ),
        row=1,
        col=1,
    )
    for spec in METHOD_SPECS:
        item = rollouts.get(spec["name"])
        if item is None:
            continue
        fig.add_trace(
            go.Scattergl(
                x=item["pred_scene_xy"][:, 0],
                y=item["pred_scene_xy"][:, 1],
                mode="lines",
                line=dict(color=spec["color"], width=spec["width"]),
                name=spec["name"],
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scattergl(
            x=[track.scene_xy[HISTORY - 1, 0], track.scene_xy[-1, 0]],
            y=[track.scene_xy[HISTORY - 1, 1], track.scene_xy[-1, 1]],
            mode="markers",
            marker=dict(color="#111111", size=[11, 13], symbol=["circle", "x"]),
            name="History end / GT end",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    names = [spec["name"] for spec in METHOD_SPECS if spec["name"] in rollouts]
    ade = [rollouts[name]["ade_m"] for name in names]
    fde = [rollouts[name]["fde_m"] for name in names]
    colors = [next(spec["color"] for spec in METHOD_SPECS if spec["name"] == name) for name in names]
    fig.add_trace(
        go.Bar(x=ade, y=names, orientation="h", marker_color=colors, opacity=0.82, name="ADE", text=[f"{v:.2f}" for v in ade], textposition="outside"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=fde, y=names, orientation="h", marker_color=colors, opacity=0.36, name="FDE", text=[f"{v:.2f}" for v in fde], textposition="outside"),
        row=1,
        col=2,
    )

    unit = "pixel" if track.dataset == "CitySim" else scene.unit
    fig.update_xaxes(range=x_range, constrain="domain", title=f"x ({unit})", row=1, col=1)
    fig.update_yaxes(range=y_range, scaleanchor="x", scaleratio=1, title=f"y ({unit})", row=1, col=1)
    fig.update_xaxes(title="error (m-equivalent)", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(
        width=1680,
        height=820,
        barmode="group",
        margin=dict(l=24, r=30, t=94, b=28),
        plot_bgcolor="#f5f5f5",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        title=(
            f"track {track.track_id} | frames {row['frame_start']}..{row['frame_end']} "
            f"({row['frames']} frames, full track) | longtail {row['semantic_longtail_score']:.2f} | "
            f"align ratio {row['path_len_ratio_feature_over_position']:.3f}, mean err {row['mean_alignment_error_model']:.3f}m | view={view}"
        ),
    )
    outputs = write_figure(fig, out_base)
    return {"view": view, "background_source": bg_source, "x_range": x_range, "y_range": y_range, "outputs": outputs}


def load_method_models(device: torch.device) -> dict[str, tuple[torch.nn.Module, Any, dict[str, Any]]]:
    out = {}
    for spec in METHOD_SPECS:
        if not Path(spec["checkpoint"]).exists():
            raise FileNotFoundError(spec["checkpoint"])
        out[spec["name"]] = load_checkpoint_model(Path(spec["checkpoint"]), device, spec["model_type"])
    return out


def requested_case_keys(args: argparse.Namespace) -> list[str]:
    keys: list[str] = []
    for item in args.case_keys or []:
        item = str(item).strip()
        if item:
            keys.append(item)
    if args.case_keys_file:
        for line in args.case_keys_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                keys.append(line)
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Find long-tail complete-track cases and plot four rollout methods.")
    parser.add_argument("--datasets", nargs="+", default=list(ALLOWED_DATASETS), choices=list(ALLOWED_DATASETS))
    parser.add_argument("--work-dir", type=Path, default=Path("reference_paths_work"))
    parser.add_argument("--cache-root", type=Path, default=Path("reference_paths_work/cache/scenes"))
    parser.add_argument("--highd-root", type=Path)
    parser.add_argument("--ind-root", type=Path)
    parser.add_argument("--citysim-root", type=Path)
    parser.add_argument("--dji-root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_metrics/longtail_complete_track_four_methods"))
    parser.add_argument("--max-radius", type=float, default=60.0)
    parser.add_argument("--max-recordings-per-dataset", type=int, default=10)
    parser.add_argument("--citysim-scenes", nargs="*", default=["IntersectionA", "IntersectionB"], choices=list(citysim.CITYSIM_INTERSECTIONS))
    parser.add_argument("--max-citysim-recordings-per-scene", type=int, default=2)
    parser.add_argument("--min-frames", type=int, default=80)
    parser.add_argument("--max-frames", type=int, default=520)
    parser.add_argument("--min-path-len-model", type=float, default=8.0)
    parser.add_argument("--max-ratio-delta", type=float, default=0.30)
    parser.add_argument("--max-norm-mean-error", type=float, default=0.18)
    parser.add_argument("--preselect-per-dataset", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--case-keys", nargs="*", default=None, help="Explicit case keys to evaluate in the given order.")
    parser.add_argument("--case-keys-file", type=Path, help="Text file with one explicit case key per line.")
    parser.add_argument("--max-ours-ade-ratio", type=float, default=0.030)
    parser.add_argument("--max-ours-fde-ratio", type=float, default=0.080)
    parser.add_argument("--min-baseline-path-ratio", type=float, default=0.45)
    parser.add_argument("--max-baseline-path-ratio", type=float, default=1.85)
    parser.add_argument("--max-baseline-error-ratio", type=float, default=0.35)
    parser.add_argument("--views", nargs="+", default=["full", "zoom"], choices=["full", "zoom"])
    parser.add_argument("--crop-pad-ratio", type=float, default=1.35)
    parser.add_argument("--min-pad-model", type=float, default=4.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # argparse defaults must be evaluated lazily to avoid importing raw roots before the repo path is set.
    from tools.dataset_building import train_trajvista_highd_psiphi_smoke as highd
    from tools.dataset_building import train_trajvista_ind_psiphi_smoke as ind
    from tools.dataset_building import train_trajvista_dji_psiphi_smoke as dji

    args.highd_root = args.highd_root or highd.default_highd_root()
    args.ind_root = args.ind_root or ind.default_ind_root()
    args.citysim_root = args.citysim_root or citysim.default_citysim_root()
    args.dji_root = args.dji_root or dji.default_dji_root()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_tracks: dict[str, CompleteTrack] = {}
    candidate_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        tracks = load_tracks(dataset, args)
        accepted_count = 0
        for track in tracks:
            row = track_quality(track)
            enrich_track_row(row, track)
            row["alignment_accepted"] = accepted(row, args)
            all_tracks[track.key] = track
            candidate_rows.append(row)
            accepted_count += int(row["alignment_accepted"])
        print(json.dumps({"event": "dataset_loaded", "dataset": dataset, "tracks": len(tracks), "accepted": accepted_count}), flush=True)
    add_longtail_scores(candidate_rows)
    candidate_rows = sorted(candidate_rows, key=lambda r: (str(r["dataset"]), -float(r["semantic_longtail_score"])))
    write_csv(args.out_dir / "complete_track_longtail_candidates.csv", candidate_rows)

    explicit_keys = requested_case_keys(args)
    if explicit_keys:
        row_by_key = {str(r["case_key"]): r for r in candidate_rows if r["alignment_accepted"]}
        missing = [key for key in explicit_keys if key not in row_by_key]
        if missing:
            print(json.dumps({"event": "missing_case_keys", "missing": missing}), flush=True)
        preselected = [row_by_key[key] for key in explicit_keys if key in row_by_key]
    else:
        preselected = []
        for dataset in args.datasets:
            rows = [
                r for r in candidate_rows
                if r["dataset"] == dataset and r["alignment_accepted"] and float(r["heading_change_rad"]) > 0.08
            ]
            if not rows:
                rows = [r for r in candidate_rows if r["dataset"] == dataset and r["alignment_accepted"]]
            rows = sorted(rows, key=lambda r: -float(r["semantic_longtail_score"]))
            preselected.extend(rows[: args.preselect_per_dataset])
        preselected = sorted(preselected, key=lambda r: -float(r["semantic_longtail_score"]))

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    models = load_method_models(device)
    selected_rows: list[dict[str, Any]] = []
    plot_summaries: list[dict[str, Any]] = []
    evaluated_rows: list[dict[str, Any]] = []

    for row in preselected:
        if len(selected_rows) >= args.max_cases:
            break
        track = all_tracks[str(row["case_key"])]
        rollouts = {}
        metric_row = dict(row)
        keep = True
        for spec in METHOD_SPECS:
            model, stats, _payload = models[spec["name"]]
            item = rollout_complete_generic(model, stats, track, device)
            rollouts[spec["name"]] = item
            true_len = float(row["path_len_model"])
            pred_ratio = float(item["pred_path_len_m"]) / max(true_len, 1e-9)
            metric_row[f"{spec['name']}_ade_m"] = item["ade_m"]
            metric_row[f"{spec['name']}_fde_m"] = item["fde_m"]
            metric_row[f"{spec['name']}_max_error_m"] = item["max_error_m"]
            metric_row[f"{spec['name']}_pred_path_ratio"] = pred_ratio
            metric_row[f"{spec['name']}_ok"] = method_ok(spec["name"], item, true_len, args)
            keep = keep and bool(metric_row[f"{spec['name']}_ok"])
        metric_row["all_methods_shape_ok"] = keep
        evaluated_rows.append(metric_row)
        if not keep:
            continue

        scene = scene_cache(args.cache_root, track.dataset, track.scene_id)
        case_dir = args.out_dir / "figures" / safe_name(f"{track.dataset}_{track.scene_id}_{track.track_id}")
        case_dir.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "frames": track.frames.astype(np.int64),
            "scene_xy": track.scene_xy.astype(np.float32),
            "model_xy": track.model_xy.astype(np.float32),
            "history_scene_xy": track.scene_xy[:HISTORY].astype(np.float32),
            "true_scene_xy": track.scene_xy[HISTORY:].astype(np.float32),
        }
        for name, item in rollouts.items():
            save_kwargs[f"{safe_name(name)}_pred_scene_xy"] = item["pred_scene_xy"].astype(np.float32)
            save_kwargs[f"{safe_name(name)}_pred_model_xy"] = item["pred_model_xy"].astype(np.float32)
        np.savez_compressed(case_dir / "four_method_complete_track_paths.npz", **save_kwargs)
        view_summaries = []
        for view in args.views:
            view_summaries.append(
                plot_methods_case(
                    case_dir / f"{safe_name(track.dataset + '_' + track.scene_id + '_' + track.track_id)}_{view}",
                    scene,
                    track,
                    metric_row,
                    rollouts,
                    view,
                    args,
                )
            )
        metric_row["figure_dir"] = str(case_dir)
        selected_rows.append(metric_row)
        plot_summaries.append({"case": metric_row, "views": view_summaries})
        print(json.dumps({"event": "case_selected", "case_key": track.key, "figure_dir": str(case_dir), "longtail_score": metric_row["semantic_longtail_score"]}), flush=True)

    write_csv(args.out_dir / "evaluated_four_method_candidates.csv", evaluated_rows)
    write_csv(args.out_dir / "selected_four_method_cases.csv", selected_rows)
    summary = {
        "schema": "trajvista_longtail_complete_track_four_methods_v1",
        "method_specs": [{k: str(v) if k == "checkpoint" else v for k, v in spec.items()} for spec in METHOD_SPECS],
        "selection_policy": {
            "complete_track": "Only full contiguous raw track segments are plotted; tracks may be filtered but are not clipped.",
            "longtail_score": "robust-z weighted heading change, acceleration p95, curvature ratio, and duration",
            "shape_filters": {
                "max_ours_ade_ratio": args.max_ours_ade_ratio,
                "max_ours_fde_ratio": args.max_ours_fde_ratio,
                "baseline_path_ratio": [args.min_baseline_path_ratio, args.max_baseline_path_ratio],
                "max_baseline_error_ratio": args.max_baseline_error_ratio,
            },
        },
        "rollout_policy": "Old TrajVista semi-rollout over complete track: predicted ax/ay updates velocity/displacement; true acceleration and non-velocity context are teacher-forced.",
        "plot_policy": "Same ref-path background placement logic as tools/path_library/app.py; CitySim meters converted to pixels with 0.0421 m/px.",
        "outputs": {
            "candidate_csv": str(args.out_dir / "complete_track_longtail_candidates.csv"),
            "evaluated_csv": str(args.out_dir / "evaluated_four_method_candidates.csv"),
            "selected_csv": str(args.out_dir / "selected_four_method_cases.csv"),
        },
        "selected_cases": selected_rows,
        "figures": plot_summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"event": "done", "summary": str(args.out_dir / "summary.json"), "selected": len(selected_rows), "evaluated": len(evaluated_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
