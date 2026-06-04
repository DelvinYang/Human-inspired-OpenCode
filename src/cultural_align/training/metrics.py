from __future__ import annotations

from typing import Any

import numpy as np


def rbf_mmd(x: np.ndarray, y: np.ndarray, max_samples: int = 2048, seed: int = 42) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(x) > max_samples:
        x = x[np.sort(rng.choice(len(x), size=max_samples, replace=False))]
    if len(y) > max_samples:
        y = y[np.sort(rng.choice(len(y), size=max_samples, replace=False))]
    pooled = np.concatenate([x, y], axis=0)
    if len(pooled) < 2:
        return 0.0
    diffs = pooled[: min(len(pooled), 512), None, :] - pooled[None, : min(len(pooled), 512), :]
    dist = np.sum(diffs * diffs, axis=-1)
    positive = dist[dist > 1e-12]
    sigma2 = float(np.median(positive)) if positive.size else 1.0
    sigma2 = max(sigma2, 1e-6)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-d / (2.0 * sigma2))

    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean())


def regression_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mmd_max_samples: int = 2048,
) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    err = pred - target
    ss_res = np.sum(err * err, axis=0)
    ss_tot = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    out: dict[str, Any] = {
        "raw_rmse": np.sqrt(np.mean(err * err, axis=0)).astype(float).tolist(),
        "raw_mae": np.mean(np.abs(err), axis=0).astype(float).tolist(),
        "raw_rmse_mean": float(np.sqrt(np.mean(err * err, axis=0)).mean()),
        "raw_mae_mean": float(np.mean(np.abs(err), axis=0).mean()),
        "r2": r2.astype(float).tolist(),
        "r2_mean": float(r2.mean()),
        "rbf_mmd": rbf_mmd(pred, target, max_samples=mmd_max_samples),
        "target_std": target.std(axis=0).astype(float).tolist(),
        "samples": int(len(target)),
    }
    return out
