from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)


def _gaussian_kernel(x: np.ndarray, y: np.ndarray, bandwidths: Sequence[float]) -> np.ndarray:
    dists = _pairwise_sq_dists(x.astype(np.float64), y.astype(np.float64))
    kernel = np.zeros_like(dists, dtype=np.float64)
    for bandwidth in bandwidths:
        gamma = 1.0 / (2.0 * (float(bandwidth) ** 2))
        kernel += np.exp(-gamma * dists)
    return kernel / float(len(bandwidths))


def _paired_rbf_values(x: np.ndarray, y: np.ndarray, bandwidths: Sequence[float]) -> np.ndarray:
    dists = np.sum((x.astype(np.float64) - y.astype(np.float64)) ** 2, axis=-1)
    values = np.zeros_like(dists, dtype=np.float64)
    for bandwidth in bandwidths:
        gamma = 1.0 / (2.0 * (float(bandwidth) ** 2))
        values += np.exp(-gamma * dists)
    return values / float(len(bandwidths))


def rbf_mmd2(
    x: np.ndarray,
    y: np.ndarray,
    bandwidths: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    exact_threshold: int = 4096,
    fallback_sample: int = 2048,
    seed: int = 42,
    eps: float = 1e-12,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape: {x.shape} vs {y.shape}")
    n = int(x.shape[0])
    if n < 2:
        return 0.0
    if n <= exact_threshold:
        k_xx = _gaussian_kernel(x, x, bandwidths)
        k_yy = _gaussian_kernel(y, y, bandwidths)
        k_xy = _gaussian_kernel(x, y, bandwidths)
        mmd_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        mmd_yy = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
        mmd = float(mmd_xx + mmd_yy - 2.0 * k_xy.mean())
        if mmd < -eps:
            mmd = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
        return float(max(mmd, 0.0))

    if n % 2 == 1:
        x = x[:-1]
        y = y[:-1]
        n -= 1
    x_a = x[0::2]
    x_b = x[1::2]
    y_a = y[0::2]
    y_b = y[1::2]
    paired = (
        _paired_rbf_values(x_a, x_b, bandwidths)
        + _paired_rbf_values(y_a, y_b, bandwidths)
        - _paired_rbf_values(x_a, y_b, bandwidths)
        - _paired_rbf_values(x_b, y_a, bandwidths)
    )
    mmd = float(2.0 * paired.mean())
    if mmd < -eps:
        rng = np.random.default_rng(seed)
        subset = min(n, int(fallback_sample))
        idx = np.sort(rng.choice(n, size=subset, replace=False))
        x_sub = x[idx]
        y_sub = y[idx]
        k_xx = _gaussian_kernel(x_sub, x_sub, bandwidths)
        k_yy = _gaussian_kernel(y_sub, y_sub, bandwidths)
        k_xy = _gaussian_kernel(x_sub, y_sub, bandwidths)
        mmd = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    return float(max(mmd, 0.0))


def rbf_mmd(x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
    return float(np.sqrt(rbf_mmd2(x, y, **kwargs)))


def regression_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    err = pred - target
    ss_res = np.sum(err * err, axis=0)
    ss_tot = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    mmd2 = rbf_mmd2(pred, target)
    out: dict[str, Any] = {
        "raw_rmse": np.sqrt(np.mean(err * err, axis=0)).astype(float).tolist(),
        "raw_mae": np.mean(np.abs(err), axis=0).astype(float).tolist(),
        "raw_rmse_mean": float(np.sqrt(np.mean(err * err, axis=0)).mean()),
        "raw_mae_mean": float(np.mean(np.abs(err), axis=0).mean()),
        "r2": r2.astype(float).tolist(),
        "r2_mean": float(r2.mean()),
        "rbf_mmd": float(np.sqrt(mmd2)),
        "rbf_mmd2": float(mmd2),
        "target_std": target.std(axis=0).astype(float).tolist(),
        "samples": int(len(target)),
    }
    return out
