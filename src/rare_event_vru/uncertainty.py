from __future__ import annotations

import numpy as np


def ensemble_decomposition(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 4:
        raise ValueError("predictions must have shape (ensemble, batch, time, 2)")
    return predictions.mean(axis=0), predictions.var(axis=0)


def gaussian_nll(mean: np.ndarray, scale: np.ndarray, target: np.ndarray) -> float:
    mean = np.asarray(mean)
    scale = np.maximum(np.asarray(scale), 1e-6)
    target = np.asarray(target)
    return float(np.mean(0.5 * ((target - mean) / scale) ** 2 + np.log(scale)))
