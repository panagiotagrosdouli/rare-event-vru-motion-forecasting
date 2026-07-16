from __future__ import annotations

import numpy as np


def fit_endpoint_radius(predicted: np.ndarray, target: np.ndarray, coverage: float = 0.9) -> float:
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")
    errors = np.linalg.norm(np.asarray(predicted)[:, -1] - np.asarray(target)[:, -1], axis=-1)
    rank = min(len(errors) - 1, int(np.ceil((len(errors) + 1) * coverage)) - 1)
    return float(np.sort(errors)[rank])


def empirical_endpoint_coverage(predicted: np.ndarray, target: np.ndarray, radius: float) -> float:
    errors = np.linalg.norm(np.asarray(predicted)[:, -1] - np.asarray(target)[:, -1], axis=-1)
    return float(np.mean(errors <= radius))
