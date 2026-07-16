from __future__ import annotations

import numpy as np


def ade_fde(predicted: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(np.asarray(predicted) - np.asarray(target), axis=-1)
    return distance.mean(axis=-1), distance[..., -1]


def minade_minfde(predictions: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(np.asarray(predictions) - np.asarray(target)[:, None], axis=-1)
    return distance.mean(axis=-1).min(axis=1), distance[..., -1].min(axis=1)
