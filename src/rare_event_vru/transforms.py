from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LocalFrame:
    origin: np.ndarray
    angle: float


def _heading(points: np.ndarray, eps: float = 1e-8) -> float:
    deltas = np.diff(points, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    valid = np.flatnonzero(norms > eps)
    if valid.size == 0:
        return 0.0
    dx, dy = deltas[valid[-1]]
    return float(np.arctan2(dy, dx))


def to_local(points: np.ndarray, observed_steps: int) -> tuple[np.ndarray, LocalFrame]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (T, 2)")
    if not 2 <= observed_steps <= len(points):
        raise ValueError("observed_steps must be in [2, T]")
    observed = points[:observed_steps]
    origin = observed[-1].copy()
    angle = _heading(observed)
    c, s = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (points - origin) @ rotation.T, LocalFrame(origin=origin, angle=angle)


def to_global(local_points: np.ndarray, frame: LocalFrame) -> np.ndarray:
    local_points = np.asarray(local_points, dtype=np.float32)
    c, s = np.cos(frame.angle), np.sin(frame.angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local_points @ rotation.T + frame.origin
