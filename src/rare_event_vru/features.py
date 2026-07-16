from __future__ import annotations

import numpy as np


def extract_motion_features(points: np.ndarray, dt: float = 0.1) -> dict[str, float]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError("points must have shape (T, 2), T >= 4")
    velocity = np.diff(points, axis=0) / dt
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0) / dt
    tangential = np.diff(speed) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    headings = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    heading_change = np.abs(np.diff(headings))
    curvature = heading_change / np.maximum(speed[1:] * dt, 1e-6)
    return {
        "max_speed": float(speed.max(initial=0.0)),
        "max_acceleration": float(np.linalg.norm(acceleration, axis=1).max(initial=0.0)),
        "max_deceleration": float(max(0.0, -tangential.min(initial=0.0))),
        "max_jerk": float(np.linalg.norm(jerk, axis=1).max(initial=0.0)),
        "max_heading_change": float(heading_change.max(initial=0.0)),
        "max_curvature": float(curvature.max(initial=0.0)),
        "total_displacement": float(np.linalg.norm(points[-1] - points[0])),
    }
