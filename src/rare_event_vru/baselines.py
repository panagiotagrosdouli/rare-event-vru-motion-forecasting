from __future__ import annotations

import numpy as np


def constant_position(past: np.ndarray, future_steps: int) -> np.ndarray:
    return np.repeat(np.asarray(past)[-1][None, :], future_steps, axis=0)


def constant_velocity(past: np.ndarray, future_steps: int) -> np.ndarray:
    past = np.asarray(past)
    velocity = past[-1] - past[-2]
    return past[-1] + np.arange(1, future_steps + 1)[:, None] * velocity


def constant_acceleration(past: np.ndarray, future_steps: int) -> np.ndarray:
    past = np.asarray(past)
    velocity = past[-1] - past[-2]
    acceleration = velocity - (past[-2] - past[-3])
    t = np.arange(1, future_steps + 1)[:, None]
    return past[-1] + t * velocity + 0.5 * (t**2) * acceleration
