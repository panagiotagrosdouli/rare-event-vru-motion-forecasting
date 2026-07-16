from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ("pedestrian", "cyclist", "motorcyclist")


def make_fixture(seed: int = 0, samples_per_class: int = 24, steps: int = 110) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    trajectories: list[np.ndarray] = []
    rows: list[dict[str, str]] = []
    for class_index, object_type in enumerate(CLASSES):
        base_speed = 0.08 + 0.05 * class_index
        for index in range(samples_per_class):
            velocity = np.array([base_speed, 0.0])
            points = [np.zeros(2)]
            rare = index >= samples_per_class - 3
            for timestep in range(1, steps):
                if rare and timestep > 55:
                    angle = 0.025 * (timestep - 55) * (1 if index % 2 else -1)
                    rotation = np.array(
                        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                    )
                    velocity = rotation @ velocity * 1.002
                points.append(points[-1] + velocity + rng.normal(0, 0.002, 2))
            trajectories.append(np.asarray(points, dtype=np.float32))
            rows.append(
                {
                    "scenario_id": f"fixture-{object_type}-{index}",
                    "object_type": object_type,
                }
            )
    return np.stack(trajectories), pd.DataFrame(rows)
