from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.beam_codebook import UniformBeamCodebook


@dataclass(frozen=True)
class TailRiskTopKPolicy:
    """Map predicted rare-event probability to beam-search width."""

    thresholds: tuple[float, float, float] = (0.20, 0.50, 0.80)
    k_values: tuple[int, int, int, int] = (1, 2, 3, 5)

    def __post_init__(self) -> None:
        if tuple(sorted(self.thresholds)) != self.thresholds:
            raise ValueError("thresholds must be sorted")
        if any(not 0.0 <= value <= 1.0 for value in self.thresholds):
            raise ValueError("thresholds must lie in [0, 1]")
        if any(value < 1 for value in self.k_values):
            raise ValueError("all k_values must be positive")

    def choose_k(self, tail_probability: float) -> int:
        probability = float(np.clip(tail_probability, 0.0, 1.0))
        for threshold, k in zip(self.thresholds, self.k_values):
            if probability < threshold:
                return k
        return self.k_values[-1]

    def select(
        self,
        codebook: UniformBeamCodebook,
        predicted_angle_rad: float,
        tail_probability: float,
    ) -> list[int]:
        center = int(codebook.angle_to_index(predicted_angle_rad))
        k = min(self.choose_k(tail_probability), codebook.num_beams)
        return codebook.neighbouring_indices(center, k)
