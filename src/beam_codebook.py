from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UniformBeamCodebook:
    """Uniform azimuth codebook over a configurable field of view.

    Angles are expressed in radians. Samples outside the field of view are
    clipped to the nearest edge beam, which models a practical fallback.
    """

    num_beams: int = 32
    min_angle_deg: float = -90.0
    max_angle_deg: float = 90.0

    def __post_init__(self) -> None:
        if self.num_beams < 2:
            raise ValueError("num_beams must be at least 2")
        if self.min_angle_deg >= self.max_angle_deg:
            raise ValueError("min_angle_deg must be smaller than max_angle_deg")

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(
            np.deg2rad(self.min_angle_deg),
            np.deg2rad(self.max_angle_deg),
            self.num_beams + 1,
            dtype=np.float64,
        )

    @property
    def centers(self) -> np.ndarray:
        edges = self.edges
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def beam_width_rad(self) -> float:
        return float(self.edges[1] - self.edges[0])

    def angle_to_index(self, angle_rad: np.ndarray | float) -> np.ndarray:
        angle = np.asarray(angle_rad, dtype=np.float64)
        clipped = np.clip(angle, self.edges[0], self.edges[-1] - 1e-12)
        indices = np.searchsorted(self.edges, clipped, side="right") - 1
        return np.clip(indices, 0, self.num_beams - 1).astype(np.int64)

    def neighbouring_indices(self, center_index: int, k: int) -> list[int]:
        """Return k beams ordered by proximity to a center beam."""
        if not 0 <= center_index < self.num_beams:
            raise ValueError("center_index outside codebook")
        if not 1 <= k <= self.num_beams:
            raise ValueError("k must be in [1, num_beams]")

        candidates = list(range(self.num_beams))
        candidates.sort(key=lambda index: (abs(index - center_index), index))
        return candidates[:k]
