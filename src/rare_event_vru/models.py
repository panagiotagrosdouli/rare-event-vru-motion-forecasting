from __future__ import annotations

import torch
from torch import nn


class MultiModalGRU(nn.Module):
    """K genuine trajectory hypotheses with normalized mode probabilities and scales."""

    def __init__(self, future_steps: int = 60, modes: int = 3, hidden_size: int = 64) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.modes = modes
        self.encoder = nn.GRU(2, hidden_size, batch_first=True)
        self.trajectory_head = nn.Linear(hidden_size, modes * future_steps * 2)
        self.score_head = nn.Linear(hidden_size, modes)
        self.log_scale_head = nn.Linear(hidden_size, modes * future_steps * 2)

    def forward(self, past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, hidden = self.encoder(past)
        latent = hidden[-1]
        trajectories = self.trajectory_head(latent).view(
            -1, self.modes, self.future_steps, 2
        )
        probabilities = torch.softmax(self.score_head(latent), dim=-1)
        scales = torch.nn.functional.softplus(
            self.log_scale_head(latent).view(-1, self.modes, self.future_steps, 2)
        ) + 1e-4
        return trajectories, probabilities, scales


def best_of_k_loss(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    per_mode = ((predictions - target[:, None]) ** 2).mean(dim=(-1, -2))
    return per_mode.min(dim=1).values.mean()
