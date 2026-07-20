from __future__ import annotations

import torch
import torch.nn as nn


class ShortHorizonTailGRU(nn.Module):
    """Tail-aware GRU for fast short-term trajectory forecasting.

    The model consumes only the most recent ``history_steps`` positions and
    predicts a small future window as incremental displacements. Cumulative
    displacements are anchored at the last observed position, which keeps the
    output continuous with the observed trajectory.

    Defaults correspond to Argoverse 2 sampling at 10 Hz:
        history_steps=20 -> 2.0 seconds of history
        future_steps=10  -> 1.0 second of prediction
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        history_steps: int = 20,
        future_steps: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if history_steps < 2:
            raise ValueError("history_steps must be at least 2")
        if future_steps < 1:
            raise ValueError("future_steps must be positive")

        self.history_steps = history_steps
        self.future_steps = future_steps

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )

        self.displacement_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, future_steps * 2),
        )

        self.tail_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict future positions and a tail-event logit.

        Args:
            past: Tensor shaped ``(batch, time, 2)``. When more than
                ``history_steps`` observations are supplied, only the latest
                observations are used.

        Returns:
            A tuple containing predicted positions shaped
            ``(batch, future_steps, 2)`` and tail logits shaped ``(batch,)``.
        """
        if past.ndim != 3 or past.shape[-1] != 2:
            raise ValueError("past must have shape (batch, time, 2)")
        if past.shape[1] < self.history_steps:
            raise ValueError(
                f"past contains {past.shape[1]} steps; "
                f"at least {self.history_steps} are required"
            )

        recent = past[:, -self.history_steps :, :]
        last_position = recent[:, -1, :]

        _, hidden = self.encoder(recent)
        encoded = hidden[-1]

        increments = self.displacement_head(encoded).view(
            -1,
            self.future_steps,
            2,
        )
        future = last_position.unsqueeze(1) + torch.cumsum(increments, dim=1)
        tail_logits = self.tail_head(encoded).squeeze(-1)

        return future, tail_logits
