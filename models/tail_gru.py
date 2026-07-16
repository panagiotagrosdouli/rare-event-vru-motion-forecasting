import torch.nn as nn


class TailAwareGRU(nn.Module):
    """
    Multi-task GRU.

    Task 1:
        Future trajectory prediction.

    Task 2:
        Tail-event classification.
    """

    def __init__(
        self,
        input_size=2,
        hidden_size=128,
        num_layers=2,
        future_steps=60,
    ):
        super().__init__()

        self.future_steps = future_steps

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, future_steps * 2),
        )

        self.tail_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, past):
        _, hidden = self.encoder(past)
        hidden = hidden[-1]

        future = self.trajectory_head(hidden)
        future = future.view(
            -1,
            self.future_steps,
            2,
        )

        tail_logits = self.tail_head(hidden)
        return future, tail_logits.squeeze(1)
