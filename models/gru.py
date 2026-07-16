import torch.nn as nn


class GRUTrajectoryPredictor(nn.Module):
    """
    Baseline GRU for VRU trajectory prediction.

    Input:
        past trajectory (B, 50, 2)

    Output:
        future trajectory (B, 60, 2)
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

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(
                128,
                future_steps * 2,
            ),
        )

    def forward(self, past):
        _, hidden = self.gru(past)
        hidden = hidden[-1]

        future = self.decoder(hidden)
        future = future.view(
            -1,
            self.future_steps,
            2,
        )
        return future
