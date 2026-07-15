import torch
import torch.nn as nn


class TypeAwareGRU(nn.Module):
    def __init__(
        self,
        hidden_size=128,
        embedding_dim=16,
        future_steps=60,
    ):
        super().__init__()

        self.future_steps = future_steps

        # 5 object types:
        # vehicle, bus, pedestrian, cyclist, motorcyclist
        self.type_embedding = nn.Embedding(
            num_embeddings=5,
            embedding_dim=embedding_dim,
        )

        self.gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, future_steps * 2),
        )

    def forward(self, past, object_type):

        _, hidden = self.gru(past)

        hidden = hidden.squeeze(0)

        emb = self.type_embedding(object_type)

        x = torch.cat([hidden, emb], dim=1)

        future = self.decoder(x)

        future = future.view(-1, self.future_steps, 2)

        return future
