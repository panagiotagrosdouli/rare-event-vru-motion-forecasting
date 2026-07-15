import torch

from models.tail_gru import TailAwareGRU

model = TailAwareGRU()

x = torch.randn(
    4,
    50,
    2,
)

future,tail = model(x)

print()

print("Future:",future.shape)

print("Tail:",tail.shape)
