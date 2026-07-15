import torch

from models.gru import GRUTrajectoryPredictor


model = GRUTrajectoryPredictor()

x = torch.randn(
    8,
    50,
    2,
)

y = model(x)

print()

print("Input :", x.shape)
print("Output:", y.shape)

print()

print(model)
