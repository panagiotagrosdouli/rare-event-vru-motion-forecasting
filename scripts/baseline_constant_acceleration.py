import torch
from torch.utils.data import DataLoader

from src.dataset import ArgoverseFocalDataset


def constant_acceleration_prediction(past, future_steps=60):
    """
    past: [B, 50, 2]
    returns: [B, 60, 2]
    """

    last_position = past[:, -1, :]

    velocity_t = past[:, -1, :] - past[:, -2, :]
    velocity_prev = past[:, -2, :] - past[:, -3, :]

    acceleration = velocity_t - velocity_prev

    steps = torch.arange(
        1,
        future_steps + 1,
        device=past.device,
        dtype=past.dtype,
    ).view(1, future_steps, 1)

    prediction = (
        last_position.unsqueeze(1)
        + steps * velocity_t.unsqueeze(1)
        + 0.5 * (steps ** 2) * acceleration.unsqueeze(1)
    )

    return prediction


def compute_metrics(prediction, target):
    errors = torch.linalg.norm(prediction - target, dim=-1)
    return errors.mean().item(), errors[:, -1].mean().item()


dataset = ArgoverseFocalDataset("val")

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
)

total_ade = 0.0
total_fde = 0.0
total_samples = 0

for batch in loader:
    past = batch["past"].float()
    future = batch["future"].float()

    prediction = constant_acceleration_prediction(past)
    ade, fde = compute_metrics(prediction, future)

    batch_size = past.shape[0]

    total_ade += ade * batch_size
    total_fde += fde * batch_size
    total_samples += batch_size

print(f"Validation samples: {total_samples}")
print(f"Constant Acceleration ADE: {total_ade / total_samples:.4f} m")
print(f"Constant Acceleration FDE: {total_fde / total_samples:.4f} m")
