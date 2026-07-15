import torch
from torch.utils.data import DataLoader

from src.dataset import ArgoverseFocalDataset


def constant_velocity_prediction(past, future_steps=60):
    """
    past: [B, 50, 2]
    returns: [B, 60, 2]
    """

    last_position = past[:, -1, :]
    previous_position = past[:, -2, :]

    velocity = last_position - previous_position

    steps = torch.arange(
        1,
        future_steps + 1,
        device=past.device,
        dtype=past.dtype,
    ).view(1, future_steps, 1)

    prediction = last_position.unsqueeze(1) + steps * velocity.unsqueeze(1)

    return prediction


def compute_metrics(prediction, target):
    errors = torch.linalg.norm(prediction - target, dim=-1)

    ade = errors.mean()
    fde = errors[:, -1].mean()

    return ade.item(), fde.item()


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

    prediction = constant_velocity_prediction(past)

    ade, fde = compute_metrics(prediction, future)

    batch_size = past.shape[0]

    total_ade += ade * batch_size
    total_fde += fde * batch_size
    total_samples += batch_size

final_ade = total_ade / total_samples
final_fde = total_fde / total_samples

print(f"Validation samples: {total_samples}")
print(f"Constant Velocity ADE: {final_ade:.4f} m")
print(f"Constant Velocity FDE: {final_fde:.4f} m")
