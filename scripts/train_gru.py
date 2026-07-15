import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import ArgoverseFocalDataset
from models.gru import GRUTrajectoryPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# -----------------------
# Dataset
# -----------------------

train_dataset = ArgoverseFocalDataset(
    root_dir="train",
    only_vrus=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

print("Training samples:", len(train_dataset))

# -----------------------
# Model
# -----------------------

model = GRUTrajectoryPredictor().to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

# -----------------------
# Training
# -----------------------

EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0.0

    for batch in train_loader:

        past = batch["past"].float().to(DEVICE)

        future = batch["future"].float().to(DEVICE)

        prediction = model(past)

        loss = criterion(
            prediction,
            future,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)

    print(
        f"Epoch {epoch+1:02d} | Loss = {epoch_loss:.4f}"
    )

print("\nTraining finished!")

torch.save(
    model.state_dict(),
    "outputs/gru_baseline.pth",
)

print("Model saved to outputs/gru_baseline.pth")
