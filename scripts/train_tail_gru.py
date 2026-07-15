from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.tail_gru import TailAwareGRU
from src.tail_dataset import ArgoverseTailDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
TAIL_LOSS_WEIGHT = 0.2

TRAIN_CSV = "outputs/tail_dataset_train.csv"
MODEL_PATH = Path("outputs/tail_aware_gru.pth")


def create_balanced_sampler(
    dataset: ArgoverseTailDataset,
) -> WeightedRandomSampler:
    """
    Εξισορροπεί normal και tail samples για το tail-classification task.
    """
    labels = (
        dataset.metadata["tail_event"]
        .astype(int)
        .tolist()
    )

    normal_count = labels.count(0)
    tail_count = labels.count(1)

    if normal_count == 0 or tail_count == 0:
        raise RuntimeError(
            "Both normal and tail samples are required."
        )

    class_weights = {
        0: 1.0 / normal_count,
        1: 1.0 / tail_count,
    }

    sample_weights = [
        class_weights[label]
        for label in labels
    ]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def main() -> None:
    print("Using device:", DEVICE)

    train_dataset = ArgoverseTailDataset(
        TRAIN_CSV
    )

    sampler = create_balanced_sampler(
        train_dataset
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    print("Training samples:", len(train_dataset))

    print(
        "Original tail distribution:\n",
        train_dataset.metadata[
            "tail_event"
        ].value_counts(),
    )

    model = TailAwareGRU().to(DEVICE)

    trajectory_criterion = nn.MSELoss()

    tail_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss_sum = 0.0
        trajectory_loss_sum = 0.0
        tail_loss_sum = 0.0
        correct_tail = 0
        total_samples = 0

        for batch in train_loader:
            past = (
                batch["past"]
                .float()
                .to(DEVICE)
            )

            future = (
                batch["future"]
                .float()
                .to(DEVICE)
            )

            tail_target = (
                batch["tail_event"]
                .float()
                .to(DEVICE)
            )

            predicted_future, tail_logits = model(
                past
            )

            trajectory_loss = trajectory_criterion(
                predicted_future,
                future,
            )

            tail_loss = tail_criterion(
                tail_logits,
                tail_target,
            )

            total_loss = (
                trajectory_loss
                + TAIL_LOSS_WEIGHT * tail_loss
            )

            optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            batch_size = past.shape[0]

            total_loss_sum += (
                total_loss.item() * batch_size
            )

            trajectory_loss_sum += (
                trajectory_loss.item() * batch_size
            )

            tail_loss_sum += (
                tail_loss.item() * batch_size
            )

            tail_prediction = (
                torch.sigmoid(tail_logits) >= 0.5
            ).long()

            correct_tail += (
                tail_prediction
                == tail_target.long()
            ).sum().item()

            total_samples += batch_size

        mean_total_loss = (
            total_loss_sum / total_samples
        )

        mean_trajectory_loss = (
            trajectory_loss_sum / total_samples
        )

        mean_tail_loss = (
            tail_loss_sum / total_samples
        )

        tail_accuracy = (
            correct_tail / total_samples
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Total Loss = {mean_total_loss:.4f} | "
            f"Trajectory Loss = "
            f"{mean_trajectory_loss:.4f} | "
            f"Tail Loss = {mean_tail_loss:.4f} | "
            f"Tail Acc = {tail_accuracy:.4f}"
        )

        if mean_total_loss < best_loss:
            best_loss = mean_total_loss

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            print(
                f"  Saved best model to "
                f"{MODEL_PATH}"
            )

    print("\nTraining finished.")
    print("Best training loss:", best_loss)
    print("Model:", MODEL_PATH)


if __name__ == "__main__":
    main()
