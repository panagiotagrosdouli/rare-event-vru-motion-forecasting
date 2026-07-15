from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.tail_gru import TailAwareGRU
from src.tail_dataset import ArgoverseTailDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3

# Μικρό βάρος ώστε το auxiliary task να μην καταστρέφει
# την κύρια πρόβλεψη τροχιάς.
TAIL_LOSS_WEIGHT = 0.05

TRAIN_CSV = "outputs/tail_dataset_train.csv"
MODEL_PATH = Path("outputs/tail_aware_gru_v2.pth")


def main() -> None:
    print("Using device:", DEVICE)

    dataset = ArgoverseTailDataset(TRAIN_CSV)

    # Κανονικό shuffle — όχι τεχνητή εξισορρόπηση των batches.
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    labels = dataset.metadata["tail_event"].astype(int)

    normal_count = int((labels == 0).sum())
    tail_count = int((labels == 1).sum())

    # Η BCE δίνει μεγαλύτερη σημασία στη μειοψηφική θετική κλάση,
    # χωρίς να αλλάζει την κατανομή των trajectory samples.
    positive_weight = normal_count / tail_count

    print("Training samples:", len(dataset))
    print("Normal samples:", normal_count)
    print("Tail samples:", tail_count)
    print(f"BCE positive weight: {positive_weight:.4f}")

    model = TailAwareGRU().to(DEVICE)

    trajectory_criterion = nn.MSELoss()

    tail_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_weight],
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_trajectory_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss_sum = 0.0
        trajectory_loss_sum = 0.0
        tail_loss_sum = 0.0
        total_samples = 0

        true_positive = 0
        false_positive = 0
        false_negative = 0

        for batch in loader:
            past = batch["past"].float().to(DEVICE)
            future = batch["future"].float().to(DEVICE)
            tail_target = batch["tail_event"].float().to(DEVICE)

            predicted_future, tail_logits = model(past)

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
            total_samples += batch_size

            total_loss_sum += total_loss.item() * batch_size
            trajectory_loss_sum += (
                trajectory_loss.item() * batch_size
            )
            tail_loss_sum += tail_loss.item() * batch_size

            tail_prediction = (
                torch.sigmoid(tail_logits) >= 0.5
            ).long()

            target_int = tail_target.long()

            true_positive += (
                (tail_prediction == 1) & (target_int == 1)
            ).sum().item()

            false_positive += (
                (tail_prediction == 1) & (target_int == 0)
            ).sum().item()

            false_negative += (
                (tail_prediction == 0) & (target_int == 1)
            ).sum().item()

        mean_total_loss = total_loss_sum / total_samples
        mean_trajectory_loss = (
            trajectory_loss_sum / total_samples
        )
        mean_tail_loss = tail_loss_sum / total_samples

        precision = true_positive / max(
            true_positive + false_positive,
            1,
        )

        recall = true_positive / max(
            true_positive + false_negative,
            1,
        )

        f1 = (
            2 * precision * recall
            / max(precision + recall, 1e-8)
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Total={mean_total_loss:.4f} | "
            f"Trajectory={mean_trajectory_loss:.4f} | "
            f"Tail={mean_tail_loss:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

        # Επιλέγουμε μοντέλο με βάση την κύρια εργασία,
        # όχι το συνολικό multi-task loss.
        if mean_trajectory_loss < best_trajectory_loss:
            best_trajectory_loss = mean_trajectory_loss

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            print(f"  Saved best model to {MODEL_PATH}")

    print("\nTraining finished.")
    print(
        "Best trajectory training loss:",
        best_trajectory_loss,
    )
    print("Model:", MODEL_PATH)


if __name__ == "__main__":
    main()
