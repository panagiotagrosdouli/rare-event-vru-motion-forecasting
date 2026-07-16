from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.gru import GRUTrajectoryPredictor
from src.tail_dataset import ArgoverseTailDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the deterministic GRU with a bounded, continuous "
            "tail-score-weighted trajectory loss."
        )
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("outputs/tail_dataset_train.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tail_weighted_gru.pth"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Strength of continuous tail-score weighting.",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=3.0,
        help="Upper bound on each sample weight.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tail_score: torch.Tensor,
    *,
    alpha: float,
    max_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted loss, unweighted MSE, and normalized sample weights.

    The weighting is continuous and bounded:

        raw_weight = clamp(1 + alpha * tail_score, 1, max_weight)

    We divide by the batch mean weight so that changing alpha does not
    trivially change the global loss scale or effective learning rate.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if max_weight < 1:
        raise ValueError("max_weight must be at least 1")

    per_sample_mse = (prediction - target).pow(2).mean(dim=(1, 2))
    raw_weights = torch.clamp(
        1.0 + alpha * tail_score,
        min=1.0,
        max=max_weight,
    )
    normalized_weights = raw_weights / raw_weights.mean().clamp_min(1e-8)
    weighted_loss = (normalized_weights * per_sample_mse).mean()
    unweighted_loss = per_sample_mse.mean()
    return weighted_loss, unweighted_loss, normalized_weights


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = ArgoverseTailDataset(str(args.train_csv))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = GRUTrajectoryPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_weighted_loss = float("inf")
    history: list[dict[str, float | int]] = []

    print("Training samples:", len(dataset))
    print(f"Alpha: {args.alpha:.4f}")
    print(f"Maximum sample weight: {args.max_weight:.4f}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        weighted_sum = 0.0
        unweighted_sum = 0.0
        weight_sum = 0.0
        weight_max = 0.0
        sample_count = 0

        for batch in loader:
            past = batch["past"].float().to(device)
            future = batch["future"].float().to(device)
            tail_score = batch["tail_score"].float().to(device)

            prediction = model(past)
            weighted_loss, unweighted_loss, sample_weights = (
                weighted_trajectory_loss(
                    prediction,
                    future,
                    tail_score,
                    alpha=args.alpha,
                    max_weight=args.max_weight,
                )
            )

            optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = past.shape[0]
            sample_count += batch_size
            weighted_sum += weighted_loss.item() * batch_size
            unweighted_sum += unweighted_loss.item() * batch_size
            weight_sum += sample_weights.sum().item()
            weight_max = max(weight_max, sample_weights.max().item())

        mean_weighted = weighted_sum / sample_count
        mean_unweighted = unweighted_sum / sample_count
        mean_normalized_weight = weight_sum / sample_count
        history.append(
            {
                "epoch": epoch,
                "weighted_loss": mean_weighted,
                "unweighted_loss": mean_unweighted,
                "mean_normalized_weight": mean_normalized_weight,
                "max_normalized_weight": weight_max,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Weighted={mean_weighted:.4f} | "
            f"Unweighted={mean_unweighted:.4f} | "
            f"MeanWeight={mean_normalized_weight:.4f} | "
            f"MaxWeight={weight_max:.4f}"
        )

        if mean_weighted < best_weighted_loss:
            best_weighted_loss = mean_weighted
            torch.save(model.state_dict(), args.output)
            print(f"  Saved best model to {args.output}")

    metadata = {
        "model": "GRUTrajectoryPredictor",
        "objective": "bounded_continuous_tail_weighted_mse",
        "train_csv": str(args.train_csv),
        "checkpoint": str(args.output),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "alpha": args.alpha,
        "max_weight": args.max_weight,
        "seed": args.seed,
        "device": device,
        "best_weighted_training_loss": best_weighted_loss,
        "history": history,
        "note": (
            "Weights are normalized within each batch and capped to avoid "
            "destructive rare-sample dominance."
        ),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nTraining finished.")
    print("Best weighted training loss:", best_weighted_loss)
    print("Model:", args.output)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
