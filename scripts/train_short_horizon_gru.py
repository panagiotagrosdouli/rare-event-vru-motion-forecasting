from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.short_horizon_gru import ShortHorizonTailGRU
from src.tail_dataset import ArgoverseTailDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tail-aware GRU for short-term VRU forecasting."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("outputs/tail_dataset_train.csv"),
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("outputs/tail_dataset_val.csv"),
    )
    parser.add_argument("--history-steps", type=int, default=20)
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tail-loss-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/short_horizon_tail_gru.pth"),
    )
    return parser.parse_args()


def displacement_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    errors = torch.linalg.norm(prediction - target, dim=-1)
    return errors.mean(), errors[:, -1].mean()


def evaluate(
    model: ShortHorizonTailGRU,
    loader: DataLoader,
    device: torch.device,
    future_steps: int,
) -> tuple[float, float]:
    model.eval()
    ade_sum = 0.0
    fde_sum = 0.0
    samples = 0

    with torch.no_grad():
        for batch in loader:
            past = batch["past"].float().to(device)
            target = batch["future"][:, :future_steps].float().to(device)
            prediction, _ = model(past)
            ade, fde = displacement_metrics(prediction, target)
            batch_size = past.shape[0]
            ade_sum += ade.item() * batch_size
            fde_sum += fde.item() * batch_size
            samples += batch_size

    return ade_sum / samples, fde_sum / samples


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ArgoverseTailDataset(str(args.train_csv))
    val_dataset = ArgoverseTailDataset(str(args.val_csv))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ShortHorizonTailGRU(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    trajectory_loss = nn.SmoothL1Loss()
    tail_loss = nn.BCEWithLogitsLoss()

    best_ade = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        samples = 0

        for batch in train_loader:
            past = batch["past"].float().to(device)
            target = batch["future"][:, : args.future_steps].float().to(device)
            tail_target = batch["tail_event"].float().to(device)

            prediction, tail_logits = model(past)
            loss = trajectory_loss(prediction, target)
            loss = loss + args.tail_loss_weight * tail_loss(
                tail_logits,
                tail_target,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = past.shape[0]
            loss_sum += loss.item() * batch_size
            samples += batch_size

        val_ade, val_fde = evaluate(
            model,
            val_loader,
            device,
            args.future_steps,
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={loss_sum / samples:.6f} "
            f"val_ADE={val_ade:.4f}m "
            f"val_FDE={val_fde:.4f}m"
        )

        if val_ade < best_ade:
            best_ade = val_ade
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history_steps": args.history_steps,
                    "future_steps": args.future_steps,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "validation_ade_m": val_ade,
                    "validation_fde_m": val_fde,
                },
                args.output,
            )

    print(f"Best validation ADE: {best_ade:.4f} m")
    print(f"Saved checkpoint to: {args.output}")


if __name__ == "__main__":
    main()
