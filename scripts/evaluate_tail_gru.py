from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.tail_gru import TailAwareGRU
from src.tail_dataset import ArgoverseTailDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

MODEL_PATH = Path("outputs/tail_aware_gru.pth")
VAL_CSV = "outputs/tail_dataset_val.csv"


def displacement_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.linalg.vector_norm(
        prediction - target,
        dim=-1,
    )

    ade = distances.mean(dim=1)
    fde = distances[:, -1]

    return ade, fde


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")

    return sum(values) / len(values)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("Using device:", DEVICE)

    dataset = ArgoverseTailDataset(VAL_CSV)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Validation samples:", len(dataset))

    model = TailAwareGRU().to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(state_dict)
    model.eval()

    overall_ade: list[float] = []
    overall_fde: list[float] = []

    normal_ade: list[float] = []
    normal_fde: list[float] = []

    tail_ade: list[float] = []
    tail_fde: list[float] = []

    class_ade: dict[str, list[float]] = defaultdict(list)
    class_fde: dict[str, list[float]] = defaultdict(list)

    class_tail_ade: dict[str, list[float]] = defaultdict(list)
    class_tail_fde: dict[str, list[float]] = defaultdict(list)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for batch in loader:
            past = batch["past"].float().to(DEVICE)
            future = batch["future"].float().to(DEVICE)

            tail_target = (
                batch["tail_event"]
                .long()
                .to(DEVICE)
            )

            predicted_future, tail_logits = model(past)

            batch_ade, batch_fde = displacement_errors(
                predicted_future,
                future,
            )

            tail_probability = torch.sigmoid(tail_logits)

            tail_prediction = (
                tail_probability >= 0.5
            ).long()

            true_positive += (
                (tail_prediction == 1)
                & (tail_target == 1)
            ).sum().item()

            true_negative += (
                (tail_prediction == 0)
                & (tail_target == 0)
            ).sum().item()

            false_positive += (
                (tail_prediction == 1)
                & (tail_target == 0)
            ).sum().item()

            false_negative += (
                (tail_prediction == 0)
                & (tail_target == 1)
            ).sum().item()

            batch_ade = batch_ade.cpu().tolist()
            batch_fde = batch_fde.cpu().tolist()
            tail_target = tail_target.cpu().tolist()

            object_types = batch["object_type"]

            for ade, fde, is_tail, object_type in zip(
                batch_ade,
                batch_fde,
                tail_target,
                object_types,
            ):
                object_type = str(object_type)

                overall_ade.append(ade)
                overall_fde.append(fde)

                class_ade[object_type].append(ade)
                class_fde[object_type].append(fde)

                if is_tail == 1:
                    tail_ade.append(ade)
                    tail_fde.append(fde)

                    class_tail_ade[object_type].append(ade)
                    class_tail_fde[object_type].append(fde)
                else:
                    normal_ade.append(ade)
                    normal_fde.append(fde)

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    accuracy = safe_divide(
        true_positive + true_negative,
        true_positive
        + true_negative
        + false_positive
        + false_negative,
    )

    print("\n" + "=" * 60)
    print("TAIL-AWARE GRU EVALUATION")
    print("=" * 60)

    print("\nOverall trajectory metrics:")
    print(f"ADE: {mean_or_nan(overall_ade):.4f} m")
    print(f"FDE: {mean_or_nan(overall_fde):.4f} m")
    print(f"Samples: {len(overall_ade)}")

    print("\nNormal trajectories:")
    print(f"ADE: {mean_or_nan(normal_ade):.4f} m")
    print(f"FDE: {mean_or_nan(normal_fde):.4f} m")
    print(f"Samples: {len(normal_ade)}")

    print("\nTail trajectories:")
    print(f"ADE: {mean_or_nan(tail_ade):.4f} m")
    print(f"FDE: {mean_or_nan(tail_fde):.4f} m")
    print(f"Samples: {len(tail_ade)}")

    print("\nPer-class trajectory metrics:")

    for object_type in sorted(class_ade):
        print(f"\n{object_type}:")
        print(
            f"  ADE: "
            f"{mean_or_nan(class_ade[object_type]):.4f} m"
        )
        print(
            f"  FDE: "
            f"{mean_or_nan(class_fde[object_type]):.4f} m"
        )
        print(
            f"  Samples: "
            f"{len(class_ade[object_type])}"
        )

        print(
            f"  Tail ADE: "
            f"{mean_or_nan(class_tail_ade[object_type]):.4f} m"
        )
        print(
            f"  Tail FDE: "
            f"{mean_or_nan(class_tail_fde[object_type]):.4f} m"
        )
        print(
            f"  Tail samples: "
            f"{len(class_tail_ade[object_type])}"
        )

    print("\nTail classification metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion matrix:")
    print(f"TP: {true_positive}")
    print(f"TN: {true_negative}")
    print(f"FP: {false_positive}")
    print(f"FN: {false_negative}")


if __name__ == "__main__":
    main()
