from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.gru import GRUTrajectoryPredictor
from src.tail_dataset import ArgoverseTailDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the tail-weighted deterministic GRU."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/tail_weighted_gru.pth"),
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("outputs/tail_dataset_val.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tail_weighted_gru_metrics.json"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def displacement_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.linalg.vector_norm(prediction - target, dim=-1)
    return distances.mean(dim=1), distances[:, -1]


def summarize(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {args.val_csv}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = ArgoverseTailDataset(str(args.val_csv))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print("Validation samples:", len(dataset))

    model = GRUTrajectoryPredictor().to(device)
    state_dict = torch.load(args.model, map_location=device)
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
    class_normal_ade: dict[str, list[float]] = defaultdict(list)
    class_normal_fde: dict[str, list[float]] = defaultdict(list)
    class_tail_ade: dict[str, list[float]] = defaultdict(list)
    class_tail_fde: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            past = batch["past"].float().to(device)
            future = batch["future"].float().to(device)
            prediction = model(past)
            batch_ade, batch_fde = displacement_errors(prediction, future)

            for ade, fde, is_tail, object_type in zip(
                batch_ade.cpu().tolist(),
                batch_fde.cpu().tolist(),
                batch["tail_event"].tolist(),
                batch["object_type"],
            ):
                object_type = str(object_type)
                overall_ade.append(ade)
                overall_fde.append(fde)
                class_ade[object_type].append(ade)
                class_fde[object_type].append(fde)

                if int(is_tail) == 1:
                    tail_ade.append(ade)
                    tail_fde.append(fde)
                    class_tail_ade[object_type].append(ade)
                    class_tail_fde[object_type].append(fde)
                else:
                    normal_ade.append(ade)
                    normal_fde.append(fde)
                    class_normal_ade[object_type].append(ade)
                    class_normal_fde[object_type].append(fde)

    metrics: dict[str, object] = {
        "model": str(args.model),
        "validation_csv": str(args.val_csv),
        "device": device,
        "overall": {
            "ade": summarize(overall_ade),
            "fde": summarize(overall_fde),
            "samples": len(overall_ade),
        },
        "normal": {
            "ade": summarize(normal_ade),
            "fde": summarize(normal_fde),
            "samples": len(normal_ade),
        },
        "tail": {
            "ade": summarize(tail_ade),
            "fde": summarize(tail_fde),
            "samples": len(tail_ade),
        },
        "per_class": {},
    }

    per_class = metrics["per_class"]
    assert isinstance(per_class, dict)
    for object_type in sorted(class_ade):
        per_class[object_type] = {
            "overall": {
                "ade": summarize(class_ade[object_type]),
                "fde": summarize(class_fde[object_type]),
                "samples": len(class_ade[object_type]),
            },
            "normal": {
                "ade": summarize(class_normal_ade[object_type]),
                "fde": summarize(class_normal_fde[object_type]),
                "samples": len(class_normal_ade[object_type]),
            },
            "tail": {
                "ade": summarize(class_tail_ade[object_type]),
                "fde": summarize(class_tail_fde[object_type]),
                "samples": len(class_tail_ade[object_type]),
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("TAIL-WEIGHTED GRU EVALUATION")
    print("=" * 60)
    for subset_name in ("overall", "normal", "tail"):
        subset = metrics[subset_name]
        assert isinstance(subset, dict)
        print(f"\n{subset_name.title()} trajectories:")
        print(f"ADE: {subset['ade']:.4f} m")
        print(f"FDE: {subset['fde']:.4f} m")
        print(f"Samples: {subset['samples']}")

    print("\nPer-class trajectory metrics:")
    for object_type, values in per_class.items():
        assert isinstance(values, dict)
        overall = values["overall"]
        tail = values["tail"]
        print(f"\n{object_type}:")
        print(f"  ADE: {overall['ade']:.4f} m")
        print(f"  FDE: {overall['fde']:.4f} m")
        print(f"  Samples: {overall['samples']}")
        print(f"  Tail ADE: {tail['ade']:.4f} m")
        print(f"  Tail FDE: {tail['fde']:.4f} m")
        print(f"  Tail samples: {tail['samples']}")

    print(f"\nSaved metrics to: {args.output}")


if __name__ == "__main__":
    main()
