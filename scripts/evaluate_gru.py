from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.gru import GRUTrajectoryPredictor
from src.dataset import ArgoverseFocalDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

MODEL_PATH = Path("outputs/gru_baseline.pth")
TAIL_CSV_PATH = Path("outputs/tail_dataset_val.csv")


def displacement_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    prediction, target: (B, T, 2)

    ADE: μέση ευκλείδεια απόσταση σε όλα τα future timesteps.
    FDE: ευκλείδεια απόσταση στο τελευταίο future timestep.
    """
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


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not TAIL_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Tail labels not found: {TAIL_CSV_PATH}"
        )

    print("Using device:", DEVICE)

    tail_df = pd.read_csv(TAIL_CSV_PATH)

    required_columns = {
        "scenario_path",
        "tail_event",
        "object_type",
    }

    missing_columns = required_columns - set(tail_df.columns)

    if missing_columns:
        raise RuntimeError(
            f"Missing CSV columns: {sorted(missing_columns)}"
        )

    tail_lookup = {
        str(row["scenario_path"]): int(row["tail_event"])
        for _, row in tail_df.iterrows()
    }

    val_dataset = ArgoverseFocalDataset(
        root_dir="val",
        only_vrus=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Validation samples:", len(val_dataset))

    model = GRUTrajectoryPredictor().to(DEVICE)

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

    missing_tail_labels = 0

    with torch.no_grad():
        for batch in val_loader:
            past = batch["past"].float().to(DEVICE)
            future = batch["future"].float().to(DEVICE)

            prediction = model(past)

            batch_ade, batch_fde = displacement_errors(
                prediction,
                future,
            )

            batch_ade = batch_ade.cpu().tolist()
            batch_fde = batch_fde.cpu().tolist()

            object_types = batch["object_type"]
            scenario_paths = batch["scenario_path"]

            for ade, fde, object_type, scenario_path in zip(
                batch_ade,
                batch_fde,
                object_types,
                scenario_paths,
            ):
                scenario_path = str(scenario_path)
                object_type = str(object_type)

                overall_ade.append(ade)
                overall_fde.append(fde)

                class_ade[object_type].append(ade)
                class_fde[object_type].append(fde)

                tail_event = tail_lookup.get(scenario_path)

                if tail_event is None:
                    missing_tail_labels += 1
                    continue

                if tail_event == 1:
                    tail_ade.append(ade)
                    tail_fde.append(fde)

                    class_tail_ade[object_type].append(ade)
                    class_tail_fde[object_type].append(fde)
                else:
                    normal_ade.append(ade)
                    normal_fde.append(fde)

    print("\n" + "=" * 60)
    print("GRU BASELINE EVALUATION")
    print("=" * 60)

    print("\nOverall:")
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

    print("\nPer-class results:")

    for object_type in sorted(class_ade):
        print(f"\n{object_type}:")
        print(
            f"  ADE: {mean_or_nan(class_ade[object_type]):.4f} m"
        )
        print(
            f"  FDE: {mean_or_nan(class_fde[object_type]):.4f} m"
        )
        print(
            f"  Samples: {len(class_ade[object_type])}"
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

    print("\nMissing tail labels:", missing_tail_labels)


if __name__ == "__main__":
    main()
