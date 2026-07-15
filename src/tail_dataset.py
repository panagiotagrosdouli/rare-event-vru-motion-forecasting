from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from src.preprocess import preprocess_focal_agent


OBJECT_TYPE_TO_ID = {
    "vehicle": 0,
    "bus": 1,
    "pedestrian": 2,
    "cyclist": 3,
    "motorcyclist": 4,
}


class ArgoverseTailDataset(Dataset):
    """
    Dataset για VRU trajectory prediction με tail labels.

    Διαβάζει ένα CSV όπως:
        outputs/tail_dataset_train.csv
        outputs/tail_dataset_val.csv

    και επιστρέφει:
        past
        future
        object_type
        object_type_id
        tail_event
        tail_score
        origin
        angle
        scenario_path
    """

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_path}"
            )

        self.metadata = pd.read_csv(self.csv_path)

        required_columns = {
            "scenario_path",
            "object_type",
            "tail_event",
            "tail_score",
        }

        missing_columns = required_columns - set(
            self.metadata.columns
        )

        if missing_columns:
            raise RuntimeError(
                f"Missing columns in {self.csv_path}: "
                f"{sorted(missing_columns)}"
            )

        self.metadata = self.metadata.reset_index(
            drop=True
        )

        if len(self.metadata) == 0:
            raise RuntimeError(
                f"No samples found in {self.csv_path}"
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict:
        row = self.metadata.iloc[index]

        scenario_path = Path(
            str(row["scenario_path"])
        )

        if not scenario_path.exists():
            raise FileNotFoundError(
                f"Scenario file not found: {scenario_path}"
            )

        df = pd.read_parquet(scenario_path)

        (
            past,
            future,
            object_type,
            origin,
            angle,
        ) = preprocess_focal_agent(df)

        if object_type not in OBJECT_TYPE_TO_ID:
            raise ValueError(
                f"Unsupported object type: {object_type}"
            )

        csv_object_type = str(row["object_type"])

        if object_type != csv_object_type:
            raise RuntimeError(
                "Object type mismatch. "
                f"CSV={csv_object_type}, "
                f"Parquet={object_type}, "
                f"Scenario={scenario_path}"
            )

        return {
            "past": past,
            "future": future,
            "object_type": object_type,
            "object_type_id": OBJECT_TYPE_TO_ID[
                object_type
            ],
            "tail_event": int(row["tail_event"]),
            "tail_score": float(row["tail_score"]),
            "origin": origin,
            "angle": angle,
            "scenario_path": str(scenario_path),
        }
