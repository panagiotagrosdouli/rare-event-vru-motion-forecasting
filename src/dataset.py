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

VRU_TYPES = {
    "pedestrian",
    "cyclist",
    "motorcyclist",
}


class ArgoverseFocalDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        only_vrus: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.only_vrus = only_vrus

        all_scenario_files = sorted(
            self.root_dir.glob("*/scenario_*.parquet")
        )

        if not all_scenario_files:
            raise RuntimeError(
                f"No scenario parquet files found in {self.root_dir}"
            )

        if self.only_vrus:
            self.scenario_files = self._filter_vru_scenarios(
                all_scenario_files
            )
        else:
            self.scenario_files = all_scenario_files

        if not self.scenario_files:
            raise RuntimeError(
                f"No valid scenarios found in {self.root_dir}"
            )

    def _filter_vru_scenarios(self, scenario_files):
        vru_files = []

        for index, scenario_path in enumerate(
            scenario_files,
            start=1,
        ):
            df = pd.read_parquet(
                scenario_path,
                columns=[
                    "track_id",
                    "object_type",
                    "focal_track_id",
                ],
            )

            focal_id = df["focal_track_id"].iloc[0]

            focal_rows = df[
                df["track_id"] == focal_id
            ]

            if focal_rows.empty:
                continue

            object_type = focal_rows[
                "object_type"
            ].iloc[0]

            if object_type in VRU_TYPES:
                vru_files.append(scenario_path)

            if index % 10000 == 0:
                print(
                    f"Processed {index} scenarios - "
                    f"VRUs found: {len(vru_files)}"
                )

        return vru_files

    def __len__(self):
        return len(self.scenario_files)

    def __getitem__(self, index):
        scenario_path = self.scenario_files[index]

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
                f"Unsupported focal object type: "
                f"{object_type}"
            )

        if (
            self.only_vrus
            and object_type not in VRU_TYPES
        ):
            raise RuntimeError(
                f"Non-VRU sample found in VRU dataset: "
                f"{object_type}"
            )

        object_type_id = OBJECT_TYPE_TO_ID[
            object_type
        ]

        return {
            "past": past,
            "future": future,
            "object_type": object_type,
            "object_type_id": object_type_id,
            "origin": origin,
            "angle": angle,
            "scenario_path": str(scenario_path),
        }
