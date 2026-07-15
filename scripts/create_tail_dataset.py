from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import preprocess_focal_agent


VRU_TYPES = {
    "pedestrian",
    "cyclist",
    "motorcyclist",
}


def compute_features(
    past: np.ndarray,
    future: np.ndarray,
    dt: float = 0.1,
) -> dict[str, float]:
    """
    Υπολογίζει βασικά κινηματικά χαρακτηριστικά
    από ολόκληρη την τροχιά του focal VRU.
    """
    trajectory = np.concatenate([past, future], axis=0)

    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError(
            f"Invalid trajectory shape: {trajectory.shape}"
        )

    displacement = np.diff(trajectory, axis=0)

    if len(displacement) < 2:
        raise ValueError("Trajectory is too short.")

    velocity = displacement / dt
    speed = np.linalg.norm(velocity, axis=1)

    acceleration = np.diff(velocity, axis=0) / dt
    acceleration_norm = np.linalg.norm(
        acceleration,
        axis=1,
    )

    heading = np.arctan2(
        displacement[:, 1],
        displacement[:, 0],
    )

    heading_change = np.diff(heading)

    # Περιορισμός της γωνιακής διαφοράς στο [-π, π]
    heading_change = np.arctan2(
        np.sin(heading_change),
        np.cos(heading_change),
    )

    absolute_heading_change = np.abs(
        heading_change
    )

    return {
        "mean_speed": float(np.mean(speed)),
        "max_speed": float(np.max(speed)),
        "mean_acceleration": float(
            np.mean(acceleration_norm)
        ),
        "max_acceleration": float(
            np.max(acceleration_norm)
        ),
        "mean_heading_change": float(
            np.mean(absolute_heading_change)
        ),
        "max_heading_change": float(
            np.max(absolute_heading_change)
        ),
        "total_displacement": float(
            np.linalg.norm(
                trajectory[-1] - trajectory[0]
            )
        ),
    }


def build_statistics(
    split: str,
    output_dir: str = "outputs",
) -> Path:
    """
    Δημιουργεί τα motion statistics για
    train ή validation split.
    """
    split_dir = Path(split)

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory does not exist: {split_dir}"
        )

    scenario_files = sorted(
        split_dir.glob("*/scenario_*.parquet")
    )

    if not scenario_files:
        raise RuntimeError(
            f"No scenario parquet files found in {split_dir}"
        )

    rows: list[dict] = []
    skipped = 0

    print("=" * 60)
    print(f"CREATING VRU TAIL STATISTICS: {split.upper()}")
    print("=" * 60)
    print(f"Scenario files found: {len(scenario_files)}")

    for index, scenario_path in enumerate(
        scenario_files,
        start=1,
    ):
        try:
            df = pd.read_parquet(scenario_path)

            (
                past,
                future,
                object_type,
                origin,
                angle,
            ) = preprocess_focal_agent(df)

            if object_type not in VRU_TYPES:
                continue

            features = compute_features(
                past=past,
                future=future,
            )

            scenario_id = (
                str(df["scenario_id"].iloc[0])
                if "scenario_id" in df.columns
                else scenario_path.parent.name
            )

            rows.append(
                {
                    "scenario_path": str(
                        scenario_path
                    ),
                    "scenario_id": scenario_id,
                    "object_type": object_type,
                    "origin_x": float(origin[0]),
                    "origin_y": float(origin[1]),
                    "rotation_angle": float(angle),
                    **features,
                }
            )

        except Exception as exc:
            skipped += 1
            print(
                f"Skipped {scenario_path}: {exc}"
            )

        if index % 10000 == 0:
            print(
                f"Processed {index} scenarios - "
                f"VRUs kept: {len(rows)} - "
                f"Skipped: {skipped}"
            )

    if not rows:
        raise RuntimeError(
            f"No VRU trajectories were found in {split_dir}"
        )

    stats_df = pd.DataFrame(rows)

    output_path = (
        Path(output_dir)
        / f"tail_statistics_{split}.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("FINISHED")
    print("=" * 60)

    print(f"Saved to: {output_path}")
    print(f"Total VRU trajectories: {len(stats_df)}")
    print(f"Skipped scenarios: {skipped}")

    print("\nClass distribution:")
    print(
        stats_df["object_type"].value_counts()
    )

    print("\nFeature summary:")
    print(
        stats_df[
            [
                "mean_speed",
                "max_speed",
                "mean_acceleration",
                "max_acceleration",
                "mean_heading_change",
                "max_heading_change",
                "total_displacement",
            ]
        ].describe()
    )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create VRU trajectory statistics "
            "for Argoverse 2."
        )
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val"],
        help="Dataset split to process.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for generated CSV files.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_statistics(
        split=args.split,
        output_dir=args.output_dir,
    )
