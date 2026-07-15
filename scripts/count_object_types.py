from collections import Counter
from pathlib import Path

import pandas as pd


def count_focal_types(split_dir: str) -> Counter:
    counts = Counter()
    files = sorted(Path(split_dir).glob("*/scenario_*.parquet"))

    for i, scenario_path in enumerate(files, start=1):
        df = pd.read_parquet(
            scenario_path,
            columns=["track_id", "object_type", "focal_track_id"],
        )

        focal_id = df["focal_track_id"].iloc[0]
        focal_rows = df[df["track_id"] == focal_id]

        if focal_rows.empty:
            counts["missing_focal"] += 1
            continue

        object_type = focal_rows["object_type"].iloc[0]
        counts[object_type] += 1

        if i % 10000 == 0:
            print(f"{split_dir}: processed {i} scenarios")

    return counts


for split in ["train", "val"]:
    counts = count_focal_types(split)

    print(f"\n{split.upper()} FOCAL OBJECT TYPES")
    total = sum(counts.values())

    for object_type, count in counts.most_common():
        percentage = 100.0 * count / total
        print(f"{object_type:25s} {count:8d}  {percentage:6.2f}%")

    vru_types = {
        "pedestrian",
        "cyclist",
        "motorcyclist",
        "riderless_bicycle",
    }

    vru_count = sum(
        count for object_type, count in counts.items()
        if object_type in vru_types
    )

    print(f"\nTotal focal agents: {total}")
    print(f"VRU focal agents:   {vru_count}")
    print(f"VRU percentage:     {100.0 * vru_count / total:.2f}%")
