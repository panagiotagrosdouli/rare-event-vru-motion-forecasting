from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TAIL_PERCENTILE = 95

FEATURES = [
    "max_speed",
    "max_acceleration",
    "max_heading_change",
]


def normalize(series):
    minimum = series.min()
    maximum = series.max()

    if maximum - minimum < 1e-8:
        return series * 0.0

    return (series - minimum) / (maximum - minimum)


def create_labels(split: str):

    input_file = Path(
        f"outputs/tail_statistics_{split}.csv"
    )

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    df = pd.read_csv(input_file)

    df["tail_score"] = 0.0
    df["tail_event"] = 0

    print("=" * 60)
    print(f"CLASS-WISE TAIL LABELING ({split.upper()})")
    print("=" * 60)

    for object_type in sorted(df["object_type"].unique()):

        print(f"\nProcessing {object_type}")

        mask = df["object_type"] == object_type

        subset = df.loc[mask].copy()

        speed = normalize(subset["max_speed"])
        acc = normalize(subset["max_acceleration"])
        heading = normalize(subset["max_heading_change"])

        subset["tail_score"] = (
            0.4 * speed
            + 0.4 * acc
            + 0.2 * heading
        )

        threshold = subset["tail_score"].quantile(
            TAIL_PERCENTILE / 100
        )

        subset["tail_event"] = (
            subset["tail_score"] >= threshold
        ).astype(int)

        df.loc[
            mask,
            "tail_score",
        ] = subset["tail_score"]

        df.loc[
            mask,
            "tail_event",
        ] = subset["tail_event"]

        print(
            f"Threshold: {threshold:.4f}"
        )

        print(
            subset["tail_event"].value_counts()
        )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        pd.crosstab(
            df["object_type"],
            df["tail_event"],
            normalize="index",
        ) * 100
    )

    output = Path(
        f"outputs/tail_dataset_{split}.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print("\nSaved:")
    print(output)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=[
            "train",
            "val",
        ],
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    create_labels(args.split)
