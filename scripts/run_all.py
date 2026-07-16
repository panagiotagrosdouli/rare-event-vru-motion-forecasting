from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rare_event_vru.baselines import constant_velocity
from rare_event_vru.conformal import empirical_endpoint_coverage, fit_endpoint_radius
from rare_event_vru.features import extract_motion_features
from rare_event_vru.fixture import make_fixture
from rare_event_vru.metrics import ade_fde
from rare_event_vru.rarity import apply_tail_statistics, fit_tail_statistics
from rare_event_vru.transforms import to_local


def run_fixture() -> dict[str, float | int | str]:
    trajectories, metadata = make_fixture(seed=0)
    feature_rows: list[dict[str, object]] = []
    local_trajectories: list[np.ndarray] = []
    for trajectory, row in zip(trajectories, metadata.to_dict("records")):
        local, _ = to_local(trajectory, observed_steps=50)
        local_trajectories.append(local)
        feature_rows.append({**row, **extract_motion_features(local)})
    frame = pd.DataFrame(feature_rows)
    train = frame.groupby("object_type", group_keys=False).head(18).reset_index(drop=True)
    validation = frame.groupby("object_type", group_keys=False).tail(6).reset_index(drop=True)
    artifact = fit_tail_statistics(train, quantile=0.9, seed=0)

    artifacts = ROOT / "artifacts"
    results = ROOT / "results"
    figures = ROOT / "assets" / "figures"
    artifacts.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    artifact.save(artifacts / "tail_statistics.json")
    labelled = apply_tail_statistics(validation, artifact)
    labelled.to_csv(results / "fixture_labels.csv", index=False)

    indices = [metadata.index[metadata.scenario_id == scenario_id][0] for scenario_id in validation.scenario_id]
    selected = np.stack([local_trajectories[index] for index in indices])
    predictions = np.stack([constant_velocity(item[:50], 60) for item in selected])
    target = selected[:, 50:]
    ade, fde = ade_fde(predictions, target)
    radius = fit_endpoint_radius(predictions[:9], target[:9], coverage=0.9)
    coverage = empirical_endpoint_coverage(predictions[9:], target[9:], radius)

    metrics: dict[str, float | int | str] = {
        "fixture_samples": int(len(labelled)),
        "fixture_tail_rate": float(labelled.tail_event.mean()),
        "constant_velocity_ade": float(ade.mean()),
        "constant_velocity_fde": float(fde.mean()),
        "conformal_endpoint_radius": radius,
        "conformal_test_coverage": coverage,
        "evidence_label": "Synthetic Validation",
    }
    (results / "fixture_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7, 5))
    for object_type, subset in labelled.groupby("object_type"):
        axis.scatter(subset.max_speed, subset.max_heading_change, label=object_type, alpha=0.8)
    axis.set_xlabel("Maximum speed")
    axis.set_ylabel("Maximum heading change")
    axis.set_title("Deterministic fixture: class-wise rarity features")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "rarity_feature_distributions.png", dpi=160)
    plt.close(figure)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["smoke", "fixture", "figures", "media", "full"], required=True
    )
    args = parser.parse_args()
    if args.mode in {"smoke", "fixture", "figures", "full"}:
        print(json.dumps(run_fixture(), indent=2))
    if args.mode == "media":
        print("Media generation is Pending Environment Validation for optional codecs.")


if __name__ == "__main__":
    main()
