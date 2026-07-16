from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rare_event_vru.baselines import constant_position, constant_velocity
from rare_event_vru.conformal import empirical_endpoint_coverage, fit_endpoint_radius
from rare_event_vru.features import extract_motion_features
from rare_event_vru.fixture import make_fixture
from rare_event_vru.metrics import ade_fde, minade_minfde
from rare_event_vru.models import MultiModalGRU, best_of_k_loss
from rare_event_vru.rarity import apply_tail_statistics, fit_tail_statistics
from rare_event_vru.transforms import to_global, to_local
from rare_event_vru.uncertainty import ensemble_decomposition


def test_transform_round_trip_and_stationary() -> None:
    trajectory = np.stack([np.arange(110), np.arange(110) * 0.5], axis=1).astype(np.float32)
    local, frame = to_local(trajectory, 50)
    np.testing.assert_allclose(to_global(local, frame), trajectory, atol=1e-5)
    stationary = np.zeros((110, 2), dtype=np.float32)
    local_stationary, stationary_frame = to_local(stationary, 50)
    np.testing.assert_allclose(to_global(local_stationary, stationary_frame), stationary)


def test_training_only_artifact_prevents_refit(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "scenario_id": [f"t{i}" for i in range(20)],
            "object_type": ["cyclist"] * 20,
            "max_speed": np.arange(20),
            "max_acceleration": np.arange(20),
            "max_heading_change": np.arange(20),
        }
    )
    validation = pd.DataFrame(
        {
            "scenario_id": ["v0", "v1"],
            "object_type": ["cyclist", "cyclist"],
            "max_speed": [0, 1000],
            "max_acceleration": [0, 1000],
            "max_heading_change": [0, 1000],
        }
    )
    artifact = fit_tail_statistics(train, quantile=0.9)
    path = tmp_path / "tail_statistics.json"
    artifact.save(path)
    before = json.loads(path.read_text())
    labelled = apply_tail_statistics(validation, artifact)
    assert before == json.loads(path.read_text())
    assert labelled.tail_event.tolist() == [0, 1]


def test_classwise_thresholds_avoid_global_bias() -> None:
    rows = []
    for name, offset in [("pedestrian", 0.0), ("motorcyclist", 100.0)]:
        for index in range(20):
            rows.append(
                {
                    "scenario_id": f"{name}-{index}",
                    "object_type": name,
                    "max_speed": offset + index,
                    "max_acceleration": offset + index,
                    "max_heading_change": offset + index,
                }
            )
    frame = pd.DataFrame(rows)
    labelled = apply_tail_statistics(frame, fit_tail_statistics(frame, quantile=0.9))
    rates = labelled.groupby("object_type").tail_event.mean()
    assert abs(rates["pedestrian"] - rates["motorcyclist"]) < 0.06


def test_baselines_metrics_and_features() -> None:
    trajectory = np.stack([np.arange(110) * 0.1, np.zeros(110)], axis=1)
    assert extract_motion_features(trajectory)["max_speed"] > 0
    prediction = constant_velocity(trajectory[:50], 60)
    ade, fde = ade_fde(prediction[None], trajectory[50:][None])
    assert ade[0] < 1e-8 and fde[0] < 1e-8
    modes = np.stack([prediction, constant_position(trajectory[:50], 60)])[None]
    min_ade, min_fde = minade_minfde(modes, trajectory[50:][None])
    assert min_ade[0] < 1e-8 and min_fde[0] < 1e-8


def test_multimodal_uncertainty_and_conformal() -> None:
    model = MultiModalGRU(future_steps=12, modes=4, hidden_size=16)
    past = torch.zeros(3, 20, 2)
    target = torch.zeros(3, 12, 2)
    trajectories, probabilities, scales = model(past)
    assert trajectories.shape == (3, 4, 12, 2)
    assert scales.shape == (3, 4, 12, 2)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(3))
    assert torch.isfinite(best_of_k_loss(trajectories, target))
    mean, epistemic = ensemble_decomposition(np.zeros((3, 4, 5, 2)))
    assert mean.shape == epistemic.shape == (4, 5, 2)
    predicted = np.zeros((10, 5, 2))
    truth = np.zeros((10, 5, 2))
    radius = fit_endpoint_radius(predicted[:6], truth[:6], coverage=0.9)
    assert empirical_endpoint_coverage(predicted[6:], truth[6:], radius) == 1.0


def test_fixture_is_deterministic() -> None:
    first, first_metadata = make_fixture(seed=7)
    second, second_metadata = make_fixture(seed=7)
    np.testing.assert_allclose(first, second)
    pd.testing.assert_frame_equal(first_metadata, second_metadata)
