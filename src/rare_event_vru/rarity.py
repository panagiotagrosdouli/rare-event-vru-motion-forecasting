from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_FEATURES = ("max_speed", "max_acceleration", "max_heading_change")


@dataclass(frozen=True)
class ClassStatistics:
    median: dict[str, float]
    iqr: dict[str, float]
    threshold: float


@dataclass(frozen=True)
class TailStatisticsArtifact:
    version: str
    method: str
    quantile: float
    features: tuple[str, ...]
    weights: dict[str, float]
    classes: dict[str, ClassStatistics]
    train_fingerprint: str
    seed: int
    created_at: str

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TailStatisticsArtifact":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["features"] = tuple(payload["features"])
        payload["classes"] = {k: ClassStatistics(**v) for k, v in payload["classes"].items()}
        return cls(**payload)


def fingerprint_frame(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    stable = frame.loc[:, list(columns)].sort_values(list(columns)).to_csv(index=False)
    return hashlib.sha256(stable.encode()).hexdigest()


def _robust_scale(values: pd.Series, median: float, iqr: float) -> np.ndarray:
    return (values.to_numpy(dtype=float) - median) / max(iqr, 1e-8)


def fit_tail_statistics(
    train: pd.DataFrame,
    quantile: float = 0.95,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    weights: dict[str, float] | None = None,
    seed: int = 0,
) -> TailStatisticsArtifact:
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be between 0.5 and 1")
    required = {"scenario_id", "object_type", *features}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"missing training columns: {sorted(missing)}")
    weights = weights or {name: 1.0 / len(features) for name in features}
    if set(weights) != set(features) or any(value < 0 for value in weights.values()):
        raise ValueError("weights must be non-negative and exactly match features")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    weights = {name: value / total for name, value in weights.items()}
    classes: dict[str, ClassStatistics] = {}
    for object_type, subset in train.groupby("object_type", sort=True):
        medians = {feature: float(subset[feature].median()) for feature in features}
        iqrs = {
            feature: float(subset[feature].quantile(0.75) - subset[feature].quantile(0.25))
            for feature in features
        }
        score = sum(
            weights[feature]
            * np.maximum(_robust_scale(subset[feature], medians[feature], iqrs[feature]), 0.0)
            for feature in features
        )
        classes[str(object_type)] = ClassStatistics(
            median=medians,
            iqr=iqrs,
            threshold=float(np.quantile(score, quantile)),
        )
    return TailStatisticsArtifact(
        version="tail-stats-v1",
        method="classwise_positive_robust_z",
        quantile=quantile,
        features=features,
        weights=weights,
        classes=classes,
        train_fingerprint=fingerprint_frame(train, ("scenario_id", "object_type", *features)),
        seed=seed,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def apply_tail_statistics(frame: pd.DataFrame, artifact: TailStatisticsArtifact) -> pd.DataFrame:
    result = frame.copy()
    scores = np.empty(len(result), dtype=float)
    labels = np.empty(len(result), dtype=np.int64)
    reasons: list[str] = []
    for position, (_, row) in enumerate(result.iterrows()):
        object_type = str(row["object_type"])
        if object_type not in artifact.classes:
            raise ValueError(f"class {object_type!r} missing from training artifact")
        stats = artifact.classes[object_type]
        contributions = {
            feature: artifact.weights[feature]
            * max(
                (float(row[feature]) - stats.median[feature])
                / max(stats.iqr[feature], 1e-8),
                0.0,
            )
            for feature in artifact.features
        }
        score = sum(contributions.values())
        scores[position] = score
        labels[position] = int(score >= stats.threshold)
        reasons.append(max(contributions, key=contributions.get))
    result["tail_score"] = scores
    result["tail_event"] = labels
    result["rarity_reason"] = reasons
    result["tail_statistics_version"] = artifact.version
    return result
