from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.tail_gru import TailAwareGRU
from src.adaptive_topk import TailRiskTopKPolicy
from src.beam_codebook import UniformBeamCodebook
from src.tail_dataset import ArgoverseTailDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trajectory-to-beam control using the tail-aware GRU. "
            "The focal prediction is transformed back to the AV2 scenario "
            "frame and evaluated relative to the AV future trajectory."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/tail_aware_gru_v2.pth"),
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("outputs/tail_dataset_val.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=32)
    parser.add_argument("--min-angle-deg", type=float, default=-90.0)
    parser.add_argument("--max-angle-deg", type=float, default=90.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/beam_control_metrics.json"),
    )
    return parser.parse_args()


def inverse_local_transform(
    local_xy: np.ndarray,
    origin: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Map focal-centric coordinates back to the AV2 scenario frame."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = np.array(
        [[cos_a, -sin_a], [sin_a, cos_a]],
        dtype=np.float64,
    )
    return local_xy @ rotation.T + origin


def extract_av_future(scenario_path: str, future_steps: int) -> np.ndarray:
    """Read the AV trajectory, preferring the canonical AV track id."""
    frame = pd.read_parquet(
        scenario_path,
        columns=["track_id", "timestep", "observed", "position_x", "position_y"],
    )

    av = frame[frame["track_id"].astype(str).str.upper() == "AV"].copy()
    if av.empty:
        raise RuntimeError(f"No AV track found in {scenario_path}")

    av = av.sort_values("timestep")
    future = av[~av["observed"]][["position_x", "position_y"]].to_numpy(
        dtype=np.float64
    )

    if len(future) < future_steps:
        raise RuntimeError(
            f"AV future has {len(future)} points, expected at least {future_steps}: "
            f"{scenario_path}"
        )

    return future[:future_steps]


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    value = np.asarray(angle, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


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

    model = TailAwareGRU().to(device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    codebook = UniformBeamCodebook(
        num_beams=args.num_beams,
        min_angle_deg=args.min_angle_deg,
        max_angle_deg=args.max_angle_deg,
    )
    policy = TailRiskTopKPolicy()

    samples = 0
    top1_hits = 0
    adaptive_hits = 0
    fixed_top3_hits = 0
    exhaustive_beam_probes = 0
    adaptive_beam_probes = 0
    angular_errors_deg: list[float] = []
    shadow_hits = 0
    shadow_widths_deg: list[float] = []
    per_class: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for batch in loader:
            past = batch["past"].float().to(device)
            predicted_local, tail_logits = model(past)
            tail_probabilities = torch.sigmoid(tail_logits).cpu().numpy()
            predicted_local = predicted_local.cpu().numpy()
            target_local = batch["future"].numpy()
            origins = batch["origin"].numpy()
            angles = batch["angle"].numpy()

            for index, scenario_path in enumerate(batch["scenario_path"]):
                object_type = str(batch["object_type"][index])
                predicted_global = inverse_local_transform(
                    predicted_local[index], origins[index], float(angles[index])
                )
                target_global = inverse_local_transform(
                    target_local[index], origins[index], float(angles[index])
                )
                av_future = extract_av_future(
                    str(scenario_path), predicted_global.shape[0]
                )

                predicted_relative = predicted_global - av_future
                target_relative = target_global - av_future

                predicted_angle = float(
                    np.arctan2(predicted_relative[-1, 1], predicted_relative[-1, 0])
                )
                target_angle = float(
                    np.arctan2(target_relative[-1, 1], target_relative[-1, 0])
                )

                predicted_beam = int(codebook.angle_to_index(predicted_angle))
                target_beam = int(codebook.angle_to_index(target_angle))
                selected = policy.select(
                    codebook,
                    predicted_angle,
                    float(tail_probabilities[index]),
                )
                fixed_top3 = codebook.neighbouring_indices(predicted_beam, 3)

                top1_hit = int(predicted_beam == target_beam)
                adaptive_hit = int(target_beam in selected)
                fixed_top3_hit = int(target_beam in fixed_top3)

                angle_error = abs(float(wrap_angle(predicted_angle - target_angle)))
                angle_error_deg = float(np.rad2deg(angle_error))

                # Tail risk expands the predictive shadow interval. This is a
                # transparent first-order ADB proxy, not a photometric model.
                half_width_deg = 1.5 + 6.0 * float(tail_probabilities[index])
                shadow_hit = int(angle_error_deg <= half_width_deg)

                samples += 1
                top1_hits += top1_hit
                adaptive_hits += adaptive_hit
                fixed_top3_hits += fixed_top3_hit
                exhaustive_beam_probes += codebook.num_beams
                adaptive_beam_probes += len(selected)
                angular_errors_deg.append(angle_error_deg)
                shadow_hits += shadow_hit
                shadow_widths_deg.append(2.0 * half_width_deg)

                class_stats = per_class.setdefault(
                    object_type,
                    {
                        "samples": 0,
                        "top1_hits": 0,
                        "adaptive_hits": 0,
                        "angular_error_sum_deg": 0.0,
                        "beam_probes": 0,
                    },
                )
                class_stats["samples"] += 1
                class_stats["top1_hits"] += top1_hit
                class_stats["adaptive_hits"] += adaptive_hit
                class_stats["angular_error_sum_deg"] += angle_error_deg
                class_stats["beam_probes"] += len(selected)

    average_k = safe_rate(adaptive_beam_probes, samples)
    metrics = {
        "evaluation_frame": (
            "future focal-agent line-of-sight relative to the AV future trajectory"
        ),
        "samples": samples,
        "num_beams": codebook.num_beams,
        "field_of_view_deg": [args.min_angle_deg, args.max_angle_deg],
        "top1_accuracy": safe_rate(top1_hits, samples),
        "fixed_top3_coverage": safe_rate(fixed_top3_hits, samples),
        "adaptive_topk_coverage": safe_rate(adaptive_hits, samples),
        "average_k": average_k,
        "beam_overhead_reduction_vs_exhaustive": (
            1.0 - average_k / codebook.num_beams
        ),
        "mean_final_angular_error_deg": float(np.mean(angular_errors_deg)),
        "median_final_angular_error_deg": float(np.median(angular_errors_deg)),
        "predictive_adb_shadow_coverage": safe_rate(shadow_hits, samples),
        "mean_adb_shadow_width_deg": float(np.mean(shadow_widths_deg)),
        "notes": [
            "The AV future trajectory is treated as known ego motion, as in planned-motion control.",
            "Beam labels are geometry-derived; no measured optical channel or multipath is used.",
            "The ADB metric is an angular-coverage proxy rather than a full photometric SAE evaluation.",
        ],
        "per_class": {},
    }

    for object_type, values in sorted(per_class.items()):
        count = int(values["samples"])
        metrics["per_class"][object_type] = {
            "samples": count,
            "top1_accuracy": safe_rate(int(values["top1_hits"]), count),
            "adaptive_topk_coverage": safe_rate(
                int(values["adaptive_hits"]), count
            ),
            "mean_final_angular_error_deg": safe_rate(
                values["angular_error_sum_deg"], count
            ),
            "average_k": safe_rate(int(values["beam_probes"]), count),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("TAIL-RISK ADAPTIVE BEAM / ADB EVALUATION")
    print("=" * 64)
    print(f"Samples:                    {samples}")
    print(f"Top-1 beam accuracy:        {metrics['top1_accuracy']:.4f}")
    print(f"Fixed Top-3 coverage:       {metrics['fixed_top3_coverage']:.4f}")
    print(f"Adaptive Top-K coverage:    {metrics['adaptive_topk_coverage']:.4f}")
    print(f"Average K:                  {metrics['average_k']:.3f}")
    print(
        "Overhead reduction:          "
        f"{metrics['beam_overhead_reduction_vs_exhaustive']:.4f}"
    )
    print(
        "Mean angular error:          "
        f"{metrics['mean_final_angular_error_deg']:.3f} deg"
    )
    print(
        "Predictive ADB coverage:     "
        f"{metrics['predictive_adb_shadow_coverage']:.4f}"
    )
    print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()
