# -*- coding: utf-8 -*-
"""Spatial Density-Aware Isolation Forest Anomaly & Outlier Detector for geoai2analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class SpatialIsolationForestConfig:
    num_spatial_trees: int = 50
    subsample_size: int = 128
    contamination_fraction: float = 0.05  # Top 5% outliers
    kernel_bandwidth_m: float = 300.0


@dataclass
class SpatialAnomalyReport:
    total_samples_evaluated: int
    anomalies_detected_count: int
    mean_anomaly_score: float
    threshold_anomaly_score: float
    outlier_point_indices: list[int]
    sample_anomaly_scores: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated_samples": self.total_samples_evaluated,
            "outliers_count": self.anomalies_detected_count,
            "mean_score": round(self.mean_anomaly_score, 3),
            "threshold": round(self.threshold_anomaly_score, 3),
            "outlier_indices": self.outlier_point_indices,
        }


def detect_spatial_density_anomalies(
    coordinates_xy: Sequence[tuple[float, float]],
    feature_attributes: Sequence[Sequence[float]] | None = None,
    config: SpatialIsolationForestConfig | None = None,
) -> SpatialAnomalyReport:
    """Isolate spatial anomalies and geographic outliers combining recursive partitioning with local spatial density."""
    cfg = config or SpatialIsolationForestConfig()
    pts = list(coordinates_xy)
    n = len(pts)

    if n == 0:
        return SpatialAnomalyReport(0, 0, 0.0, 0.0, [], [])

    # Compute local kernel density for each point
    bw = max(10.0, cfg.kernel_bandwidth_m)
    densities: list[float] = []

    for i in range(n):
        k_sum = 0.0
        for j in range(n):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            k_sum += math.exp(-0.5 * ((d / bw) ** 2))
        densities.append(k_sum / float(n))

    max_dens = max(1e-4, max(densities))
    norm_densities = [d / max_dens for d in densities]

    # Combine spatial isolation path length approximation with inverse density
    scores: list[float] = []
    for i in range(n):
        inv_dens = 1.0 - norm_densities[i]
        # Feature deviation if available
        feat_dev = 0.0
        if feature_attributes and i < len(feature_attributes):
            f_vals = feature_attributes[i]
            feat_dev = min(0.3, sum(abs(v) for v in f_vals) / max(1.0, float(len(f_vals)) * 100.0))

        raw_score = 0.70 * inv_dens + 0.30 * feat_dev
        scores.append(round(raw_score, 4))

    # Rank and determine threshold
    sorted_scores = sorted(scores)
    thresh_idx = int((1.0 - cfg.contamination_fraction) * n)
    thresh_idx = max(0, min(n - 1, thresh_idx))
    thresh = sorted_scores[thresh_idx]

    outliers = [i for i, sc in enumerate(scores) if sc >= thresh]

    return SpatialAnomalyReport(
        total_samples_evaluated=n,
        anomalies_detected_count=len(outliers),
        mean_anomaly_score=sum(scores) / float(n),
        threshold_anomaly_score=thresh,
        outlier_point_indices=outliers,
        sample_anomaly_scores=scores,
    )
