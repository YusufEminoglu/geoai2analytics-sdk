# -*- coding: utf-8 -*-
"""Inhomogeneous Poisson Point Process & Adaptive K-Nearest Intensity Estimator for geoai2analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class SpatialBandwidthProfile:
    point_index: int
    k_distance_m: float
    local_intensity: float


@dataclass
class PointProcessIntensityResult:
    total_events_count: int
    mean_intensity_per_km2: float
    max_hotspot_intensity_per_km2: float
    k_neighbors_used: int
    adaptive_intensities: list[float]
    bandwidth_profiles: list[SpatialBandwidthProfile]

    def to_dict(self) -> dict[str, Any]:
        return {
            "events_count": self.total_events_count,
            "mean_intensity": round(self.mean_intensity_per_km2, 2),
            "max_hotspot_intensity": round(self.max_hotspot_intensity_per_km2, 2),
            "k_neighbors": self.k_neighbors_used,
        }


def estimate_inhomogeneous_intensity(
    event_coordinates: Sequence[tuple[float, float]],
    k_nearest_neighbors: int = 4,
    study_area_km2: float | None = None,
) -> PointProcessIntensityResult:
    """Compute variable bandwidth adaptive spatial point process intensity using k-th nearest neighbor distance.
    
    Formula:
    lambda_hat(x_i) = (k - 1) / (pi * d_k(x_i)^2)
    """
    pts = list(event_coordinates)
    n = len(pts)
    if n == 0:
        return PointProcessIntensityResult(0, 0.0, 0.0, k_nearest_neighbors, [], [])

    k = min(n - 1, max(1, k_nearest_neighbors))

    intensities: list[float] = []
    profiles: list[SpatialBandwidthProfile] = []

    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                dists.append(d)
        dists.sort()
        d_k = max(1.0, dists[k - 1] if len(dists) >= k else 10.0)

        # Intensity per km^2: d_k in meters -> d_k_km = d_k / 1000
        d_k_km = d_k / 1000.0
        lam = (k - 1) / (math.pi * (d_k_km ** 2)) if k > 1 else (1.0 / (math.pi * (d_k_km ** 2)))
        intensities.append(lam)

        profiles.append(
            SpatialBandwidthProfile(
                point_index=i,
                k_distance_m=d_k,
                local_intensity=lam,
            )
        )

    mean_lam = sum(intensities) / float(n)
    max_lam = max(intensities) if intensities else 0.0

    return PointProcessIntensityResult(
        total_events_count=n,
        mean_intensity_per_km2=mean_lam,
        max_hotspot_intensity_per_km2=max_lam,
        k_neighbors_used=k,
        adaptive_intensities=intensities,
        bandwidth_profiles=profiles,
    )
