# -*- coding: utf-8 -*-
"""Multi-Scale Spatial Information Entropy & Urban Sprawl Dispersion Engine (Batty & Shannon) for geoai2analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class ScaleEntropyProfile:
    grid_resolution_m: float
    shannon_entropy: float
    max_theoretical_entropy: float
    normalized_entropy: float  # 0.0 (concentrated monocentric) to 1.0 (uniform dispersion / sprawl)
    occupied_cells_count: int


@dataclass
class SpatialEntropyReport:
    total_features_count: int
    mean_normalized_entropy: float
    is_high_urban_sprawl: bool  # True if normalized entropy > 0.70 across scales
    scale_profiles: list[ScaleEntropyProfile]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_features": self.total_features_count,
            "mean_entropy": round(self.mean_normalized_entropy, 3),
            "high_sprawl": self.is_high_urban_sprawl,
            "scales_evaluated": len(self.scale_profiles),
        }


def calculate_spatial_information_entropy(
    point_coordinates: Sequence[tuple[float, float]],
    scale_grid_resolutions_m: Sequence[float] = (100.0, 250.0, 500.0, 1000.0),
) -> SpatialEntropyReport:
    """Compute Batty & Shannon multi-scale spatial information entropy across hierarchical grid tessellations.

    Formula:
    H = - sum_i (p_i * ln(p_i)) where p_i = n_i / N
    H_max = ln(K) where K = number of occupied / total cells
    H_norm = H / H_max
    """
    pts = list(point_coordinates)
    n = len(pts)
    if n == 0:
        return SpatialEntropyReport(0, 0.0, False, [])

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x = min(xs)
    min_y = min(ys)

    profiles: list[ScaleEntropyProfile] = []
    norm_entropies: list[float] = []

    for res in scale_grid_resolutions_m:
        # Bin points into 2D grid cells
        cells: dict[tuple[int, int], int] = {}
        for x, y in pts:
            gx = int((x - min_x) // res)
            gy = int((y - min_y) // res)
            cells[(gx, gy)] = cells.get((gx, gy), 0) + 1

        k_occupied = len(cells)
        if k_occupied <= 1:
            profiles.append(ScaleEntropyProfile(res, 0.0, 1.0, 0.0, k_occupied))
            norm_entropies.append(0.0)
            continue

        # Compute Shannon entropy
        h_entropy = 0.0
        for count in cells.values():
            p_i = count / float(n)
            if p_i > 0:
                h_entropy -= p_i * math.log(p_i)

        h_max = math.log(k_occupied)
        h_norm = min(1.0, max(0.0, h_entropy / max(1e-4, h_max)))

        profiles.append(
            ScaleEntropyProfile(
                grid_resolution_m=res,
                shannon_entropy=h_entropy,
                max_theoretical_entropy=h_max,
                normalized_entropy=h_norm,
                occupied_cells_count=k_occupied,
            )
        )
        norm_entropies.append(h_norm)

    mean_norm = sum(norm_entropies) / max(1, len(norm_entropies))
    is_sprawl = mean_norm > 0.70

    return SpatialEntropyReport(
        total_features_count=n,
        mean_normalized_entropy=mean_norm,
        is_high_urban_sprawl=is_sprawl,
        scale_profiles=profiles,
    )
