# -*- coding: utf-8 -*-
"""Ripley's K and Besag's L Bivariate Cross-Function Spatial Point Process Analyzer for geoai2analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class RipleysKCrossResult:
    radius_distances_r: list[float]
    observed_k_cross_values: list[float]
    theoretical_k_csr_values: list[float]  # \pi * r^2
    besag_l_cross_values: list[float]      # \sqrt{K(r)/\pi} - r
    is_spatial_attraction: bool
    is_spatial_repulsion: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "radii_count": len(self.radius_distances_r),
            "max_l_val": round(max(self.besag_l_cross_values) if self.besag_l_cross_values else 0.0, 3),
            "attraction": self.is_spatial_attraction,
            "repulsion": self.is_spatial_repulsion,
        }


def calculate_ripleys_k_bivariate(
    points_type_a: Sequence[tuple[float, float]],
    points_type_b: Sequence[tuple[float, float]],
    study_area_m2: float,
    max_radius_r_m: float = 1000.0,
    radius_steps_count: int = 10,
) -> RipleysKCrossResult:
    """Compute Bivariate Ripley's K_12(r) cross-interaction metric testing spatial attraction vs repulsion."""
    pts_a = list(points_type_a)
    pts_b = list(points_type_b)
    n_a = len(pts_a)
    n_b = len(pts_b)

    if n_a == 0 or n_b == 0 or study_area_m2 <= 0:
        return RipleysKCrossResult([], [], [], [], False, False)

    lambda_b = n_b / study_area_m2

    step_size = max_radius_r_m / float(radius_steps_count)
    radii = [step_size * (i + 1) for i in range(radius_steps_count)]

    k_obs_list: list[float] = []
    k_csr_list: list[float] = []
    l_cross_list: list[float] = []

    # Pre-calculate pairwise distances
    dist_matrix = []
    for pa in pts_a:
        row = [math.hypot(pa[0] - pb[0], pa[1] - pb[1]) for pb in pts_b]
        dist_matrix.append(row)

    for r in radii:
        # Sum indicator functions I(d_ij <= r)
        count_in_radius = 0
        for row in dist_matrix:
            for d in row:
                if d <= r:
                    count_in_radius += 1

        k_val = (study_area_m2 / (n_a * n_b)) * count_in_radius
        k_csr = math.pi * (r**2)
        l_val = math.sqrt(k_val / math.pi) - r

        k_obs_list.append(round(k_val, 2))
        k_csr_list.append(round(k_csr, 2))
        l_cross_list.append(round(l_val, 2))

    # Test attraction vs repulsion based on mean L(r)
    mean_l = sum(l_cross_list) / len(l_cross_list)
    attraction = mean_l > (step_size * 0.1)
    repulsion = mean_l < -(step_size * 0.1)

    return RipleysKCrossResult(
        radius_distances_r=radii,
        observed_k_cross_values=k_obs_list,
        theoretical_k_csr_values=k_csr_list,
        besag_l_cross_values=l_cross_list,
        is_spatial_attraction=attraction,
        is_spatial_repulsion=repulsion,
    )
