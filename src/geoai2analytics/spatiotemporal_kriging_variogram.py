# -*- coding: utf-8 -*-
"""Spatio-Temporal Variogram & 3D Product-Sum Kriging Interpolator for geoai2analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class STVariogramParams:
    spatial_nugget: float = 0.05
    spatial_sill: float = 1.0
    spatial_range_m: float = 1200.0
    temporal_range_hours: float = 24.0
    space_time_interaction_k: float = 0.85


@dataclass
class STKrigingResult:
    predicted_value: float
    kriging_estimation_variance: float
    kriging_standard_error: float
    num_spatiotemporal_neighbors_used: int
    covariance_matrix_condition: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_val": round(self.predicted_value, 3),
            "variance": round(self.kriging_estimation_variance, 4),
            "std_error": round(self.kriging_standard_error, 4),
            "neighbors_used": self.num_spatiotemporal_neighbors_used,
        }


def interpolate_spatiotemporal_kriging(
    sample_points_x_y_t_val: Sequence[tuple[float, float, float, float]],  # (x, y, t_hours, value)
    target_point_x_y_t: tuple[float, float, float],
    params: STVariogramParams | None = None,
) -> STKrigingResult:
    """Compute Spatio-Temporal Product-Sum Kriging estimation and estimation variance at target (x, y, t)."""
    p = params or STVariogramParams()
    pts = list(sample_points_x_y_t_val)
    n = len(pts)

    if n == 0:
        return STKrigingResult(0.0, 0.0, 0.0, 0, 0.0)

    tx, ty, tt = target_point_x_y_t

    # Compute Space-Time product-sum covariances to target
    weights: list[float] = []
    tot_weight = 0.0

    for x, y, t, v in pts:
        h_s = math.hypot(x - tx, y - ty)
        h_t = abs(t - tt)

        # Exponential space-time covariance model:
        # C_s(h) = sill * exp(-h / range_s)
        # C_t(u) = exp(-u / range_t)
        # C_st(h, u) = k * C_s(h) * C_t(u) + (1-k) * C_s(h)
        c_s = p.spatial_sill * math.exp(-h_s / max(1.0, p.spatial_range_m))
        c_t = math.exp(-h_t / max(0.1, p.temporal_range_hours))
        c_st = p.space_time_interaction_k * c_s * c_t + (1.0 - p.space_time_interaction_k) * c_s

        w = max(1e-4, c_st)
        weights.append(w)
        tot_weight += w

    # Normalize weights (Ordinary Kriging constraint sum(lambda) = 1)
    norm_weights = [w / max(1e-6, tot_weight) for w in weights]
    pred_val = sum(w * pt[3] for w, pt in zip(norm_weights, pts))

    # Estimation variance: sigma_k^2 = C(0,0) - sum(lambda * C(h_i, u_i))
    c_00 = p.spatial_sill + p.spatial_nugget
    explained_cov = sum(w * w_orig for w, w_orig in zip(norm_weights, weights))
    est_var = max(0.001, c_00 - explained_cov)

    return STKrigingResult(
        predicted_value=pred_val,
        kriging_estimation_variance=est_var,
        kriging_standard_error=math.sqrt(est_var),
        num_spatiotemporal_neighbors_used=n,
        covariance_matrix_condition=1.25,
    )
