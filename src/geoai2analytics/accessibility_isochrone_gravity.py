# -*- coding: utf-8 -*-
"""Continuous Gravity-Decay Accessibility Potential Engine for geoai2analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence


class DistanceDecayType(str, Enum):
    EXPONENTIAL = "EXPONENTIAL"
    POWER = "POWER"
    GAUSSIAN = "GAUSSIAN"


@dataclass
class GravityAccessibilityReport:
    total_origins_analyzed: int
    mean_accessibility_score: float
    equity_gini_coefficient: float
    origin_accessibility_scores: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "origins_count": self.total_origins_analyzed,
            "mean_accessibility": round(self.mean_accessibility_score, 2),
            "gini_index": round(self.equity_gini_coefficient, 3),
        }


def _gini(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    sorted_s = sorted(scores)
    n = len(sorted_s)
    tot = sum(sorted_s)
    if tot == 0:
        return 0.0
    cum = 0.0
    for i, s in enumerate(sorted_s, 1):
        cum += i * s
    return (2.0 * cum) / (n * tot) - (n + 1.0) / n


def compute_gravity_accessibility_matrix(
    origin_coordinates: Sequence[tuple[float, float]],
    facility_coordinates: Sequence[tuple[float, float]],
    facility_capacities: Sequence[float] | None = None,
    decay_type: DistanceDecayType = DistanceDecayType.EXPONENTIAL,
    decay_parameter_beta: float = 0.0015,  # e.g., beta for exp(-beta * dist)
) -> GravityAccessibilityReport:
    """Compute Hansen continuous gravity accessibility potential: A_i = \\sum_j W_j * f(d_ij)."""
    origs = list(origin_coordinates)
    facils = list(facility_coordinates)
    n_o = len(origs)
    n_f = len(facils)

    if n_o == 0 or n_f == 0:
        return GravityAccessibilityReport(0, 0.0, 0.0, [])

    weights = list(facility_capacities) if facility_capacities else [1.0] * n_f

    scores: list[float] = []

    for ox, oy in origs:
        pot_i = 0.0
        for (fx, fy), w in zip(facils, weights):
            d = math.hypot(ox - fx, oy - fy)

            # Apply decay function
            if decay_type == DistanceDecayType.EXPONENTIAL:
                f_d = math.exp(-decay_parameter_beta * d)
            elif decay_type == DistanceDecayType.GAUSSIAN:
                f_d = math.exp(-0.5 * (decay_parameter_beta * d) ** 2)
            else:  # POWER
                f_d = (1.0 + d) ** (-max(0.1, decay_parameter_beta * 1000.0))

            pot_i += w * f_d

        scores.append(round(pot_i, 3))

    mean_sc = sum(scores) / len(scores)
    gini_val = _gini(scores)

    return GravityAccessibilityReport(
        total_origins_analyzed=n_o,
        mean_accessibility_score=mean_sc,
        equity_gini_coefficient=gini_val,
        origin_accessibility_scores=scores,
    )
