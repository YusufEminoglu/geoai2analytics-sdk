# -*- coding: utf-8 -*-
"""Spatial Interaction, Retail Gravitation & Gravity Flow Engines for geoai2analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class HuffModelResult:
    """Probabilistic market share matrix and estimated store patrons."""

    probabilities: list[list[float]]  # (N_origins, N_destinations)
    expected_patrons_by_destination: list[float]
    origin_total_demand: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_origins": len(self.probabilities),
            "n_destinations": len(self.expected_patrons_by_destination),
            "expected_patrons": [round(v, 2) for v in self.expected_patrons_by_destination],
        }


@dataclass
class ReillyResult:
    """Reilly's Law of Retail Gravitation breaking-point result."""

    distance_km: float
    breaking_point_from_a_km: float
    market_boundary_ratio: float


@dataclass
class WilsonFlowResult:
    """Doubly-constrained entropy-maximizing spatial interaction matrix."""

    flow_matrix: list[list[float]]
    total_flow: float
    origin_balancing_factors: list[float]
    destination_balancing_factors: list[float]


def huff_model(
    origin_coords: Sequence[tuple[float, float]],
    origin_demands: Sequence[float],
    destination_coords: Sequence[tuple[float, float]],
    destination_attractiveness: Sequence[float],
    distance_decay_exponent: float = 2.0,
) -> HuffModelResult:
    r"""Huff's Probabilistic Retail Market Share Model.

    P_{ij} = (S_j / D_{ij}^\lambda) / \sum_k (S_k / D_{ik}^\lambda)
    """
    n_orig = len(origin_coords)
    n_dest = len(destination_coords)

    prob_matrix: list[list[float]] = []
    dest_patrons = [0.0] * n_dest

    for i in range(n_orig):
        ox, oy = origin_coords[i]
        d_i = origin_demands[i]

        utilities = []
        for j in range(n_dest):
            dx, dy = destination_coords[j]
            dist = max(10.0, math.hypot(ox - dx, oy - dy))  # Min distance 10m
            s_j = max(1.0, destination_attractiveness[j])
            u_ij = s_j / (dist ** distance_decay_exponent)
            utilities.append(u_ij)

        sum_u = sum(utilities)
        row_probs = [(u / sum_u if sum_u > 0 else 1.0 / n_dest) for u in utilities]
        prob_matrix.append(row_probs)

        for j in range(n_dest):
            dest_patrons[j] += d_i * row_probs[j]

    return HuffModelResult(
        probabilities=prob_matrix,
        expected_patrons_by_destination=dest_patrons,
        origin_total_demand=list(origin_demands),
    )


def reilly_law_breaking_point(
    population_a: float,
    population_b: float,
    distance_km: float,
) -> ReillyResult:
    r"""Calculate the breaking point distance between two retail centers using Reilly's Law.

    D_A = Distance / (1 + \sqrt{P_B / P_A})
    """
    ratio = population_b / max(1.0, population_a)
    d_a = distance_km / (1.0 + math.sqrt(ratio))
    return ReillyResult(
        distance_km=distance_km,
        breaking_point_from_a_km=d_a,
        market_boundary_ratio=d_a / max(1e-4, distance_km),
    )


def wilson_spatial_interaction(
    origins_supply: Sequence[float],
    destinations_demand: Sequence[float],
    cost_matrix: Sequence[Sequence[float]] | np.ndarray,
    beta: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> WilsonFlowResult:
    """Doubly-constrained entropy-maximizing spatial interaction gravity model (Wilson 1971).

    T_{ij} = A_i * B_j * O_i * D_j * exp(-\beta * c_{ij})
    """
    O_supply = np.asarray(origins_supply, dtype=float)
    D = np.asarray(destinations_demand, dtype=float)
    C = np.asarray(cost_matrix, dtype=float)

    n_orig, n_dest = len(O_supply), len(D)
    f_mat = np.exp(-beta * C)

    A = np.ones(n_orig)
    B = np.ones(n_dest)

    for _ in range(max_iter):
        A_old = A.copy()
        # Update A_i = 1 / \sum_j (B_j * D_j * f_{ij})
        denom_A = np.dot(f_mat, B * D)
        A = 1.0 / np.maximum(1e-9, denom_A)

        # Update B_j = 1 / \sum_i (A_i * O_i * f_{ij})
        denom_B = np.dot(f_mat.T, A * O_supply)
        B = 1.0 / np.maximum(1e-9, denom_B)

        if np.max(np.abs(A - A_old)) < tol:
            break

    # Flow matrix T_{ij} = A_i * O_i * B_j * D_j * f_{ij}
    T_mat = (A * O_supply)[:, None] * (B * D)[None, :] * f_mat

    return WilsonFlowResult(
        flow_matrix=T_mat.tolist(),
        total_flow=float(np.sum(T_mat)),
        origin_balancing_factors=A.tolist(),
        destination_balancing_factors=B.tolist(),
    )
