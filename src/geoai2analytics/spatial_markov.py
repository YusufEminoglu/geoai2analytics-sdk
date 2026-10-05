# -*- coding: utf-8 -*-
"""Spatial Markov Chains & Cellular Land-Use Transition Engine for geoai2analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class SpatialMarkovResult:
    num_classes: int
    class_labels: list[str]
    global_transition_matrix: list[list[float]]
    spatial_conditional_matrices: dict[str, list[list[float]]]  # Keyed by spatial lag class state
    steady_state_distribution: list[float]
    chi2_spatial_independence_stat: float
    p_value_approx: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "classes": self.class_labels,
            "global_matrix": [[round(v, 4) for v in row] for row in self.global_transition_matrix],
            "steady_state": [round(v, 4) for v in self.steady_state_distribution],
            "chi2_stat": round(self.chi2_spatial_independence_stat, 2),
            "is_spatially_dependent": self.p_value_approx < 0.05,
        }


def simulate_landuse_transition(
    states_t0: Sequence[int | str],
    states_t1: Sequence[int | str],
    spatial_lag_states_t0: Sequence[int | str] | None = None,
    class_names: Sequence[str] | None = None,
) -> SpatialMarkovResult:
    """Compute global and spatially conditioned Markov transition probability matrices."""
    unique_states = sorted(set(list(states_t0) + list(states_t1)))
    k = len(unique_states)
    state_to_idx = {st: i for i, st in enumerate(unique_states)}
    labels = [str(c) for c in (class_names if class_names else unique_states)]

    # 1. Global Transition Counts
    global_counts = np.zeros((k, k), dtype=np.float64)
    for s0, s1 in zip(states_t0, states_t1):
        i = state_to_idx[s0]
        j = state_to_idx[s1]
        global_counts[i, j] += 1.0

    # Normalize to probabilities
    row_sums = global_counts.sum(axis=1, keepdims=True)
    global_mat = np.divide(global_counts, row_sums, out=np.zeros_like(global_counts), where=row_sums > 0)

    # 2. Spatially Conditioned Transitions
    cond_matrices: dict[str, list[list[float]]] = {}
    chi2_stat = 0.0

    if spatial_lag_states_t0 is not None:
        for lag_st in unique_states:
            cond_counts = np.zeros((k, k), dtype=np.float64)
            for s0, s1, l_st in zip(states_t0, states_t1, spatial_lag_states_t0):
                if l_st == lag_st:
                    cond_counts[state_to_idx[s0], state_to_idx[s1]] += 1.0

            c_row_sums = cond_counts.sum(axis=1, keepdims=True)
            c_mat = np.divide(cond_counts, c_row_sums, out=np.zeros_like(cond_counts), where=c_row_sums > 0)
            cond_matrices[str(lag_st)] = c_mat.tolist()

            # Chi-Square discrepancy: \sum (O - E)^2 / E
            for i in range(k):
                for j in range(k):
                    exp_val = c_row_sums[i, 0] * global_mat[i, j]
                    obs_val = cond_counts[i, j]
                    if exp_val > 1e-4:
                        chi2_stat += ((obs_val - exp_val) ** 2) / exp_val

    # 3. Steady State Distribution (Left eigenvector corresponding to eigenvalue 1)
    try:
        evals, evecs = np.linalg.eig(global_mat.T)
        idx = np.argmin(np.abs(evals - 1.0))
        steady = np.real(evecs[:, idx])
        steady = steady / np.sum(steady)
        steady_list = [float(v) for v in np.abs(steady)]
    except Exception:
        steady_list = [1.0 / k] * k

    # Approximate p-value
    dof = max(1, (k - 1) * (k - 1) * k)
    p_val = math.exp(-0.5 * chi2_stat / max(1.0, dof))

    return SpatialMarkovResult(
        num_classes=k,
        class_labels=labels,
        global_transition_matrix=global_mat.tolist(),
        spatial_conditional_matrices=cond_matrices,
        steady_state_distribution=steady_list,
        chi2_spatial_independence_stat=float(chi2_stat),
        p_value_approx=float(p_val),
    )
