# -*- coding: utf-8 -*-
"""Spatio-Temporal Autocorrelation & Space-Time Emerging Hotspot Analysis (ESTA)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .autocorr import LocalGetisOrdResult, local_getis_ord_gi_star
from .weights import SpatialWeights, knn_weights


@dataclass
class MannKendallResult:
    """Non-parametric Mann-Kendall trend test result."""

    tau: float
    s_stat: float
    var_s: float
    z_score: float
    p_value: float
    trend: str  # 'increasing', 'decreasing', 'no trend'
    sen_slope: float


@dataclass
class EmergingHotspotResult:
    """Classification of spatio-temporal hotspot patterns over time."""

    pattern_type: str  # 'New Hotspot', 'Intensifying Hotspot', 'Persistent Hotspot', 'Sporadic Hotspot', etc.
    gi_z_series: list[float]
    gi_p_series: list[float]
    mann_kendall: MannKendallResult
    recent_hotspot: bool
    historic_hotspot: bool


@dataclass
class SpaceTimeAnalysisResult:
    """Master output of Space-Time Emerging Hotspot Analysis."""

    n_locations: int
    n_time_steps: int
    results: list[EmergingHotspotResult]
    pattern_summary: dict[str, int]
    global_spacetime_moran_i: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_locations": self.n_locations,
            "n_time_steps": self.n_time_steps,
            "pattern_summary": self.pattern_summary,
            "global_spacetime_moran_i": round(self.global_spacetime_moran_i, 4),
            "patterns": [
                {
                    "index": i,
                    "pattern": r.pattern_type,
                    "z_latest": round(r.gi_z_series[-1], 3) if r.gi_z_series else 0.0,
                    "trend_z": round(r.mann_kendall.z_score, 3),
                    "trend_p": round(r.mann_kendall.p_value, 4),
                }
                for i, r in enumerate(self.results)
            ],
        }


class SpaceTimeCube:
    """3D Space-Time data container storing multi-period observations across locations."""

    def __init__(
        self,
        coords: Sequence[tuple[float, float]] | np.ndarray,
        time_series_matrix: Sequence[Sequence[float]] | np.ndarray,
        time_labels: Sequence[str] | None = None,
    ) -> None:
        """Args:

        coords: (N, 2) spatial coordinates.
        time_series_matrix: (N, T) matrix of values where rows are locations, columns are time steps.
        time_labels: Optional labels for each time slice.
        """
        self.coords = np.asarray(coords, dtype=float)
        self.data = np.asarray(time_series_matrix, dtype=float)
        self.n_locations, self.n_time_steps = self.data.shape

        if self.coords.shape[0] != self.n_locations:
            raise ValueError(
                f"Coords count ({self.coords.shape[0]}) does not match rows in matrix ({self.n_locations})"
            )

        self.time_labels = (
            list(time_labels)
            if time_labels is not None
            else [f"T{t+1}" for t in range(self.n_time_steps)]
        )


def mann_kendall_test(series: Sequence[float] | np.ndarray) -> MannKendallResult:
    """Compute Mann-Kendall non-parametric trend test and Sen's slope."""
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    if n < 3:
        return MannKendallResult(
            tau=0.0, s_stat=0.0, var_s=1.0, z_score=0.0, p_value=1.0, trend="no trend", sen_slope=0.0
        )

    # Compute S statistic
    s = 0.0
    slopes: list[float] = []
    for k in range(n - 1):
        for j in range(k + 1, n):
            diff = arr[j] - arr[k]
            s += np.sign(diff)
            if j != k:
                slopes.append(diff / (j - k))

    # Variance of S (assuming no ties for simplicity, or with tie correction)
    unique_vals, counts = np.unique(arr, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    var_s = max(var_s, 1e-6)

    # Standardized Z score
    if s > 0:
        z = (s - 1.0) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / math.sqrt(var_s)
    else:
        z = 0.0

    # Two-tailed p-value approximation via standard normal error function
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    p_value = max(0.0, min(1.0, p_value))

    # Kendall tau
    tau = s / (0.5 * n * (n - 1))

    # Sen's median slope
    sen_slope = float(np.median(slopes)) if slopes else 0.0

    trend = "no trend"
    if p_value < 0.05:
        trend = "increasing" if z > 0 else "decreasing"

    return MannKendallResult(
        tau=float(tau),
        s_stat=float(s),
        var_s=float(var_s),
        z_score=float(z),
        p_value=float(p_value),
        trend=trend,
        sen_slope=sen_slope,
    )


def spatiotemporal_moran(
    cube: SpaceTimeCube,
    spatial_weights: SpatialWeights | None = None,
) -> float:
    """Compute Spatio-Temporal Moran's I across space-time pooled matrix."""
    if spatial_weights is None:
        spatial_weights = knn_weights(cube.coords, k=min(6, cube.n_locations - 1))

    _, T = cube.n_locations, cube.n_time_steps
    flat_y = cube.data.flatten()
    mean_y = np.mean(flat_y)
    z = flat_y - mean_y
    ss = np.sum(z**2)
    if ss < 1e-9:
        return 0.0

    # Construct space-time lag matrix product
    # For each time step t, compute spatial lag
    lag_z = np.zeros_like(cube.data)
    for t in range(T):
        lag_z[:, t] = np.dot(spatial_weights.matrix, cube.data[:, t] - mean_y)

    flat_lag = lag_z.flatten()
    s0 = float(np.sum(spatial_weights.matrix)) * T
    if s0 < 1e-9:
        return 0.0

    i_val = (len(flat_y) / s0) * (np.sum(z * flat_lag) / ss)
    return float(i_val)


def emerging_hotspot_analysis(
    cube: SpaceTimeCube,
    spatial_weights: SpatialWeights | None = None,
) -> SpaceTimeAnalysisResult:
    """Perform Space-Time Emerging Hotspot Analysis (ESTA).

    Evaluates local Getis-Ord Gi* z-scores across all time steps and classifies
    each spatial unit into standard hotspot/coldspot evolutionary categories.
    """
    if spatial_weights is None:
        spatial_weights = knn_weights(cube.coords, k=min(6, cube.n_locations - 1))

    N, T = cube.n_locations, cube.n_time_steps

    # Compute Gi* z-scores for each time step
    gi_z_matrix = np.zeros((N, T), dtype=float)
    gi_p_matrix = np.zeros((N, T), dtype=float)

    for t in range(T):
        res: LocalGetisOrdResult = local_getis_ord_gi_star(cube.data[:, t], spatial_weights)
        gi_z_matrix[:, t] = res.gi_star
        gi_p_matrix[:, t] = res.p_values

    results: list[EmergingHotspotResult] = []
    pattern_summary: dict[str, int] = {}

    for i in range(N):
        z_series = gi_z_matrix[i, :].tolist()
        p_series = gi_p_matrix[i, :].tolist()
        mk = mann_kendall_test(z_series)

        # Classification logic based on recent and historical significant time steps (z > 1.96 / z < -1.96)
        is_hot = [z > 1.96 for z in z_series]
        is_cold = [z < -1.96 for z in z_series]

        hot_ratio = sum(is_hot) / T
        cold_ratio = sum(is_cold) / T

        recent_hot = is_hot[-1]
        recent_cold = is_cold[-1]
        prior_hot = any(is_hot[:-1]) if T > 1 else False
        prior_cold = any(is_cold[:-1]) if T > 1 else False

        pattern = "No Pattern"
        if recent_hot:
            if not prior_hot and T > 1:
                pattern = "New Hotspot"
            elif all(is_hot[-max(2, T // 3):]):
                if mk.trend == "increasing":
                    pattern = "Intensifying Hotspot"
                elif hot_ratio >= 0.90:
                    pattern = "Persistent Hotspot"
                else:
                    pattern = "Consecutive Hotspot"
            elif any(is_cold):
                pattern = "Oscillating Hotspot"
            else:
                pattern = "Sporadic Hotspot"
        elif recent_cold:
            if not prior_cold and T > 1:
                pattern = "New Coldspot"
            elif all(is_cold[-max(2, T // 3):]):
                if mk.trend == "decreasing":
                    pattern = "Intensifying Coldspot"
                elif cold_ratio >= 0.90:
                    pattern = "Persistent Coldspot"
                else:
                    pattern = "Consecutive Coldspot"
            elif any(is_hot):
                pattern = "Oscillating Coldspot"
            else:
                pattern = "Sporadic Coldspot"
        else:
            if prior_hot and hot_ratio >= 0.50:
                pattern = "Diminishing Hotspot" if mk.trend == "decreasing" else "Historical Hotspot"
            elif prior_cold and cold_ratio >= 0.50:
                pattern = "Diminishing Coldspot" if mk.trend == "increasing" else "Historical Coldspot"
            else:
                pattern = "No Pattern"

        item = EmergingHotspotResult(
            pattern_type=pattern,
            gi_z_series=z_series,
            gi_p_series=p_series,
            mann_kendall=mk,
            recent_hotspot=recent_hot,
            historic_hotspot=prior_hot,
        )
        results.append(item)
        pattern_summary[pattern] = pattern_summary.get(pattern, 0) + 1

    st_moran = spatiotemporal_moran(cube, spatial_weights)

    return SpaceTimeAnalysisResult(
        n_locations=N,
        n_time_steps=T,
        results=results,
        pattern_summary=pattern_summary,
        global_spacetime_moran_i=st_moran,
    )
