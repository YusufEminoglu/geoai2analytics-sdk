# -*- coding: utf-8 -*-
"""
Spatial Autocorrelation, Local Indicators of Spatial Association (LISA), Hotspot, and Spatial Gini Engines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .weights import SpatialWeights


@dataclass
class GlobalMoranResult:
    """Global Moran's I spatial autocorrelation test output."""

    I: float  # noqa: E741
    expected_I: float
    variance_I: float
    z_score: float
    p_value: float
    p_sim: float
    permutations: int

    def summary(self) -> dict[str, Any]:
        return {
            "morans_i": round(self.I, 5),
            "expected_i": round(self.expected_I, 5),
            "variance": round(self.variance_I, 6),
            "z_score": round(self.z_score, 4),
            "p_value_analytic": round(self.p_value, 5),
            "p_value_monte_carlo": round(self.p_sim, 5),
            "spatial_pattern": "Clustered (Positive Autocorrelation)"
            if self.z_score > 1.96
            else ("Dispersed (Negative Autocorrelation)" if self.z_score < -1.96 else "Random"),
        }


@dataclass
class LocalMoranResult:
    """Local Moran's I (LISA) decomposition with cluster quadrant classifications."""

    I_i: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    quadrants: np.ndarray  # 1: High-High, 2: Low-Low, 3: Low-High, 4: High-Low, 0: Not Significant
    labels: list[str]

    @property
    def high_high_count(self) -> int:
        return int(np.sum(self.quadrants == 1))

    @property
    def low_low_count(self) -> int:
        return int(np.sum(self.quadrants == 2))

    @property
    def low_high_count(self) -> int:
        return int(np.sum(self.quadrants == 3))

    @property
    def high_low_count(self) -> int:
        return int(np.sum(self.quadrants == 4))

    @property
    def not_significant_count(self) -> int:
        return int(np.sum(self.quadrants == 0))

    def summary(self) -> dict[str, Any]:
        return {
            "high_high_hotspots": self.high_high_count,
            "low_low_coldspots": self.low_low_count,
            "low_high_spatial_outliers": self.low_high_count,
            "high_low_spatial_outliers": self.high_low_count,
            "not_significant": self.not_significant_count,
            "total_units": len(self.quadrants),
        }


@dataclass
class LocalGetisOrdResult:
    """Local Getis-Ord Gi* Hotspot Analysis."""

    gi_star: np.ndarray
    p_values: np.ndarray
    cluster_bins: (
        np.ndarray
    )  # +3 (99% Hot), +2 (95%), +1 (90%), 0 (NS), -1 (90% Cold), -2 (95%), -3 (99%)

    @property
    def hotspot_count_99(self) -> int:
        return int(np.sum(self.cluster_bins == 3))

    @property
    def hotspot_count_95(self) -> int:
        return int(np.sum(self.cluster_bins >= 2))

    @property
    def coldspot_count_95(self) -> int:
        return int(np.sum(self.cluster_bins <= -2))


@dataclass
class GearyCResult:
    """Geary's C spatial dissimilarity index."""

    C: float
    expected_C: float
    z_score: float
    p_value: float


@dataclass
class BivariateMoranResult:
    """Bivariate Moran's I cross-spatial correlation."""

    I_xy: float
    z_score: float
    p_value: float


@dataclass
class SpatialGiniResult:
    """Spatial Gini coefficient and spatial concentration inequality index."""

    gini: float
    spatial_gini: float
    spatial_disparity_ratio: float


# ---------------------------------------------------------------------------
# Global Moran's I
# ---------------------------------------------------------------------------
def global_moran(
    y: np.ndarray | list[float],
    weights: SpatialWeights,
    permutations: int = 999,
    seed: int = 42,
) -> GlobalMoranResult:
    """Compute Global Moran's I statistic with analytical & Monte Carlo p-values."""
    y_arr = np.asarray(y, dtype=np.float64)
    n = len(y_arr)
    if n != weights.n:
        raise ValueError(f"Length of y ({n}) does not match spatial weights n ({weights.n})")

    z = y_arr - np.mean(y_arr)
    s2 = np.sum(z**2)
    if s2 == 0:
        return GlobalMoranResult(0.0, -1.0 / (n - 1), 0.0, 0.0, 1.0, 1.0, permutations)

    w_z = weights.lag(z)
    s0 = weights.s0

    I_val = float((n / s0) * (np.dot(z, w_z) / s2))
    expected_I = -1.0 / (n - 1)

    # Analytical variance under randomization assumption
    s1 = weights.s1
    w_s2 = weights.s2
    kurt = float(n * np.sum(z**4) / (s2**2))

    a = n * ((n**2 - 3 * n + 3) * s1 - n * w_s2 + 3 * (s0**2))
    b = kurt * ((n**2 - n) * s1 - 2 * n * w_s2 + 6 * (s0**2))
    c = (n - 1) * (n - 2) * (n - 3) * (s0**2)
    var_I = float((a - b) / c - (expected_I**2)) if c > 0 else 0.001
    var_I = max(var_I, 1e-9)

    z_score = float((I_val - expected_I) / math.sqrt(var_I))
    # Two-sided standard normal p-value
    p_val_analytic = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))

    # Monte Carlo Permutations
    rng = np.random.default_rng(seed)
    sim_I = np.zeros(permutations)
    for p in range(permutations):
        z_perm = rng.permutation(z)
        w_z_perm = weights.lag(z_perm)
        sim_I[p] = (n / s0) * (np.dot(z_perm, w_z_perm) / s2)

    greater_count = np.sum(sim_I >= I_val) if I_val >= expected_I else np.sum(sim_I <= I_val)
    p_sim = float((greater_count + 1.0) / (permutations + 1.0))

    return GlobalMoranResult(
        I=I_val,
        expected_I=expected_I,
        variance_I=var_I,
        z_score=z_score,
        p_value=p_val_analytic,
        p_sim=p_sim,
        permutations=permutations,
    )


# ---------------------------------------------------------------------------
# Local Moran's I (LISA)
# ---------------------------------------------------------------------------
def local_moran(
    y: np.ndarray | list[float],
    weights: SpatialWeights,
    significance_cutoff: float = 0.05,
    permutations: int = 999,
    seed: int = 42,
) -> LocalMoranResult:
    """Compute Local Indicators of Spatial Association (LISA) / Local Moran's I."""
    y_arr = np.asarray(y, dtype=np.float64)
    n = len(y_arr)
    z = y_arr - np.mean(y_arr)
    s2 = np.sum(z**2) / (n - 1)
    if s2 == 0:
        s2 = 1e-9

    w_z = weights.lag(z)
    # Local Moran I_i = (z_i / s^2) * sum_j(w_ij * z_j)
    I_i = (z / s2) * w_z

    # Pseudo p-values via conditional randomization
    rng = np.random.default_rng(seed)
    sim_counts = np.zeros(n)

    for _p in range(permutations):
        z_perm = rng.permutation(z)
        w_z_perm = weights.lag(z_perm)
        sim_I_i = (z_perm / s2) * w_z_perm
        sim_counts += np.where(np.abs(sim_I_i) >= np.abs(I_i), 1, 0)

    p_values = (sim_counts + 1.0) / (permutations + 1.0)

    # Cluster quadrants:
    # 1: High-High (z > 0 & w_z > 0)
    # 2: Low-Low   (z < 0 & w_z < 0)
    # 3: Low-High  (z < 0 & w_z > 0)
    # 4: High-Low  (z > 0 & w_z < 0)
    quadrants = np.zeros(n, dtype=np.int32)
    labels: list[str] = []

    for i in range(n):
        if p_values[i] <= significance_cutoff:
            if z[i] > 0 and w_z[i] > 0:
                quadrants[i] = 1
                labels.append("High-High (Hotspot)")
            elif z[i] < 0 and w_z[i] < 0:
                quadrants[i] = 2
                labels.append("Low-Low (Coldspot)")
            elif z[i] < 0 and w_z[i] > 0:
                quadrants[i] = 3
                labels.append("Low-High (Spatial Outlier)")
            elif z[i] > 0 and w_z[i] < 0:
                quadrants[i] = 4
                labels.append("High-Low (Spatial Outlier)")
            else:
                quadrants[i] = 0
                labels.append("Not Significant")
        else:
            quadrants[i] = 0
            labels.append("Not Significant")

    z_scores = np.where(p_values > 0, (I_i - np.mean(I_i)) / max(1e-9, np.std(I_i)), 0.0)

    return LocalMoranResult(
        I_i=I_i,
        z_scores=z_scores,
        p_values=p_values,
        quadrants=quadrants,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Getis-Ord Local Gi* (Hotspot Analysis)
# ---------------------------------------------------------------------------
def local_getis_ord_gi_star(
    y: np.ndarray | list[float],
    weights: SpatialWeights,
) -> LocalGetisOrdResult:
    """Compute Getis-Ord Gi* statistic for local hotspot and coldspot detection."""
    x = np.asarray(y, dtype=np.float64)
    n = len(x)
    x_bar = np.mean(x)
    s = np.std(x)
    if s == 0:
        s = 1e-9

    w_mat = weights.matrix
    # Add self-weight (Gi* includes focal unit i)
    w_star = w_mat.copy()
    np.fill_diagonal(w_star, 1.0)
    w_sum = np.sum(w_star, axis=1)
    w_sq_sum = np.sum(w_star**2, axis=1)

    numer = np.dot(w_star, x) - (x_bar * w_sum)
    denom = s * np.sqrt(((n * w_sq_sum) - (w_sum**2)) / (n - 1))
    denom = np.where(denom == 0, 1e-9, denom)

    gi_star = numer / denom

    # Calculate two-tailed normal p-values
    p_values = np.zeros(n)
    cluster_bins = np.zeros(n, dtype=np.int32)

    for i in range(n):
        z = gi_star[i]
        p_values[i] = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
        if z >= 2.576 and p_values[i] <= 0.01:
            cluster_bins[i] = 3  # Hotspot 99%
        elif z >= 1.96 and p_values[i] <= 0.05:
            cluster_bins[i] = 2  # Hotspot 95%
        elif z >= 1.645 and p_values[i] <= 0.10:
            cluster_bins[i] = 1  # Hotspot 90%
        elif z <= -2.576 and p_values[i] <= 0.01:
            cluster_bins[i] = -3  # Coldspot 99%
        elif z <= -1.96 and p_values[i] <= 0.05:
            cluster_bins[i] = -2  # Coldspot 95%
        elif z <= -1.645 and p_values[i] <= 0.10:
            cluster_bins[i] = -1  # Coldspot 90%
        else:
            cluster_bins[i] = 0

    return LocalGetisOrdResult(gi_star=gi_star, p_values=p_values, cluster_bins=cluster_bins)


# ---------------------------------------------------------------------------
# Bivariate Moran's I
# ---------------------------------------------------------------------------
def bivariate_moran(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    weights: SpatialWeights,
    permutations: int = 999,
    seed: int = 42,
) -> BivariateMoranResult:
    """Compute Bivariate Moran's I cross-spatial correlation between X and spatial lag of Y."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(x_arr) != weights.n or len(y_arr) != weights.n:
        raise ValueError(f"Input lengths ({len(x_arr)}, {len(y_arr)}) must match spatial weights ({weights.n})")

    zx = (x_arr - np.mean(x_arr)) / max(1e-9, np.std(x_arr))
    zy = (y_arr - np.mean(y_arr)) / max(1e-9, np.std(y_arr))

    w_zy = weights.lag(zy)
    s0 = weights.s0

    I_xy = float((np.dot(zx, w_zy)) / s0)

    # Permutation test
    rng = np.random.default_rng(seed)
    sim_I = np.zeros(permutations)
    for p in range(permutations):
        zx_perm = rng.permutation(zx)
        sim_I[p] = (np.dot(zx_perm, w_zy)) / s0

    greater = np.sum(np.abs(sim_I) >= abs(I_xy))
    p_val = float((greater + 1.0) / (permutations + 1.0))
    z_score = float((I_xy - np.mean(sim_I)) / max(1e-9, np.std(sim_I)))

    return BivariateMoranResult(I_xy=I_xy, z_score=z_score, p_value=p_val)


# ---------------------------------------------------------------------------
# Spatial Gini Index
# ---------------------------------------------------------------------------
def spatial_gini(
    y: np.ndarray | list[float],
    weights: SpatialWeights,
) -> SpatialGiniResult:
    """Compute standard Gini inequality and Rey's Spatial Gini concentration index."""
    y_arr = np.asarray(y, dtype=np.float64)
    n = len(y_arr)
    mean_y = np.mean(y_arr)
    if mean_y == 0:
        return SpatialGiniResult(0.0, 0.0, 1.0)

    # Standard Gini
    diff_matrix = np.abs(y_arr[:, np.newaxis] - y_arr[np.newaxis, :])
    gini = float(np.sum(diff_matrix) / (2.0 * (n**2) * mean_y))

    # Spatial Gini (neighbors only)
    w_mat = weights.matrix
    spatial_diff = np.sum(diff_matrix * w_mat)
    spatial_gini_val = float(spatial_diff / (2.0 * np.sum(w_mat) * mean_y))

    ratio = float(spatial_gini_val / max(1e-9, gini))

    return SpatialGiniResult(
        gini=gini, spatial_gini=spatial_gini_val, spatial_disparity_ratio=ratio
    )
