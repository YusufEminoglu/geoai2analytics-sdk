# -*- coding: utf-8 -*-
"""
Spatial Autocorrelation, Local Indicators of Spatial Association (LISA),
Hotspot (Gi*), General G, Geary's C, Lee's L, Join Count,
Colocation Quotient (CLQ), Geodetector Q, Incremental Autocorrelation, and Spatial Gini.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .weights import SpatialWeights, distance_band_weights


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
class GeneralGResult:
    """Getis-Ord General G High/Low Clustering statistic."""

    G: float
    expected_G: float
    variance_G: float
    z_score: float
    p_value: float
    clustering_type: str  # High-Value Clustering, Low-Value Clustering, Random


@dataclass
class GearyCResult:
    """Geary's C spatial dissimilarity index."""

    C: float
    expected_C: float
    variance_C: float
    z_score: float
    p_value: float


@dataclass
class LocalGearyResult:
    """Local Geary's C local dissimilarity index."""

    c_i: np.ndarray
    p_values: np.ndarray
    cluster_labels: list[str]


@dataclass
class LeeLResult:
    """Lee's L spatial association integrating spatial smoothing and Pearson correlation."""

    L: float
    expected_L: float
    z_score: float
    p_value: float


@dataclass
class JoinCountResult:
    """Join Count statistics for binary categorical spatial autocorrelation."""

    bb_count: float  # Black-Black
    ww_count: float  # White-White
    bw_count: float  # Black-White
    bb_z_score: float
    ww_z_score: float
    p_value_bb: float


@dataclass
class ColocationQuotientResult:
    """Global and Local Colocation Quotient (CLQ) measuring categorical spatial co-occurrence."""

    global_clq: float
    local_clq: np.ndarray
    p_values: np.ndarray
    is_colocated: bool


@dataclass
class GeodetectorQResult:
    """Wang's GeoDetector Q-statistic assessing spatial stratified heterogeneity."""

    q_statistic: float  # 0.0 to 1.0 (100% of spatial variance explained by stratification)
    f_statistic: float
    p_value: float
    ssw: float  # Within Sum of Squares
    sst: float  # Total Sum of Squares


@dataclass
class IncrementalAutocorrResult:
    """Peak distance band detection for maximum spatial clustering."""

    distances: np.ndarray
    moran_i_values: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    peak_distance: float
    max_z_score: float


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
# 1. Global Moran's I
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
    if s0 == 0:
        return GlobalMoranResult(0.0, -1.0 / (n - 1), 0.0, 0.0, 1.0, 1.0, permutations)

    I_val = float((n / s0) * (np.dot(z, w_z) / s2))
    expected_I = -1.0 / (n - 1)

    s1 = weights.s1
    w_s2 = weights.s2
    kurt = float(n * np.sum(z**4) / (s2**2))

    a = n * ((n**2 - 3 * n + 3) * s1 - n * w_s2 + 3 * (s0**2))
    b = kurt * ((n**2 - n) * s1 - 2 * n * w_s2 + 6 * (s0**2))
    c = (n - 1) * (n - 2) * (n - 3) * (s0**2)
    var_I = float((a - b) / c - (expected_I**2)) if c > 0 else 0.001
    var_I = max(var_I, 1e-9)

    z_score = float((I_val - expected_I) / math.sqrt(var_I))
    p_val_analytic = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))

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
# 2. Local Moran's I (LISA)
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
    I_i = (z / s2) * w_z

    rng = np.random.default_rng(seed)
    sim_counts = np.zeros(n)

    for _p in range(permutations):
        z_perm = rng.permutation(z)
        w_z_perm = weights.lag(z_perm)
        sim_I_i = (z_perm / s2) * w_z_perm
        sim_counts += np.where(np.abs(sim_I_i) >= np.abs(I_i), 1, 0)

    p_values = (sim_counts + 1.0) / (permutations + 1.0)

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
# 3. Getis-Ord Local Gi* (Hotspot Analysis)
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
    w_star = w_mat.copy()
    np.fill_diagonal(w_star, 1.0)
    w_sum = np.sum(w_star, axis=1)
    w_sq_sum = np.sum(w_star**2, axis=1)

    numer = np.dot(w_star, x) - (x_bar * w_sum)
    denom = s * np.sqrt(((n * w_sq_sum) - (w_sum**2)) / (n - 1))
    denom = np.where(denom == 0, 1e-9, denom)

    gi_star = numer / denom
    p_values = np.zeros(n)
    cluster_bins = np.zeros(n, dtype=np.int32)

    for i in range(n):
        z = gi_star[i]
        p_values[i] = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
        if z >= 2.576 and p_values[i] <= 0.01:
            cluster_bins[i] = 3
        elif z >= 1.96 and p_values[i] <= 0.05:
            cluster_bins[i] = 2
        elif z >= 1.645 and p_values[i] <= 0.10:
            cluster_bins[i] = 1
        elif z <= -2.576 and p_values[i] <= 0.01:
            cluster_bins[i] = -3
        elif z <= -1.96 and p_values[i] <= 0.05:
            cluster_bins[i] = -2
        elif z <= -1.645 and p_values[i] <= 0.10:
            cluster_bins[i] = -1
        else:
            cluster_bins[i] = 0

    return LocalGetisOrdResult(gi_star=gi_star, p_values=p_values, cluster_bins=cluster_bins)


# ---------------------------------------------------------------------------
# 4. Getis-Ord General G (High/Low Clustering)
# ---------------------------------------------------------------------------
def general_g(
    y: np.ndarray | list[float],
    weights: SpatialWeights,
) -> GeneralGResult:
    """Calculate Getis-Ord General G statistic measuring concentration of high vs low values."""
    x = np.asarray(y, dtype=np.float64)
    n = len(x)
    w_mat = weights.matrix

    # Remove diagonals for General G
    w_no_diag = w_mat.copy()
    np.fill_diagonal(w_no_diag, 0.0)

    xx_mat = x[:, np.newaxis] * x[np.newaxis, :]
    np.fill_diagonal(xx_mat, 0.0)

    sum_wxx = float(np.sum(w_no_diag * xx_mat))
    sum_xx = float(np.sum(xx_mat))
    if sum_xx == 0:
        return GeneralGResult(0.0, 0.0, 0.0, 0.0, 1.0, "Random")

    G_val = sum_wxx / sum_xx
    s0 = float(np.sum(w_no_diag))
    exp_G = s0 / (n * (n - 1))

    # General G variance
    b0 = (n**2 - 3 * n + 3) * s0 - n * float(np.sum(np.sum(w_no_diag, axis=1) ** 2)) + 3 * (s0**2)
    var_G = max(1e-9, (b0 / (n * (n - 1) * (n - 2) * (n - 3))) - (exp_G**2))

    z_score = float((G_val - exp_G) / math.sqrt(var_G))
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))

    clustering = (
        "High-Value Clustering"
        if z_score > 1.96
        else ("Low-Value Clustering" if z_score < -1.96 else "Random")
    )

    return GeneralGResult(
        G=G_val,
        expected_G=exp_G,
        variance_G=var_G,
        z_score=z_score,
        p_value=p_value,
        clustering_type=clustering,
    )


# ---------------------------------------------------------------------------
# 5. Geary's C & Local Geary
# ---------------------------------------------------------------------------
def geary_c(
    y: np.ndarray | list[float],
    weights: SpatialWeights,
    permutations: int = 999,
    seed: int = 42,
) -> GearyCResult:
    """Compute Geary's C spatial dissimilarity index (C < 1: positive autocorr, C > 1: negative)."""
    x = np.asarray(y, dtype=np.float64)
    n = len(x)
    z = x - np.mean(x)
    s2 = np.sum(z**2) / (n - 1)

    diff_sq = (x[:, np.newaxis] - x[np.newaxis, :]) ** 2
    w_mat = weights.matrix
    s0 = weights.s0

    C_val = float(((n - 1) / (2.0 * s0 * (n - 1) * s2)) * np.sum(w_mat * diff_sq))
    exp_C = 1.0

    # Variance under randomization
    var_C = 2.0 / (s0 * (n - 1))
    z_score = float((C_val - exp_C) / math.sqrt(max(1e-9, var_C)))
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))

    return GearyCResult(
        C=C_val, expected_C=exp_C, variance_C=var_C, z_score=z_score, p_value=p_value
    )


def local_geary(
    y: np.ndarray | list[float],
    weights: SpatialWeights,
    permutations: int = 999,
    seed: int = 42,
) -> LocalGearyResult:
    """Compute Local Geary's c_i measuring local attribute difference with neighbors."""
    x = np.asarray(y, dtype=np.float64)
    n = len(x)
    s2 = np.var(x)
    if s2 == 0:
        s2 = 1e-9

    w_mat = weights.matrix
    c_i = np.zeros(n)
    for i in range(n):
        diffs = (x[i] - x) ** 2
        c_i[i] = np.sum(w_mat[i] * diffs) / s2

    rng = np.random.default_rng(seed)
    sim_counts = np.zeros(n)
    for _ in range(permutations):
        x_perm = rng.permutation(x)
        for i in range(n):
            sim_ci = np.sum(w_mat[i] * ((x_perm[i] - x_perm) ** 2)) / s2
            if abs(sim_ci - 1.0) >= abs(c_i[i] - 1.0):
                sim_counts[i] += 1

    p_vals = (sim_counts + 1.0) / (permutations + 1.0)
    labels = [
        "Local Cluster"
        if (c_i[i] < 1.0 and p_vals[i] <= 0.05)
        else ("Spatial Outlier" if (c_i[i] > 1.0 and p_vals[i] <= 0.05) else "Not Significant")
        for i in range(n)
    ]

    return LocalGearyResult(c_i=c_i, p_values=p_vals, cluster_labels=labels)


# ---------------------------------------------------------------------------
# 6. Global & Bivariate Lee's L
# ---------------------------------------------------------------------------
def global_lee_l(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    weights: SpatialWeights,
) -> LeeLResult:
    """Compute Lee's L spatial association integrating spatial smoothing and Pearson correlation."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n = len(x_arr)

    wx = weights.lag(x_arr)
    wy = weights.lag(y_arr)

    # Standardized spatial lags
    zx = (wx - np.mean(wx)) / max(1e-9, np.std(wx))
    zy = (wy - np.mean(wy)) / max(1e-9, np.std(wy))

    L_val = float(np.dot(zx, zy) / (n - 1))
    exp_L = 0.0
    se_L = 1.0 / math.sqrt(n)
    z_score = float(L_val / se_L)
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))

    return LeeLResult(L=L_val, expected_L=exp_L, z_score=z_score, p_value=p_value)


# ---------------------------------------------------------------------------
# 7. Join Count (Categorical Spatial Autocorrelation)
# ---------------------------------------------------------------------------
def join_count(
    y: np.ndarray | list[int | str],
    weights: SpatialWeights,
) -> JoinCountResult:
    """Calculate Join Count statistics for binary (0/1 or Black/White) categorical spatial data."""
    arr = np.asarray(y)
    unique_vals = np.unique(arr)
    if len(unique_vals) != 2:
        raise ValueError(f"Join count requires exactly 2 binary classes, found {len(unique_vals)}")

    b_val = unique_vals[0]
    b = (arr == b_val).astype(np.float64)
    w = 1.0 - b
    n = len(b)
    w_mat = weights.matrix

    # Half symmetric sum
    bb = float(0.5 * np.sum(w_mat * (b[:, np.newaxis] * b[np.newaxis, :])))
    ww = float(0.5 * np.sum(w_mat * (w[:, np.newaxis] * w[np.newaxis, :])))
    bw = float(
        0.5
        * np.sum(
            w_mat * (b[:, np.newaxis] * w[np.newaxis, :] + w[:, np.newaxis] * b[np.newaxis, :])
        )
    )

    # Analytical expectation for BB
    p_b = np.sum(b) / n
    s0 = weights.s0
    exp_bb = 0.5 * s0 * (p_b**2)
    var_bb = max(1e-9, 0.5 * s0 * (p_b**2) * (1.0 - p_b**2))

    z_bb = float((bb - exp_bb) / math.sqrt(var_bb))
    p_bb = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_bb) / math.sqrt(2.0))))

    p_w = 1.0 - p_b
    exp_ww = 0.5 * s0 * (p_w**2)
    var_ww = max(1e-9, 0.5 * s0 * (p_w**2) * (1.0 - p_w**2))
    z_ww = float((ww - exp_ww) / math.sqrt(var_ww))

    return JoinCountResult(
        bb_count=bb,
        ww_count=ww,
        bw_count=bw,
        bb_z_score=z_bb,
        ww_z_score=z_ww,
        p_value_bb=p_bb,
    )


# ---------------------------------------------------------------------------
# 8. Colocation Quotient (CLQ)
# ---------------------------------------------------------------------------
def colocation_quotient(
    cat_A: np.ndarray | list[int | str],
    cat_B: np.ndarray | list[int | str],
    weights: SpatialWeights,
) -> ColocationQuotientResult:
    """Calculate Global and Local Colocation Quotient (Leslie & Kronenfeld)."""
    A = np.asarray(cat_A)
    B = np.asarray(cat_B)
    n = len(A)

    w_mat = weights.matrix
    local_clq = np.zeros(n)
    p_B = float(np.mean(B == 1 if np.issubdtype(B.dtype, np.number) else (B == np.unique(B)[0])))
    if p_B == 0:
        p_B = 1e-9

    for i in range(n):
        neighbors_b = np.sum(
            w_mat[i] * (B == 1 if np.issubdtype(B.dtype, np.number) else (B == np.unique(B)[0]))
        )
        sum_w = np.sum(w_mat[i])
        local_clq[i] = (neighbors_b / max(1e-9, sum_w)) / p_B

    global_clq = float(np.mean(local_clq))
    p_vals = np.where(local_clq > 1.5, 0.01, (np.where(local_clq > 1.2, 0.05, 0.50)))

    return ColocationQuotientResult(
        global_clq=global_clq,
        local_clq=local_clq,
        p_values=p_vals,
        is_colocated=global_clq > 1.0,
    )


# ---------------------------------------------------------------------------
# 9. Geodetector Q-Statistic
# ---------------------------------------------------------------------------
def geodetector_q(
    y: np.ndarray | list[float],
    strata: np.ndarray | list[int | str],
) -> GeodetectorQResult:
    """Compute Wang's GeoDetector Q-statistic assessing spatial stratified heterogeneity."""
    y_arr = np.asarray(y, dtype=np.float64)
    st = np.asarray(strata)
    n = len(y_arr)
    var_total = np.var(y_arr)
    sst = float(n * var_total)
    if sst == 0:
        return GeodetectorQResult(0.0, 0.0, 1.0, 0.0, 0.0)

    unique_strata = np.unique(st)
    l_strata = len(unique_strata)
    ssw = 0.0

    for s_val in unique_strata:
        mask = st == s_val
        n_h = np.sum(mask)
        var_h = np.var(y_arr[mask])
        ssw += float(n_h * var_h)

    q = 1.0 - (ssw / sst)
    q = max(0.0, min(1.0, q))

    # F-test for significance
    df1 = l_strata - 1
    df2 = n - l_strata
    if df1 > 0 and df2 > 0 and (1.0 - q) > 0:
        f_stat = float((q / df1) / ((1.0 - q) / df2))
        p_val = max(0.0001, 1.0 / (1.0 + f_stat))
    else:
        f_stat = 0.0
        p_val = 1.0

    return GeodetectorQResult(
        q_statistic=round(q, 4),
        f_statistic=round(f_stat, 2),
        p_value=round(p_val, 4),
        ssw=round(ssw, 2),
        sst=round(sst, 2),
    )


# ---------------------------------------------------------------------------
# 10. Incremental Spatial Autocorrelation
# ---------------------------------------------------------------------------
def incremental_autocorrelation(
    coords: np.ndarray | list[tuple[float, float]],
    y: np.ndarray | list[float],
    distance_steps: int = 10,
) -> IncrementalAutocorrResult:
    """Evaluate Moran's I across continuous distance intervals to identify peak clustering scale."""
    pts = np.asarray(coords, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)

    min_d = float(np.max(np.min(dist, axis=1))) * 1.05
    max_d = max(min_d * 2.0, float(np.max(dist)) * 0.5)
    distances = np.linspace(min_d, max_d, distance_steps)

    moran_vals = np.zeros(distance_steps)
    z_scores = np.zeros(distance_steps)
    p_values = np.zeros(distance_steps)

    for idx, d in enumerate(distances):
        w = distance_band_weights(pts, threshold=d, kernel="binary", standardize=True)
        res = global_moran(y, w, permutations=99)
        moran_vals[idx] = res.I
        z_scores[idx] = res.z_score
        p_values[idx] = res.p_value

    best_idx = int(np.argmax(z_scores))
    return IncrementalAutocorrResult(
        distances=distances,
        moran_i_values=moran_vals,
        z_scores=z_scores,
        p_values=p_values,
        peak_distance=float(distances[best_idx]),
        max_z_score=float(z_scores[best_idx]),
    )


# ---------------------------------------------------------------------------
# 11. Bivariate Moran's I
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
        raise ValueError(
            f"Input lengths ({len(x_arr)}, {len(y_arr)}) must match spatial weights ({weights.n})"
        )

    zx = (x_arr - np.mean(x_arr)) / max(1e-9, np.std(x_arr))
    zy = (y_arr - np.mean(y_arr)) / max(1e-9, np.std(y_arr))

    w_zy = weights.lag(zy)
    s0 = weights.s0

    I_xy = float((np.dot(zx, w_zy)) / s0)

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
# 12. Spatial Gini Index
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

    diff_matrix = np.abs(y_arr[:, np.newaxis] - y_arr[np.newaxis, :])
    gini = float(np.sum(diff_matrix) / (2.0 * (n**2) * mean_y))

    w_mat = weights.matrix
    spatial_diff = np.sum(diff_matrix * w_mat)
    spatial_gini_val = float(spatial_diff / (2.0 * np.sum(w_mat) * mean_y))

    ratio = float(spatial_gini_val / max(1e-9, gini))

    return SpatialGiniResult(
        gini=gini, spatial_gini=spatial_gini_val, spatial_disparity_ratio=ratio
    )
