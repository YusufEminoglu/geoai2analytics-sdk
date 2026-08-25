# -*- coding: utf-8 -*-
"""
Spatial Distribution, Central Tendency, Dispersion, and Directional Analytics:
Mean Center, Median Center, Central Feature, Standard Distance,
Standard Deviational Ellipse (SDE), Linear Directional Mean,
Average Nearest Neighbor (ANN), and Ripley's K-Function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MeanCenterResult:
    """Mean Center analytical coordinates."""

    x: float
    y: float
    weighted: bool
    total_weight: float


@dataclass
class MedianCenterResult:
    """Fermat-Weber continuous spatial median point."""

    x: float
    y: float
    iterations: int
    converged: bool


@dataclass
class CentralFeatureResult:
    """Most central feature index and coordinates minimizing total distance."""

    feature_index: int
    x: float
    y: float
    min_total_distance: float


@dataclass
class StandardDistanceResult:
    """Standard distance dispersion radius."""

    center_x: float
    center_y: float
    standard_distance: float
    circle_area: float


@dataclass
class SDEResult:
    """Standard Deviational Ellipse (SDE / Directional Distribution) output."""

    center_x: float
    center_y: float
    std_dev_x: float
    std_dev_y: float
    rotation_deg: float
    rotation_rad: float
    area: float
    eccentricity: float
    std_level: int  # 1, 2, or 3 (68%, 95%, 99%)

    def polygon_coords(self, num_points: int = 72) -> list[tuple[float, float]]:
        """Generate closed boundary polygon coordinates representing the ellipse."""
        theta = np.linspace(0, 2 * np.pi, num_points)
        cos_rot = math.cos(self.rotation_rad)
        sin_rot = math.sin(self.rotation_rad)

        ellipse_x = self.std_dev_x * np.cos(theta)
        ellipse_y = self.std_dev_y * np.sin(theta)

        # Rotate and translate
        x_rot = self.center_x + (ellipse_x * cos_rot - ellipse_y * sin_rot)
        y_rot = self.center_y + (ellipse_x * sin_rot + ellipse_y * cos_rot)

        return list(zip(x_rot.tolist(), y_rot.tolist()))


@dataclass
class LinearDirectionalMeanResult:
    """Directional trend and circular variance of line segments."""

    mean_direction_deg: float
    circular_variance: float
    resultant_length: float
    directional_strength: str  # Strong, Moderate, Weak


@dataclass
class ANNResult:
    """Average Nearest Neighbor (ANN) spatial pattern test."""

    observed_mean_dist: float
    expected_mean_dist: float
    nearest_neighbor_ratio: float  # NNI = DO / DE
    z_score: float
    p_value: float
    spatial_pattern: str  # Clustered, Dispersed, Random


@dataclass
class RipleysKResult:
    """Ripley's K-Function & Besag's L(d) multi-distance spatial clustering."""

    radii: np.ndarray
    k_values: np.ndarray
    l_values: np.ndarray
    csr_envelope_low: np.ndarray
    csr_envelope_high: np.ndarray


# ---------------------------------------------------------------------------
# 1. Mean Center
# ---------------------------------------------------------------------------
def mean_center(
    coords: np.ndarray | list[tuple[float, float]],
    weights: np.ndarray | list[float] | None = None,
) -> MeanCenterResult:
    """Calculate the geographic / weighted mean center (center of mass)."""
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Coords shape {pts.shape} must be (N, 2)")

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(pts):
            raise ValueError("Weights length must match coordinates count")
        total_w = float(np.sum(w))
        if total_w <= 0:
            raise ValueError("Sum of weights must be positive")
        mean_x = float(np.sum(pts[:, 0] * w) / total_w)
        mean_y = float(np.sum(pts[:, 1] * w) / total_w)
        return MeanCenterResult(x=mean_x, y=mean_y, weighted=True, total_weight=total_w)
    else:
        mean_x = float(np.mean(pts[:, 0]))
        mean_y = float(np.mean(pts[:, 1]))
        return MeanCenterResult(x=mean_x, y=mean_y, weighted=False, total_weight=float(len(pts)))


# ---------------------------------------------------------------------------
# 2. Median Center (Fermat-Weber Point via Weiszfeld's Algorithm)
# ---------------------------------------------------------------------------
def median_center(
    coords: np.ndarray | list[tuple[float, float]],
    weights: np.ndarray | list[float] | None = None,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> MedianCenterResult:
    """Find the Fermat-Weber continuous spatial median point minimizing L1 Euclidean distances."""
    pts = np.asarray(coords, dtype=np.float64)
    n = len(pts)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)

    # Initial guess is the mean center
    cur_x = float(np.sum(pts[:, 0] * w) / np.sum(w))
    cur_y = float(np.sum(pts[:, 1] * w) / np.sum(w))

    converged = False
    iterations = 0

    for it in range(max_iter):
        iterations = it + 1
        diff = pts - np.array([cur_x, cur_y])
        dists = np.sqrt(np.sum(diff**2, axis=1))

        # Avoid divide-by-zero
        safe_dists = np.where(dists < 1e-12, 1e-12, dists)
        inv_d = w / safe_dists
        sum_inv = np.sum(inv_d)

        new_x = float(np.sum(pts[:, 0] * inv_d) / sum_inv)
        new_y = float(np.sum(pts[:, 1] * inv_d) / sum_inv)

        shift = math.hypot(new_x - cur_x, new_y - cur_y)
        cur_x, cur_y = new_x, new_y
        if shift < tol:
            converged = True
            break

    return MedianCenterResult(x=cur_x, y=cur_y, iterations=iterations, converged=converged)


# ---------------------------------------------------------------------------
# 3. Central Feature
# ---------------------------------------------------------------------------
def central_feature(
    coords: np.ndarray | list[tuple[float, float]],
    weights: np.ndarray | list[float] | None = None,
) -> CentralFeatureResult:
    """Identify the feature in the dataset that has the shortest total distance to all other features."""
    pts = np.asarray(coords, dtype=np.float64)
    n = len(pts)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))  # (N, N)

    # Weighted distance sum per feature
    total_dists = np.sum(dists * w[np.newaxis, :], axis=1)
    best_idx = int(np.argmin(total_dists))

    return CentralFeatureResult(
        feature_index=best_idx,
        x=float(pts[best_idx, 0]),
        y=float(pts[best_idx, 1]),
        min_total_distance=float(total_dists[best_idx]),
    )


# ---------------------------------------------------------------------------
# 4. Standard Distance
# ---------------------------------------------------------------------------
def standard_distance(
    coords: np.ndarray | list[tuple[float, float]],
    weights: np.ndarray | list[float] | None = None,
) -> StandardDistanceResult:
    """Measure the degree of spatial dispersion/concentration around the mean center."""
    pts = np.asarray(coords, dtype=np.float64)
    n = len(pts)
    mc = mean_center(pts, weights)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)

    dev_x = pts[:, 0] - mc.x
    dev_y = pts[:, 1] - mc.y

    sum_w = np.sum(w)
    sd = float(math.sqrt((np.sum(w * (dev_x**2)) / sum_w) + (np.sum(w * (dev_y**2)) / sum_w)))
    area = float(math.pi * (sd**2))

    return StandardDistanceResult(
        center_x=mc.x,
        center_y=mc.y,
        standard_distance=sd,
        circle_area=area,
    )


# ---------------------------------------------------------------------------
# 5. Standard Deviational Ellipse (SDE)
# ---------------------------------------------------------------------------
def standard_deviational_ellipse(
    coords: np.ndarray | list[tuple[float, float]],
    weights: np.ndarray | list[float] | None = None,
    std_level: int = 1,
) -> SDEResult:
    """Calculate the Standard Deviational Ellipse capturing directional distribution and anisotropy."""
    pts = np.asarray(coords, dtype=np.float64)
    n = len(pts)
    mc = mean_center(pts, weights)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    sum_w = float(np.sum(w))

    x_dev = pts[:, 0] - mc.x
    y_dev = pts[:, 1] - mc.y

    sum_x2 = float(np.sum(w * (x_dev**2)))
    sum_y2 = float(np.sum(w * (y_dev**2)))
    sum_xy = float(np.sum(w * x_dev * y_dev))

    # Analytical orientation angle calculation
    A = sum_x2 - sum_y2
    B = math.sqrt(A**2 + 4 * (sum_xy**2))
    C = 2 * sum_xy

    if C != 0:
        theta_rad = math.atan((A + B) / C)
    else:
        theta_rad = 0.0 if sum_x2 >= sum_y2 else math.pi / 2.0

    sin_t = math.sin(theta_rad)
    cos_t = math.cos(theta_rad)

    # Standard deviation along major and minor axes
    term_x = (x_dev * cos_t) - (y_dev * sin_t)
    term_y = (x_dev * sin_t) + (y_dev * cos_t)

    scale_mult = {1: 1.0, 2: 2.0, 3: 3.0}.get(std_level, 1.0)
    std_x = float(scale_mult * math.sqrt(max(0.0, np.sum(w * (term_x**2)) / sum_w)))
    std_y = float(scale_mult * math.sqrt(max(0.0, np.sum(w * (term_y**2)) / sum_w)))

    # Ensure major axis is std_x
    if std_y > std_x:
        std_x, std_y = std_y, std_x
        theta_rad += math.pi / 2.0

    theta_deg = float(math.degrees(theta_rad) % 180.0)
    area = float(math.pi * std_x * std_y)
    eccentricity = float(math.sqrt(max(0.0, 1.0 - (std_y**2) / max(1e-9, std_x**2))))

    return SDEResult(
        center_x=mc.x,
        center_y=mc.y,
        std_dev_x=std_x,
        std_dev_y=std_y,
        rotation_deg=theta_deg,
        rotation_rad=theta_rad,
        area=area,
        eccentricity=eccentricity,
        std_level=std_level,
    )


# ---------------------------------------------------------------------------
# 6. Linear Directional Mean
# ---------------------------------------------------------------------------
def linear_directional_mean(
    lines: list[tuple[tuple[float, float], tuple[float, float]]],
) -> LinearDirectionalMeanResult:
    """Calculate mean directional orientation and circular variance of line features."""
    if not lines:
        return LinearDirectionalMeanResult(0.0, 1.0, 0.0, "None")

    angles: list[float] = []
    lengths: list[float] = []

    for (x1, y1), (x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length > 1e-9:
            angle = math.atan2(dy, dx)
            # Normalize to directional orientation [0, pi]
            if angle < 0:
                angle += math.pi
            angles.append(2.0 * angle)  # Double angle method for unoriented axial data
            lengths.append(length)

    if not angles:
        return LinearDirectionalMeanResult(0.0, 1.0, 0.0, "None")

    sin_sum = sum(ln * math.sin(a) for a, ln in zip(angles, lengths))
    cos_sum = sum(ln * math.cos(a) for a, ln in zip(angles, lengths))
    total_len = sum(lengths)

    mean_double_angle = math.atan2(sin_sum, cos_sum)
    mean_angle = (mean_double_angle / 2.0) % math.pi
    mean_deg = math.degrees(mean_angle)

    R = math.hypot(sin_sum, cos_sum) / total_len
    circular_var = 1.0 - R

    strength = "Strong" if R >= 0.7 else ("Moderate" if R >= 0.4 else "Weak")

    return LinearDirectionalMeanResult(
        mean_direction_deg=round(mean_deg, 2),
        circular_variance=round(circular_var, 4),
        resultant_length=round(R, 4),
        directional_strength=strength,
    )


# ---------------------------------------------------------------------------
# 7. Average Nearest Neighbor (ANN)
# ---------------------------------------------------------------------------
def average_nearest_neighbor(
    coords: np.ndarray | list[tuple[float, float]],
    study_area: float | None = None,
) -> ANNResult:
    """Test spatial point clustering vs dispersion using Clark-Evans Average Nearest Neighbor (ANN)."""
    pts = np.asarray(coords, dtype=np.float64)
    n = len(pts)
    if n < 3:
        raise ValueError("ANN requires at least 3 spatial points")

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)

    min_dists = np.min(dists, axis=1)
    obs_mean = float(np.mean(min_dists))

    # Automatic minimum bounding box area if none provided
    if study_area is None:
        min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
        study_area = max(1e-5, (max_x - min_x) * (max_y - min_y))

    density = n / study_area
    exp_mean = float(0.5 / math.sqrt(density))
    nni = obs_mean / max(1e-9, exp_mean)

    # Standard error of expected distance
    se = float(0.26136 / math.sqrt(n * density))
    z_score = float((obs_mean - exp_mean) / max(1e-9, se))
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))

    if z_score < -1.96:
        pattern = "Clustered"
    elif z_score > 1.96:
        pattern = "Dispersed"
    else:
        pattern = "Random"

    return ANNResult(
        observed_mean_dist=obs_mean,
        expected_mean_dist=exp_mean,
        nearest_neighbor_ratio=nni,
        z_score=z_score,
        p_value=p_value,
        spatial_pattern=pattern,
    )


# ---------------------------------------------------------------------------
# 8. Ripley's K-Function & Besag's L(d)
# ---------------------------------------------------------------------------
def ripleys_k(
    coords: np.ndarray | list[tuple[float, float]],
    radii: np.ndarray | list[float] | None = None,
    study_area: float | None = None,
    n_sim: int = 99,
    seed: int = 42,
) -> RipleysKResult:
    """Compute empirical Ripley's K(r), Besag's L(r), and Monte Carlo CSR confidence envelope."""
    pts = np.asarray(coords, dtype=np.float64)
    n = len(pts)

    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
    if study_area is None:
        study_area = max(1e-5, (max_x - min_x) * (max_y - min_y))

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)

    if radii is None:
        max_r = float(0.25 * math.sqrt(study_area))
        radii_arr = np.linspace(max_r / 20.0, max_r, 20)
    else:
        radii_arr = np.asarray(radii, dtype=np.float64)

    # Compute empirical K(r)
    k_vals = np.zeros(len(radii_arr))
    for idx, r in enumerate(radii_arr):
        count = np.sum(dists <= r)
        k_vals[idx] = (study_area / (n * (n - 1))) * count

    # Besag's L(r) transformation
    l_vals = np.sqrt(k_vals / math.pi) - radii_arr

    # Monte Carlo Complete Spatial Randomness (CSR) Simulations
    rng = np.random.default_rng(seed)
    sim_L = np.zeros((n_sim, len(radii_arr)))

    for s in range(n_sim):
        sim_pts = np.column_stack(
            [rng.uniform(min_x, max_x, size=n), rng.uniform(min_y, max_y, size=n)]
        )
        sim_diff = sim_pts[:, np.newaxis, :] - sim_pts[np.newaxis, :, :]
        sim_d = np.sqrt(np.sum(sim_diff**2, axis=-1))
        np.fill_diagonal(sim_d, np.inf)

        for idx, r in enumerate(radii_arr):
            c_s = np.sum(sim_d <= r)
            k_s = (study_area / (n * (n - 1))) * c_s
            sim_L[s, idx] = math.sqrt(k_s / math.pi) - r

    env_low = np.percentile(sim_L, 2.5, axis=0)
    env_high = np.percentile(sim_L, 97.5, axis=0)

    return RipleysKResult(
        radii=radii_arr,
        k_values=k_vals,
        l_values=l_vals,
        csr_envelope_low=env_low,
        csr_envelope_high=env_high,
    )
