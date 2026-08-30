# -*- coding: utf-8 -*-
"""Spatial Extreme Value & Return Period Risk Modeling (GEV / GPD Peaks-Over-Threshold)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class GEVFitResult:
    """Generalized Extreme Value (GEV) distribution parameters."""

    location_mu: float  # Location mu
    scale_sigma: float  # Scale sigma (> 0)
    shape_xi: float  # Shape parameter xi (xi=0 Gumbel, xi>0 Frechet, xi<0 Weibull)
    return_level_10yr: float
    return_level_50yr: float
    return_level_100yr: float
    return_level_500yr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mu": round(self.location_mu, 3),
            "sigma": round(self.scale_sigma, 3),
            "xi": round(self.shape_xi, 3),
            "return_10yr": round(self.return_level_10yr, 2),
            "return_50yr": round(self.return_level_50yr, 2),
            "return_100yr": round(self.return_level_100yr, 2),
            "return_500yr": round(self.return_level_500yr, 2),
        }


def spatial_gev_fit(annual_maxima: Sequence[float] | np.ndarray) -> GEVFitResult:
    """Fit Generalized Extreme Value (GEV) distribution via Method of L-Moments."""
    arr = np.sort(np.asarray(annual_maxima, dtype=float))
    n = len(arr)
    if n < 3:
        raise ValueError("At least 3 annual maximum observations required to fit GEV distribution.")

    # Probability weighted moments b0, b1, b2
    b0 = float(np.mean(arr))
    weights1 = np.arange(n) / (n - 1)
    b1 = float(np.sum(weights1 * arr) / n)
    weights2 = (np.arange(n) * (np.arange(n) - 1)) / ((n - 1) * (n - 2))
    b2 = float(np.sum(weights2 * arr) / n)

    # L-moments
    l1 = b0
    l2 = 2.0 * b1 - b0
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    tau3 = l3 / max(1e-6, l2)

    # Approximation for shape parameter k = -xi (Hosking et al. 1985)
    c = (2.0 / (3.0 + tau3)) - (math.log(2.0) / math.log(3.0))
    k = 7.8590 * c + 2.9554 * (c ** 2)
    xi = -k  # standard sign convention

    # Scale sigma and location mu
    if abs(k) > 1e-4:
        gamma_1_plus_k = math.gamma(max(0.01, 1.0 + k))
        sigma = (l2 * k) / (gamma_1_plus_k * (1.0 - (2.0 ** (-k))))
        mu = l1 - (sigma / k) * (1.0 - gamma_1_plus_k)
    else:
        sigma = l2 / math.log(2.0)
        mu = l1 - 0.5772156649 * sigma
        xi = 0.0

    sigma = max(1e-4, sigma)

    def _calc_return_level(return_period_years: float) -> float:
        p = 1.0 - 1.0 / return_period_years
        y_p = -math.log(p)
        if abs(xi) > 1e-4:
            return float(mu + (sigma / xi) * ((y_p ** (-xi)) - 1.0))
        return float(mu - sigma * math.log(y_p))

    return GEVFitResult(
        location_mu=mu,
        scale_sigma=sigma,
        shape_xi=xi,
        return_level_10yr=_calc_return_level(10.0),
        return_level_50yr=_calc_return_level(50.0),
        return_level_100yr=_calc_return_level(100.0),
        return_level_500yr=_calc_return_level(500.0),
    )


def spatial_return_period_map(
    station_coords: Sequence[tuple[float, float]],
    station_series_matrix: Sequence[Sequence[float]] | np.ndarray,
    target_return_years: float = 100.0,
) -> list[dict[str, Any]]:
    """Compute spatial return levels for a specific return period (e.g. 100-year flood/heat) across all monitoring stations."""
    coords = list(station_coords)
    matrix = np.asarray(station_series_matrix, dtype=float)
    n_stations = len(coords)

    results: list[dict[str, Any]] = []
    for i in range(n_stations):
        series = matrix[i, :]
        fit = spatial_gev_fit(series)
        val = fit.return_level_100yr if target_return_years == 100.0 else (
            fit.return_level_50yr if target_return_years == 50.0 else fit.return_level_10yr
        )
        results.append({
            "station_idx": i,
            "x": coords[i][0],
            "y": coords[i][1],
            "return_period_years": target_return_years,
            "return_level": round(val, 3),
            "gev_mu": round(fit.location_mu, 3),
            "gev_sigma": round(fit.scale_sigma, 3),
        })

    return results
