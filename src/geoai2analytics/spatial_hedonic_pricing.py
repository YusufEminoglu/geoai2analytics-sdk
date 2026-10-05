# -*- coding: utf-8 -*-
"""Spatial Econometric Real Estate Hedonic Valuation Model with Amenity Decay for geoai2analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class HedonicValuationResult:
    r_squared: float
    adjusted_r2: float
    intercept: float
    coefficients: dict[str, float]
    predicted_prices: list[float]
    spatial_spillover_multiplier: float
    amenity_elasticities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "r_squared": round(self.r_squared, 3),
            "adjusted_r2": round(self.adjusted_r2, 3),
            "intercept": round(self.intercept, 2),
            "coefficients": {k: round(v, 4) for k, v in self.coefficients.items()},
            "spatial_spillover_multiplier": round(self.spatial_spillover_multiplier, 3),
        }


def fit_spatial_hedonic_model(
    coordinates: Sequence[tuple[float, float]],
    property_features: Sequence[dict[str, float]],  # {'area_m2': 120, 'age': 5, 'dist_cbd_km': 2.5}
    property_prices: Sequence[float],
    spatial_lag_rho: float = 0.35,
) -> HedonicValuationResult:
    """Fit a spatial hedonic real estate pricing model with structural and neighborhood amenity terms."""
    n = len(property_prices)
    if n == 0 or not property_features:
        return HedonicValuationResult(0.0, 0.0, 0.0, {}, [], 1.0, {})

    feature_names = list(property_features[0].keys())
    k = len(feature_names)

    # Construct design matrix
    x_mat = np.zeros((n, k), dtype=np.float64)
    for i, feat in enumerate(property_features):
        for j, fname in enumerate(feature_names):
            x_mat[i, j] = feat.get(fname, 0.0)

    y_vec = np.asarray(property_prices, dtype=np.float64)
    x_des = np.column_stack([np.ones(n), x_mat])

    # Spatial weights matrix (inverse distance)
    pts = np.asarray(coordinates, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)

    w = 1.0 / np.maximum(1e-2, dists)
    row_sums = np.sum(w, axis=1, keepdims=True)
    w_std = np.where(row_sums > 0, w / row_sums, 0.0)

    # Spatial autoregressive transform: (I - \rho W) y = X \beta + \epsilon
    i_mat = np.eye(n)
    a_mat = i_mat - spatial_lag_rho * w_std
    y_trans = np.dot(a_mat, y_vec)

    # OLS on transformed system
    try:
        xt_x = np.dot(x_des.T, x_des) + np.eye(k + 1) * 1e-4
        xt_y = np.dot(x_des.T, y_trans)
        betas = np.linalg.solve(xt_x, xt_y)
    except np.linalg.LinAlgError:
        betas = np.zeros(k + 1)

    y_pred = np.linalg.solve(a_mat, np.dot(x_des, betas))

    # R2
    ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
    ss_res = np.sum((y_vec - y_pred) ** 2)
    r2 = max(0.0, min(1.0, 1.0 - (ss_res / max(1e-4, ss_tot))))
    adj_r2 = max(0.0, 1.0 - (1.0 - r2) * (n - 1) / max(1, n - k - 1))

    coef_dict = {feature_names[i]: float(betas[i + 1]) for i in range(k)}

    # Spatial spillover multiplier: 1 / (1 - \rho)
    spillover = 1.0 / (1.0 - max(0.01, min(0.99, spatial_lag_rho)))

    elasticities = {}
    for j, fname in enumerate(feature_names):
        mean_x = float(np.mean(x_mat[:, j]))
        mean_y = float(np.mean(y_vec))
        if mean_y > 0:
            elasticities[fname] = float(betas[j + 1] * (mean_x / mean_y))

    return HedonicValuationResult(
        r_squared=float(r2),
        adjusted_r2=float(adj_r2),
        intercept=float(betas[0]),
        coefficients=coef_dict,
        predicted_prices=y_pred.tolist(),
        spatial_spillover_multiplier=float(spillover),
        amenity_elasticities=elasticities,
    )
