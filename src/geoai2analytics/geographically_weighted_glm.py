# -*- coding: utf-8 -*-
"""Geographically Weighted Generalized Linear Models (GW-GLM Poisson & Logistic) for geoai2analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class GWGLMResult:
    """Fitted Geographically Weighted Poisson or Logistic Regression output."""

    model_family: str  # 'poisson' or 'logistic'
    bandwidth_distance: float
    n_observations: int
    local_coefficients: list[list[float]]  # [n_obs, n_features]
    predicted_means: list[float]
    residual_deviance: float
    aic_score: float
    local_r2_or_pseudo_r2: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "bandwidth_distance": round(self.bandwidth_distance, 2),
            "n_obs": self.n_observations,
            "residual_deviance": round(self.residual_deviance, 3),
            "aic_score": round(self.aic_score, 2),
            "mean_pseudo_r2": round(float(np.mean(self.local_r2_or_pseudo_r2)), 3) if self.local_r2_or_pseudo_r2 else 0.0,
        }


def fit_gw_poisson_regression(
    coords: Sequence[tuple[float, float]],
    x_features: Sequence[Sequence[float]],
    y_counts: Sequence[int | float],
    bandwidth_distance: float | None = None,
    max_iter: int = 25,
) -> GWGLMResult:
    r"""Fit local spatially varying Poisson GLM: \log(\mu_i) = \beta_0(u_i,v_i) + \sum \beta_k(u_i,v_i) x_{ik}."""
    pts = np.asarray(coords, dtype=np.float64)
    x_mat = np.asarray(x_features, dtype=np.float64)
    y_arr = np.asarray(y_counts, dtype=np.float64)
    n, k_feat = x_mat.shape

    # Add intercept column
    x_des = np.column_stack([np.ones(n), x_mat])
    p = x_des.shape[1]

    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))

    if bandwidth_distance is None:
        bandwidth_distance = float(np.median(dists) * 0.75)

    bw = max(1e-3, bandwidth_distance)
    local_betas = np.zeros((n, p), dtype=np.float64)
    y_preds = np.zeros(n, dtype=np.float64)
    pseudo_r2s = np.zeros(n, dtype=np.float64)

    for i in range(n):
        # Gaussian spatial kernel weights
        w_i = np.exp(-0.5 * (dists[i] / bw) ** 2)

        # Iteratively Reweighted Least Squares (IRLS) for Poisson
        beta = np.zeros(p, dtype=np.float64)
        beta[0] = np.log(max(1e-4, np.mean(y_arr)))

        for _ in range(max_iter):
            eta = np.dot(x_des, beta)
            eta = np.clip(eta, -15.0, 15.0)
            mu = np.exp(eta)

            # Working response z = eta + (y - mu) / mu
            with np.errstate(divide="ignore", invalid="ignore"):
                v_inv = np.where(mu > 1e-6, 1.0 / mu, 1e6)
                z = eta + (y_arr - mu) * v_inv

            # IRLS weight W = w_i * mu
            w_irls = w_i * mu
            w_diag = np.diag(w_irls)

            # (X' W X) beta = X' W z
            xt_w = np.dot(x_des.T, w_diag)
            xt_w_x = np.dot(xt_w, x_des) + np.eye(p) * 1e-4  # ridge stabilizer
            xt_w_z = np.dot(xt_w, z)

            try:
                beta_next = np.linalg.solve(xt_w_x, xt_w_z)
                if np.max(np.abs(beta_next - beta)) < 1e-4:
                    beta = beta_next
                    break
                beta = beta_next
            except np.linalg.LinAlgError:
                break

        local_betas[i] = beta
        pred_i = float(np.exp(np.clip(np.dot(x_des[i], beta), -15.0, 15.0)))
        y_preds[i] = pred_i

        # Local McFadden Pseudo R2
        null_mu = max(1e-4, np.sum(w_i * y_arr) / np.sum(w_i))
        dev_null = 2.0 * np.sum(w_i * (np.where(y_arr > 0, y_arr * np.log(np.maximum(1e-4, y_arr) / null_mu), 0) - (y_arr - null_mu)))
        dev_res = 2.0 * np.sum(w_i * (np.where(y_arr > 0, y_arr * np.log(np.maximum(1e-4, y_arr) / np.maximum(1e-4, mu)), 0) - (y_arr - mu)))
        pseudo_r2s[i] = max(0.0, min(1.0, 1.0 - (dev_res / max(1e-4, dev_null))))

    # Global deviance & AIC
    y_safe = np.maximum(1e-4, y_arr)
    pred_safe = np.maximum(1e-4, y_preds)
    res_dev = 2.0 * np.sum(np.where(y_arr > 0, y_arr * np.log(y_safe / pred_safe), 0) - (y_arr - y_preds))
    aic = res_dev + 2.0 * p * n / max(1.0, (n - p - 1.0))

    return GWGLMResult(
        model_family="poisson",
        bandwidth_distance=bw,
        n_observations=n,
        local_coefficients=local_betas.tolist(),
        predicted_means=y_preds.tolist(),
        residual_deviance=float(res_dev),
        aic_score=float(aic),
        local_r2_or_pseudo_r2=pseudo_r2s.tolist(),
    )
