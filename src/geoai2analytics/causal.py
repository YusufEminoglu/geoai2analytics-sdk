# -*- coding: utf-8 -*-
"""Spatial Causal Inference & Propensity Score Matching (SPSM / Spatial DiD)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .weights import SpatialWeights


@dataclass
class SPSMMatchPair:
    """Matched treated and control unit pair."""

    treated_idx: int
    control_idx: int
    propensity_distance: float
    geographic_distance: float
    composite_distance: float


@dataclass
class SPSMResult:
    """Spatial Propensity Score Matching result."""

    att: float  # Average Treatment Effect on Treated
    att_se: float  # Standard error of ATT
    att_p_value: float  # p-value
    matched_pairs: list[SPSMMatchPair]
    covariate_balance_before: dict[str, float]  # Standardized mean differences before
    covariate_balance_after: dict[str, float]  # Standardized mean differences after
    n_treated: int
    n_control_matched: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "att": round(self.att, 4),
            "att_se": round(self.att_se, 4),
            "att_p_value": round(self.att_p_value, 4),
            "n_treated": self.n_treated,
            "n_control_matched": self.n_control_matched,
            "covariate_balance_before": {k: round(v, 4) for k, v in self.covariate_balance_before.items()},
            "covariate_balance_after": {k: round(v, 4) for k, v in self.covariate_balance_after.items()},
        }


@dataclass
class SDIDResult:
    """Spatial Difference-in-Differences estimator result."""

    att_direct: float
    att_spillover: float
    att_total: float
    rho_spatial_spillover: float
    se_direct: float
    se_spillover: float
    se_total: float
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "att_direct": round(self.att_direct, 4),
            "att_spillover": round(self.att_spillover, 4),
            "att_total": round(self.att_total, 4),
            "rho_spatial_spillover": round(self.rho_spatial_spillover, 4),
            "se_direct": round(self.se_direct, 4),
            "p_value": round(self.p_value, 4),
        }


def _fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    lr: float = 0.05,
) -> np.ndarray:
    """Fit simple logistic regression via gradient descent with intercept."""
    N, P = X.shape
    X_design = np.hstack([np.ones((N, 1)), X])
    beta = np.zeros(P + 1)

    for _ in range(max_iter):
        linear = np.dot(X_design, beta)
        linear = np.clip(linear, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-linear))
        grad = np.dot(X_design.T, (p - y)) / N
        beta -= lr * grad

    return beta


def spatial_propensity_score_matching(
    treatment: Sequence[int | bool] | np.ndarray,
    outcome: Sequence[float] | np.ndarray,
    covariates: Sequence[Sequence[float]] | np.ndarray,
    coords: Sequence[tuple[float, float]] | np.ndarray,
    spatial_penalty_weight: float = 0.25,
    caliper: float = 0.2,
    covariate_names: list[str] | None = None,
) -> SPSMResult:
    """Perform Spatial Propensity Score Matching (SPSM).

    Matches treated units to control units using a composite distance metric:
    D(i, j) = |e_i - e_j| + spatial_penalty_weight * (dist(i, j) / max_dist)
    where e_i is the estimated propensity score.
    """
    t = np.asarray(treatment, dtype=int)
    y = np.asarray(outcome, dtype=float)
    X = np.asarray(covariates, dtype=float)
    xy = np.asarray(coords, dtype=float)

    N = len(t)
    treated_indices = np.where(t == 1)[0]
    control_indices = np.where(t == 0)[0]

    if len(treated_indices) == 0 or len(control_indices) == 0:
        raise ValueError("Treatment and control groups must both have at least 1 observation.")

    # 1. Estimate propensity scores
    beta = _fit_logistic_regression(X, t)
    X_design = np.hstack([np.ones((N, 1)), X])
    linear = np.dot(X_design, beta)
    propensity_scores = 1.0 / (1.0 + np.exp(-np.clip(linear, -20.0, 20.0)))

    # Compute max spatial distance for normalization
    dists = np.hypot(xy[:, 0:1] - xy[:, 0:1].T, xy[:, 1:2] - xy[:, 1:2].T)
    max_geo_dist = float(np.max(dists)) if np.max(dists) > 0 else 1.0

    # 2. Compute covariate balance before matching (Standardized Mean Difference)
    names = covariate_names or [f"Covariate_{i+1}" for i in range(X.shape[1])]
    smd_before: dict[str, float] = {}
    for p_idx, name in enumerate(names):
        t_vals = X[treated_indices, p_idx]
        c_vals = X[control_indices, p_idx]
        pooled_sd = math.sqrt((np.var(t_vals) + np.var(c_vals)) / 2.0)
        smd_before[name] = float(abs(np.mean(t_vals) - np.mean(c_vals)) / max(1e-6, pooled_sd))

    # 3. Spatial matching (1:1 Nearest Neighbor with replacement or without)
    matched_pairs: list[SPSMMatchPair] = []
    matched_control_indices: list[int] = []
    y_diffs: list[float] = []

    for tr_idx in treated_indices:
        tr_p = propensity_scores[tr_idx]
        tr_xy = xy[tr_idx]

        best_ctrl = -1
        best_comp_dist = float("inf")
        best_p_dist = 0.0
        best_g_dist = 0.0

        for ctrl_idx in control_indices:
            ctrl_p = propensity_scores[ctrl_idx]
            ctrl_xy = xy[ctrl_idx]

            p_dist = abs(tr_p - ctrl_p)
            if p_dist > caliper:
                continue

            geo_dist = math.hypot(tr_xy[0] - ctrl_xy[0], tr_xy[1] - ctrl_xy[1])
            norm_geo = geo_dist / max_geo_dist

            comp_dist = p_dist + spatial_penalty_weight * norm_geo
            if comp_dist < best_comp_dist:
                best_comp_dist = comp_dist
                best_ctrl = ctrl_idx
                best_p_dist = p_dist
                best_g_dist = geo_dist

        if best_ctrl != -1:
            matched_pairs.append(
                SPSMMatchPair(
                    treated_idx=int(tr_idx),
                    control_idx=int(best_ctrl),
                    propensity_distance=float(best_p_dist),
                    geographic_distance=float(best_g_dist),
                    composite_distance=float(best_comp_dist),
                )
            )
            matched_control_indices.append(best_ctrl)
            y_diffs.append(float(y[tr_idx] - y[best_ctrl]))

    if not matched_pairs:
        # Fallback if caliper was too strict
        return SPSMResult(
            att=0.0,
            att_se=1.0,
            att_p_value=1.0,
            matched_pairs=[],
            covariate_balance_before=smd_before,
            covariate_balance_after=smd_before,
            n_treated=len(treated_indices),
            n_control_matched=0,
        )

    # 4. Compute ATT & Standard Error
    att = float(np.mean(y_diffs))
    att_se = float(np.std(y_diffs) / math.sqrt(len(y_diffs))) if len(y_diffs) > 1 else 1.0
    z_stat = att / max(1e-6, att_se)
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_stat) / math.sqrt(2.0))))

    # 5. Covariate balance after matching
    smd_after: dict[str, float] = {}
    m_treated_idxs = [p.treated_idx for p in matched_pairs]
    m_control_idxs = [p.control_idx for p in matched_pairs]

    for p_idx, name in enumerate(names):
        t_vals = X[m_treated_idxs, p_idx]
        c_vals = X[m_control_idxs, p_idx]
        pooled_sd = math.sqrt((np.var(t_vals) + np.var(c_vals)) / 2.0)
        smd_after[name] = float(abs(np.mean(t_vals) - np.mean(c_vals)) / max(1e-6, pooled_sd))

    return SPSMResult(
        att=att,
        att_se=att_se,
        att_p_value=float(p_val),
        matched_pairs=matched_pairs,
        covariate_balance_before=smd_before,
        covariate_balance_after=smd_after,
        n_treated=len(matched_pairs),
        n_control_matched=len(set(matched_control_indices)),
    )


def spatial_difference_in_differences(
    y_pre: Sequence[float] | np.ndarray,
    y_post: Sequence[float] | np.ndarray,
    treatment: Sequence[int | bool] | np.ndarray,
    spatial_weights: SpatialWeights,
) -> SDIDResult:
    r"""Spatial Difference-in-Differences (SDID) with spatial spillover autoregression.

    Model: \Delta y = \alpha + \tau \cdot T + \rho \cdot W \Delta y + \varepsilon
    """
    y0 = np.asarray(y_pre, dtype=float)
    y1 = np.asarray(y_post, dtype=float)
    t = np.asarray(treatment, dtype=float)
    W = spatial_weights.matrix

    delta_y = y1 - y0
    N = len(delta_y)

    # Spatial lag of outcome change: W \Delta y
    W_delta_y = np.dot(W, delta_y)

    # Regression: delta_y = beta0 + beta1 * t + beta2 * W_delta_y
    X = np.column_stack([np.ones(N), t, W_delta_y])
    try:
        beta = np.linalg.lstsq(X, delta_y, rcond=None)[0]
    except Exception:
        beta = np.zeros(3)

    _, direct_att, rho = beta[0], beta[1], beta[2]

    # Spillover effect: average spatial indirect effect
    spillover_att = float(rho * direct_att)
    total_att = float(direct_att + spillover_att)

    # Standard error approximation
    residuals = delta_y - np.dot(X, beta)
    sigma2 = np.sum(residuals**2) / max(1, N - 3)
    cov_mat = sigma2 * np.linalg.pinv(np.dot(X.T, X))
    se_direct = math.sqrt(max(1e-8, cov_mat[1, 1]))
    se_spillover = math.sqrt(max(1e-8, cov_mat[2, 2]))
    se_total = math.sqrt(se_direct**2 + se_spillover**2)

    z = direct_att / max(1e-6, se_direct)
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))

    return SDIDResult(
        att_direct=float(direct_att),
        att_spillover=spillover_att,
        att_total=total_att,
        rho_spatial_spillover=float(rho),
        se_direct=se_direct,
        se_spillover=se_spillover,
        se_total=se_total,
        p_value=float(p_val),
    )
