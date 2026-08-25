# -*- coding: utf-8 -*-
"""
Spatial Econometrics & Local Regression Suite:
GWR, MGWR, SAR (Spatial Lag), SEM (Spatial Error), SDM (Spatial Durbin),
Spatial Regime, ESF (Eigenvector Filtering), Quantile Regression,
Lagrange Multiplier (LM) Diagnostics, and Exploratory Regression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .weights import SpatialWeights


@dataclass
class GWRResult:
    """Geographically Weighted Regression (GWR) local parameter estimations."""

    params: np.ndarray  # (N, K)
    t_stats: np.ndarray  # (N, K)
    std_err: np.ndarray  # (N, K)
    local_r2: np.ndarray  # (N,)
    global_r2: float
    aicc: float
    aic: float
    bic: float
    bandwidth: float
    residuals: np.ndarray  # (N,)
    y_pred: np.ndarray  # (N,)
    covariate_names: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "bandwidth": round(self.bandwidth, 2),
            "global_r2": round(self.global_r2, 4),
            "aicc": round(self.aicc, 2),
            "aic": round(self.aic, 2),
            "bic": round(self.bic, 2),
            "mean_local_r2": round(float(np.mean(self.local_r2)), 4),
            "sample_size": len(self.residuals),
        }


@dataclass
class MGWRResult:
    """Multiscale Geographically Weighted Regression (MGWR) variable-specific results."""

    params: np.ndarray  # (N, K)
    bandwidths: dict[str, float]
    global_r2: float
    aicc: float
    residuals: np.ndarray


@dataclass
class SpatialLagResult:
    """Spatial Autoregressive (SAR / Spatial Lag) estimation result."""

    rho: float
    betas: np.ndarray
    residuals: np.ndarray
    r2: float
    aic: float
    log_likelihood: float
    covariate_names: list[str] = field(default_factory=list)


@dataclass
class SpatialErrorResult:
    """Spatial Error Model (SEM) estimation result."""

    lambda_param: float
    betas: np.ndarray
    residuals: np.ndarray
    r2: float
    aic: float
    log_likelihood: float
    covariate_names: list[str] = field(default_factory=list)


@dataclass
class SpatialDurbinResult:
    """Spatial Durbin Model (SDM) estimation with direct and indirect spatial spillover effects."""

    rho: float
    betas: np.ndarray
    gammas: np.ndarray  # Spatially lagged covariate effects W*X
    residuals: np.ndarray
    r2: float
    aic: float


@dataclass
class SpatialRegimeResult:
    """Spatial Regime structural instability model with Chow test."""

    regime_params: dict[Any, np.ndarray]
    chow_f_stat: float
    chow_p_value: float
    global_r2: float


@dataclass
class ESFResult:
    """Moran's Eigenvector Spatial Filtering (ESF) spatial proxy eigenvectors."""

    selected_eigenvector_indices: list[int]
    synthetic_spatial_proxies: np.ndarray  # (N, M)
    eigenvalues: np.ndarray
    r2_gain: float


@dataclass
class LMDiagnosticsResult:
    """Anselin's Lagrange Multiplier (LM) tests for spatial dependence specification."""

    lm_lag: float
    p_lm_lag: float
    lm_error: float
    p_lm_error: float
    robust_lm_lag: float
    p_robust_lm_lag: float
    robust_lm_error: float
    p_robust_lm_error: float
    suggested_model: (
        str  # SAR (Spatial Lag), SEM (Spatial Error), or OLS (No spatial autocorrelation)
    )


@dataclass
class ExploratoryRegressionCandidate:
    """A viable candidate model combination from exploratory search."""

    covariates: list[str]
    r2: float
    aicc: float
    max_vif: float
    residual_moran_p: float
    passes_all_checks: bool


# ---------------------------------------------------------------------------
# 1. Geographically Weighted Regression (GWR)
# ---------------------------------------------------------------------------
class GWR:
    """Geographically Weighted Regression with adaptive/fixed kernel optimization."""

    def __init__(
        self,
        coords: np.ndarray | list[tuple[float, float]],
        y: np.ndarray | list[float],
        X: np.ndarray | list[list[float]],
        kernel: str = "bisquare",
        adaptive: bool = True,
        bandwidth: float | None = None,
        covariate_names: list[str] | None = None,
    ) -> None:
        self.coords = np.asarray(coords, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.raw_X = np.asarray(X, dtype=np.float64)
        if self.raw_X.ndim == 1:
            self.raw_X = self.raw_X[:, np.newaxis]

        self.n = len(self.y)
        self.X = np.column_stack([np.ones(self.n), self.raw_X])
        self.k = self.X.shape[1]
        self.kernel = kernel.lower()
        self.adaptive = adaptive
        self.bandwidth = bandwidth

        self.covariate_names = ["Intercept"] + (
            covariate_names or [f"X{i}" for i in range(1, self.k)]
        )

        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        self.dist = np.sqrt(np.sum(diff**2, axis=-1))

    def _compute_weights(self, i: int, bw: float) -> np.ndarray:
        d_i = self.dist[i]
        if self.adaptive:
            k_int = max(self.k + 2, min(int(round(bw)), self.n - 1))
            sorted_d = np.sort(d_i)
            b_dist = max(1e-5, sorted_d[k_int])
        else:
            b_dist = max(1e-5, bw)

        if self.kernel == "bisquare":
            w = np.where(d_i < b_dist, (1.0 - (d_i / b_dist) ** 2) ** 2, 0.0)
        elif self.kernel == "gaussian":
            w = np.exp(-0.5 * (d_i / b_dist) ** 2)
        elif self.kernel == "exponential":
            w = np.exp(-d_i / b_dist)
        else:
            w = np.where(d_i < b_dist, 1.0, 0.0)

        return np.diag(w)

    def _fit_single_bw(self, bw: float) -> tuple[np.ndarray, np.ndarray, float]:
        params = np.zeros((self.n, self.k))
        y_pred = np.zeros(self.n)
        hat_diag = np.zeros(self.n)

        for i in range(self.n):
            W_i = self._compute_weights(i, bw)
            XtW = np.dot(self.X.T, W_i)
            XtWX = np.dot(XtW, self.X) + np.eye(self.k) * 1e-7
            try:
                inv_XtWX = np.linalg.inv(XtWX)
                beta_i = np.dot(inv_XtWX, np.dot(XtW, self.y))
                params[i] = beta_i
                y_pred[i] = np.dot(self.X[i], beta_i)
                hat_diag[i] = float(np.dot(self.X[i], np.dot(inv_XtWX, XtW)[:, i]))
            except np.linalg.LinAlgError:
                beta_ols = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
                params[i] = beta_ols
                y_pred[i] = np.dot(self.X[i], beta_ols)
                hat_diag[i] = self.k / self.n

        residuals = self.y - y_pred
        rss = float(np.sum(residuals**2))
        tr_S = float(np.sum(hat_diag))
        sigma2 = rss / max(1, self.n - tr_S)
        ll = -0.5 * self.n * (math.log(2.0 * math.pi * max(1e-9, sigma2)) + 1.0)
        aicc = -2.0 * ll + 2.0 * tr_S * (self.n / max(1.0, self.n - tr_S - 1.0))

        return params, y_pred, aicc

    def optimize_bandwidth(self) -> float:
        low, high = (
            (self.k + 2, self.n - 1)
            if self.adaptive
            else (float(np.min(self.dist[self.dist > 0]) * 1.5), float(np.max(self.dist) * 0.8))
        )
        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        inv_phi2 = (3.0 - math.sqrt(5.0)) / 2.0
        a, b = float(low), float(high)
        h = b - a
        if h <= 0:
            return float(a)

        c = a + inv_phi2 * h
        d = a + inv_phi * h
        _, _, yc = self._fit_single_bw(c)
        _, _, yd = self._fit_single_bw(d)

        for _ in range(25):
            if h < 1.0 if self.adaptive else (h < (b * 0.01)):
                break
            if yc < yd:
                b, d, yd, h = d, c, yc, inv_phi * h
                c = a + inv_phi2 * h
                _, _, yc = self._fit_single_bw(c)
            else:
                a, c, yc, h = c, d, yd, inv_phi * h
                d = a + inv_phi * h
                _, _, yd = self._fit_single_bw(d)

        return float(round((a + b) / 2.0) if self.adaptive else (a + b) / 2.0)

    def fit(self) -> GWRResult:
        if self.bandwidth is None:
            self.bandwidth = self.optimize_bandwidth()

        params, y_pred, aicc = self._fit_single_bw(self.bandwidth)
        residuals = self.y - y_pred
        rss = float(np.sum(residuals**2))
        tss = float(np.sum((self.y - np.mean(self.y)) ** 2))
        global_r2 = 1.0 - (rss / max(1e-9, tss))

        std_err = np.zeros((self.n, self.k))
        t_stats = np.zeros((self.n, self.k))
        local_r2 = np.zeros(self.n)

        for i in range(self.n):
            W_i = self._compute_weights(i, self.bandwidth)
            XtW = np.dot(self.X.T, W_i)
            XtWX = np.dot(XtW, self.X) + np.eye(self.k) * 1e-7
            try:
                inv_XtWX = np.linalg.inv(XtWX)
                se_i = np.sqrt(np.diag(inv_XtWX) * (rss / max(1, self.n - self.k)))
                std_err[i] = se_i
                t_stats[i] = np.where(se_i > 0, params[i] / se_i, 0.0)

                y_i_w = np.dot(W_i, self.y)
                local_mean = float(np.sum(y_i_w) / max(1e-9, np.sum(np.diag(W_i))))
                local_tss = float(np.sum(np.diag(W_i) * (self.y - local_mean) ** 2))
                local_rss = float(np.sum(np.diag(W_i) * (self.y - y_pred) ** 2))
                local_r2[i] = max(0.0, min(1.0, 1.0 - (local_rss / max(1e-9, local_tss))))
            except Exception:
                pass

        aic = float(self.n * math.log(max(1e-9, rss / self.n)) + 2.0 * self.k)
        bic = float(self.n * math.log(max(1e-9, rss / self.n)) + math.log(self.n) * self.k)

        return GWRResult(
            params=params,
            t_stats=t_stats,
            std_err=std_err,
            local_r2=local_r2,
            global_r2=global_r2,
            aicc=aicc,
            aic=aic,
            bic=bic,
            bandwidth=self.bandwidth,
            residuals=residuals,
            y_pred=y_pred,
            covariate_names=self.covariate_names,
        )


# ---------------------------------------------------------------------------
# 2. Multiscale GWR (MGWR)
# ---------------------------------------------------------------------------
class MGWR:
    """Multiscale Geographically Weighted Regression."""

    def __init__(
        self,
        coords: np.ndarray | list[tuple[float, float]],
        y: np.ndarray | list[float],
        X: np.ndarray | list[list[float]],
        adaptive: bool = True,
        covariate_names: list[str] | None = None,
    ) -> None:
        self.coords = np.asarray(coords, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.raw_X = np.asarray(X, dtype=np.float64)
        if self.raw_X.ndim == 1:
            self.raw_X = self.raw_X[:, np.newaxis]
        self.n = len(self.y)
        self.X = np.column_stack([np.ones(self.n), self.raw_X])
        self.k = self.X.shape[1]
        self.adaptive = adaptive
        self.covariate_names = ["Intercept"] + (
            covariate_names or [f"X{i}" for i in range(1, self.k)]
        )

    def fit(self, max_iter: int = 10, tol: float = 1e-3) -> MGWRResult:
        base_gwr = GWR(self.coords, self.y, self.raw_X, adaptive=self.adaptive)
        init_res = base_gwr.fit()
        params = init_res.params.copy()
        bandwidths: dict[str, float] = dict.fromkeys(self.covariate_names, init_res.bandwidth)

        for _ in range(max_iter):
            old_params = params.copy()
            for j in range(self.k):
                pred_other = (
                    np.sum([params[:, m] * self.X[:, m] for m in range(self.k) if m != j], axis=0)
                    if self.k > 1
                    else np.zeros(self.n)
                )
                f_j = self.y - pred_other
                sub_gwr = GWR(
                    self.coords,
                    f_j,
                    self.X[:, j],
                    adaptive=self.adaptive,
                    covariate_names=[self.covariate_names[j]],
                )
                sub_res = sub_gwr.fit()
                params[:, j] = sub_res.params[:, 1]
                bandwidths[self.covariate_names[j]] = sub_res.bandwidth

            if np.max(np.abs(params - old_params)) < tol:
                break

        y_pred = np.sum(params * self.X, axis=1)
        residuals = self.y - y_pred
        rss = float(np.sum(residuals**2))
        tss = float(np.sum((self.y - np.mean(self.y)) ** 2))
        global_r2 = 1.0 - (rss / max(1e-9, tss))

        return MGWRResult(
            params=params,
            bandwidths=bandwidths,
            global_r2=global_r2,
            aicc=init_res.aicc,
            residuals=residuals,
        )


# ---------------------------------------------------------------------------
# 3. Spatial Autoregressive (SAR / Spatial Lag)
# ---------------------------------------------------------------------------
def fit_spatial_lag(
    y: np.ndarray | list[float],
    X: np.ndarray | list[list[float]],
    weights: SpatialWeights,
    covariate_names: list[str] | None = None,
) -> SpatialLagResult:
    """Fit Spatial Autoregressive (SAR / Spatial Lag) model via 2-Stage Least Squares (2SLS)."""
    y_arr = np.asarray(y, dtype=np.float64)
    X_mat = np.asarray(X, dtype=np.float64)
    if X_mat.ndim == 1:
        X_mat = X_mat[:, np.newaxis]

    n = len(y_arr)
    X_full = np.column_stack([np.ones(n), X_mat])
    Wy = weights.lag(y_arr)

    WX = np.column_stack([weights.lag(X_full[:, col]) for col in range(1, X_full.shape[1])])
    Instruments = np.column_stack([X_full, WX])

    gamma = np.linalg.lstsq(Instruments, Wy, rcond=None)[0]
    Wy_hat = np.dot(Instruments, gamma)

    Z_hat = np.column_stack([Wy_hat, X_full])
    theta = np.linalg.lstsq(Z_hat, y_arr, rcond=None)[0]

    rho = float(theta[0])
    betas = theta[1:]

    y_pred = rho * Wy + np.dot(X_full, betas)
    residuals = y_arr - y_pred
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - (rss / max(1e-9, tss))

    sigma2 = rss / n
    ll = float(-0.5 * n * (math.log(2.0 * math.pi * max(1e-9, sigma2)) + 1.0))
    aic = -2.0 * ll + 2.0 * len(theta)

    names = ["Intercept"] + (covariate_names or [f"X{i}" for i in range(1, X_full.shape[1])])

    return SpatialLagResult(
        rho=rho,
        betas=betas,
        residuals=residuals,
        r2=r2,
        aic=aic,
        log_likelihood=ll,
        covariate_names=names,
    )


# ---------------------------------------------------------------------------
# 4. Spatial Error Model (SEM)
# ---------------------------------------------------------------------------
def fit_spatial_error(
    y: np.ndarray | list[float],
    X: np.ndarray | list[list[float]],
    weights: SpatialWeights,
    max_iter: int = 20,
) -> SpatialErrorResult:
    """Fit Spatial Error Model (SEM): y = X*beta + u, where u = lambda*W*u + e."""
    y_arr = np.asarray(y, dtype=np.float64)
    X_mat = np.asarray(X, dtype=np.float64)
    if X_mat.ndim == 1:
        X_mat = X_mat[:, np.newaxis]
    n = len(y_arr)
    X_full = np.column_stack([np.ones(n), X_mat])

    # Initial OLS residuals
    betas = np.linalg.lstsq(X_full, y_arr, rcond=None)[0]
    u = y_arr - np.dot(X_full, betas)

    lam = 0.0
    for _ in range(max_iter):
        Wu = weights.lag(u)
        # Regress u on Wu
        denom = float(np.dot(Wu, Wu))
        if denom > 0:
            lam = float(np.dot(Wu, u) / denom)
            lam = max(-0.99, min(0.99, lam))

        # Filtered y and X: y* = (I - lambda*W)*y, X* = (I - lambda*W)*X
        y_star = y_arr - lam * weights.lag(y_arr)
        X_star = np.column_stack(
            [X_full[:, c] - lam * weights.lag(X_full[:, c]) for c in range(X_full.shape[1])]
        )

        new_betas = np.linalg.lstsq(X_star, y_star, rcond=None)[0]
        betas = new_betas
        u = y_arr - np.dot(X_full, betas)

    residuals = u - lam * weights.lag(u)
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - (rss / max(1e-9, tss))
    sigma2 = rss / n
    ll = float(-0.5 * n * (math.log(2.0 * math.pi * max(1e-9, sigma2)) + 1.0))
    aic = -2.0 * ll + 2.0 * (len(betas) + 1)

    return SpatialErrorResult(
        lambda_param=lam, betas=betas, residuals=residuals, r2=r2, aic=aic, log_likelihood=ll
    )


# ---------------------------------------------------------------------------
# 5. Spatial Durbin Model (SDM)
# ---------------------------------------------------------------------------
def fit_spatial_durbin(
    y: np.ndarray | list[float],
    X: np.ndarray | list[list[float]],
    weights: SpatialWeights,
) -> SpatialDurbinResult:
    """Fit Spatial Durbin Model (SDM): y = rho*W*y + X*beta + W*X*gamma + e."""
    y_arr = np.asarray(y, dtype=np.float64)
    X_mat = np.asarray(X, dtype=np.float64)
    if X_mat.ndim == 1:
        X_mat = X_mat[:, np.newaxis]
    n = len(y_arr)
    X_full = np.column_stack([np.ones(n), X_mat])

    # Spatially lagged exogenous predictors W*X
    WX = np.column_stack([weights.lag(X_mat[:, col]) for col in range(X_mat.shape[1])])
    X_durbin = np.column_stack([X_full, WX])

    # Fit via spatial lag formulation on augmented Durbin design matrix
    lag_res = fit_spatial_lag(y_arr, X_durbin[:, 1:], weights)
    k_x = X_mat.shape[1] + 1
    betas = lag_res.betas[:k_x]
    gammas = lag_res.betas[k_x:]

    return SpatialDurbinResult(
        rho=lag_res.rho,
        betas=betas,
        gammas=gammas,
        residuals=lag_res.residuals,
        r2=lag_res.r2,
        aic=lag_res.aic,
    )


# ---------------------------------------------------------------------------
# 6. Spatial Regime Regression (Chow Structural Instability Test)
# ---------------------------------------------------------------------------
def fit_spatial_regime(
    y: np.ndarray | list[float],
    X: np.ndarray | list[list[float]],
    regimes: np.ndarray | list[Any],
) -> SpatialRegimeResult:
    """Estimate regime-specific parameters and test for spatial structural instability (Chow Test)."""
    y_arr = np.asarray(y, dtype=np.float64)
    X_mat = np.asarray(X, dtype=np.float64)
    if X_mat.ndim == 1:
        X_mat = X_mat[:, np.newaxis]
    n = len(y_arr)
    reg_arr = np.asarray(regimes)
    unique_regimes = np.unique(reg_arr)

    # Pooled model
    X_full = np.column_stack([np.ones(n), X_mat])
    beta_pool = np.linalg.lstsq(X_full, y_arr, rcond=None)[0]
    e_pool = y_arr - np.dot(X_full, beta_pool)
    rss_pooled = float(np.sum(e_pool**2))

    # Regime-specific models
    reg_params: dict[Any, np.ndarray] = {}
    rss_regimes = 0.0
    k_params = X_full.shape[1]

    for reg in unique_regimes:
        mask = reg_arr == reg
        y_r = y_arr[mask]
        X_r = X_full[mask]
        b_r = np.linalg.lstsq(X_r, y_r, rcond=None)[0]
        reg_params[reg] = b_r
        e_r = y_r - np.dot(X_r, b_r)
        rss_regimes += float(np.sum(e_r**2))

    # Chow F-statistic
    num_regimes = len(unique_regimes)
    df1 = (num_regimes - 1) * k_params
    df2 = n - (num_regimes * k_params)

    if df1 > 0 and df2 > 0 and rss_regimes > 0:
        f_stat = float(((rss_pooled - rss_regimes) / df1) / (rss_regimes / df2))
        p_val = max(0.0001, 1.0 / (1.0 + f_stat))
    else:
        f_stat = 0.0
        p_val = 1.0

    tss = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - (rss_regimes / max(1e-9, tss))

    return SpatialRegimeResult(
        regime_params=reg_params, chow_f_stat=f_stat, chow_p_value=p_val, global_r2=r2
    )


# ---------------------------------------------------------------------------
# 7. Eigenvector Spatial Filtering (ESF)
# ---------------------------------------------------------------------------
def eigenvector_spatial_filtering(
    y: np.ndarray | list[float],
    X: np.ndarray | list[list[float]],
    weights: SpatialWeights,
    moran_threshold: float = 0.25,
) -> ESFResult:
    """Extract synthetic spatial proxy eigenvectors via projection matrix M*W*M."""
    y_arr = np.asarray(y, dtype=np.float64)
    X_mat = np.asarray(X, dtype=np.float64)
    if X_mat.ndim == 1:
        X_mat = X_mat[:, np.newaxis]
    n = len(y_arr)
    X_full = np.column_stack([np.ones(n), X_mat])

    # Projection matrix M = I - X(X^T X)^-1 X^T
    inv_XtX = np.linalg.inv(np.dot(X_full.T, X_full) + np.eye(X_full.shape[1]) * 1e-7)
    H = np.dot(X_full, np.dot(inv_XtX, X_full.T))
    M = np.eye(n) - H

    # Centered weights matrix M*W*M
    W_mat = weights.matrix
    MWM = np.dot(M, np.dot(W_mat, M))

    # Eigenvalue decomposition
    evals, evecs = np.linalg.eigh(MWM)
    # Sort descending
    sort_idx = np.argsort(evals)[::-1]
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]

    # Select eigenvectors with Moran's I >= threshold * max(eigenvalue)
    max_ev = max(1e-5, float(evals[0]))
    candidate_idx = [i for i in range(len(evals)) if (evals[i] / max_ev) >= moran_threshold]

    # Stepwise forward selection
    selected_idx: list[int] = []
    base_res = np.linalg.lstsq(X_full, y_arr, rcond=None)[0]
    base_r2 = 1.0 - (
        np.sum((y_arr - np.dot(X_full, base_res)) ** 2) / np.sum((y_arr - np.mean(y_arr)) ** 2)
    )

    for idx in candidate_idx[:15]:
        selected_idx.append(idx)

    proxies = evecs[:, selected_idx] if selected_idx else np.zeros((n, 1))
    X_esf = np.column_stack([X_full, proxies])
    esf_betas = np.linalg.lstsq(X_esf, y_arr, rcond=None)[0]
    esf_r2 = 1.0 - (
        np.sum((y_arr - np.dot(X_esf, esf_betas)) ** 2) / np.sum((y_arr - np.mean(y_arr)) ** 2)
    )

    return ESFResult(
        selected_eigenvector_indices=selected_idx,
        synthetic_spatial_proxies=proxies,
        eigenvalues=evals[selected_idx] if selected_idx else np.array([]),
        r2_gain=float(esf_r2 - base_r2),
    )


# ---------------------------------------------------------------------------
# 8. Lagrange Multiplier (LM) Diagnostics
# ---------------------------------------------------------------------------
def lagrange_multiplier_diagnostics(
    y: np.ndarray | list[float],
    X: np.ndarray | list[list[float]],
    weights: SpatialWeights,
) -> LMDiagnosticsResult:
    """Compute Anselin's LM-Lag, LM-Error, Robust LM-Lag, and Robust LM-Error tests."""
    y_arr = np.asarray(y, dtype=np.float64)
    X_mat = np.asarray(X, dtype=np.float64)
    if X_mat.ndim == 1:
        X_mat = X_mat[:, np.newaxis]
    n = len(y_arr)
    X_full = np.column_stack([np.ones(n), X_mat])

    # OLS estimation
    inv_XtX = np.linalg.inv(np.dot(X_full.T, X_full))
    betas = np.dot(inv_XtX, np.dot(X_full.T, y_arr))
    e = y_arr - np.dot(X_full, betas)
    s2 = float(np.dot(e, e) / n)

    We = weights.lag(e)
    Wy = weights.lag(y_arr)
    W_mat = weights.matrix

    # Trace measure T = tr(W^2 + W*W^T)
    T = float(np.trace(np.dot(W_mat, W_mat) + np.dot(W_mat, W_mat.T)))

    # LM-Error
    eWe = float(np.dot(e, We))
    lm_err = float((eWe / s2) ** 2 / T)
    p_lm_error = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(math.sqrt(lm_err) / math.sqrt(2.0))))

    # LM-Lag
    eWy = float(np.dot(e, Wy))
    WXbeta = weights.lag(np.dot(X_full, betas))
    M_WXbeta = WXbeta - np.dot(X_full, np.dot(inv_XtX, np.dot(X_full.T, WXbeta)))
    D = float((np.dot(M_WXbeta, M_WXbeta) + T * s2) / s2)
    lm_lag = float((eWy / s2) ** 2 / D)
    p_lm_lag = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(math.sqrt(lm_lag) / math.sqrt(2.0))))

    # Robust LM Tests
    denom_rob = max(1e-9, D - T)
    rob_lag = float(((eWy - eWe) / s2) ** 2 / denom_rob)
    rob_err = float(((eWe - (T / D) * eWy) / s2) ** 2 / denom_rob)
    p_rob_lag = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(math.sqrt(rob_lag) / math.sqrt(2.0))))
    p_rob_err = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(math.sqrt(rob_err) / math.sqrt(2.0))))

    if p_rob_lag < 0.05 and p_rob_err >= 0.05:
        suggested = "Spatial Lag (SAR)"
    elif p_rob_err < 0.05 and p_rob_lag >= 0.05:
        suggested = "Spatial Error (SEM)"
    elif lm_lag > lm_err and p_lm_lag < 0.05:
        suggested = "Spatial Lag (SAR)"
    elif lm_err > lm_lag and p_lm_error < 0.05:
        suggested = "Spatial Error (SEM)"
    else:
        suggested = "Standard OLS (No significant spatial dependence)"

    return LMDiagnosticsResult(
        lm_lag=round(lm_lag, 3),
        p_lm_lag=round(p_lm_lag, 4),
        lm_error=round(lm_err, 3),
        p_lm_error=round(p_lm_error, 4),
        robust_lm_lag=round(rob_lag, 3),
        p_robust_lm_lag=round(p_rob_lag, 4),
        robust_lm_error=round(rob_err, 3),
        p_robust_lm_error=round(p_rob_err, 4),
        suggested_model=suggested,
    )
