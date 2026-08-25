# -*- coding: utf-8 -*-
"""
Spatial Econometrics & Local Regression: GWR, MGWR, SAR, SEM, and ESF.
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


# ---------------------------------------------------------------------------
# Geographically Weighted Regression (GWR)
# ---------------------------------------------------------------------------
class GWR:
    """Geographically Weighted Regression (GWR) with automatic bandwidth optimization."""

    def __init__(
        self,
        coords: np.ndarray | list[tuple[float, float]],
        y: np.ndarray | list[float],
        X: np.ndarray | list[list[float]],
        kernel: str = "bisquare",  # bisquare, gaussian, exponential
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
        # Add intercept column
        self.X = np.column_stack([np.ones(self.n), self.raw_X])
        self.k = self.X.shape[1]

        self.kernel = kernel.lower()
        self.adaptive = adaptive
        self.bandwidth = bandwidth

        if covariate_names is None:
            self.covariate_names = ["Intercept"] + [f"X{i}" for i in range(1, self.k)]
        else:
            self.covariate_names = ["Intercept"] + covariate_names

        # Precompute distance matrix
        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        self.dist = np.sqrt(np.sum(diff**2, axis=-1))

    def _compute_weights(self, i: int, bw: float) -> np.ndarray:
        d_i = self.dist[i]
        if self.adaptive:
            # Bandwidth is k-nearest count
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
            XtWX = np.dot(XtW, self.X)

            # Ridge regularizer for singular local matrices
            XtWX_reg = XtWX + np.eye(self.k) * 1e-7
            try:
                inv_XtWX = np.linalg.inv(XtWX_reg)
                beta_i = np.dot(inv_XtWX, np.dot(XtW, self.y))
                params[i] = beta_i
                y_pred[i] = np.dot(self.X[i], beta_i)
                # Influence (hat matrix diagonal element)
                hat_diag[i] = float(np.dot(self.X[i], np.dot(inv_XtWX, XtW)[:, i]))
            except np.linalg.LinAlgError:
                # Fallback to global OLS
                beta_ols = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
                params[i] = beta_ols
                y_pred[i] = np.dot(self.X[i], beta_ols)
                hat_diag[i] = self.k / self.n

        residuals = self.y - y_pred
        rss = float(np.sum(residuals**2))
        tr_S = float(np.sum(hat_diag))

        # Hurvich AICc formulation
        sigma2 = rss / max(1, self.n - tr_S)
        ll = -0.5 * self.n * (math.log(2.0 * math.pi * max(1e-9, sigma2)) + 1.0)
        aicc = -2.0 * ll + 2.0 * tr_S * (self.n / max(1.0, self.n - tr_S - 1.0))

        return params, y_pred, aicc

    def optimize_bandwidth(self) -> float:
        """Golden Section Search to find optimal bandwidth minimizing AICc."""
        if self.adaptive:
            low, high = self.k + 2, self.n - 1
        else:
            low = float(np.min(self.dist[self.dist > 0]) * 1.5)
            high = float(np.max(self.dist) * 0.8)

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
                b = d
                d = c
                yd = yc
                h = inv_phi * h
                c = a + inv_phi2 * h
                _, _, yc = self._fit_single_bw(c)
            else:
                a = c
                c = d
                yc = yd
                h = inv_phi * h
                d = a + inv_phi * h
                _, _, yd = self._fit_single_bw(d)

        return float(round((a + b) / 2.0) if self.adaptive else (a + b) / 2.0)

    def fit(self) -> GWRResult:
        """Fit GWR model and return comprehensive parameter statistics."""
        if self.bandwidth is None:
            self.bandwidth = self.optimize_bandwidth()

        params, y_pred, aicc = self._fit_single_bw(self.bandwidth)
        residuals = self.y - y_pred
        rss = float(np.sum(residuals**2))
        tss = float(np.sum((self.y - np.mean(self.y)) ** 2))
        global_r2 = 1.0 - (rss / max(1e-9, tss))

        # Standard errors & local t-statistics
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

                # Local R2
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
# Multiscale Geographically Weighted Regression (MGWR)
# ---------------------------------------------------------------------------
class MGWR:
    """Multiscale Geographically Weighted Regression with variable-specific bandwidths."""

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
        """Iterative backfitting algorithm to estimate scale-specific bandwidths."""
        # Initialize with standard GWR
        base_gwr = GWR(self.coords, self.y, self.raw_X, adaptive=self.adaptive)
        init_res = base_gwr.fit()
        params = init_res.params.copy()
        bandwidths: dict[str, float] = dict.fromkeys(self.covariate_names, init_res.bandwidth)

        # Backfitting iterations
        for _ in range(max_iter):
            old_params = params.copy()
            for j in range(self.k):
                # Partial residual for covariate j
                pred_other = (
                    np.sum([params[:, m] * self.X[:, m] for m in range(self.k) if m != j], axis=0)
                    if self.k > 1
                    else np.zeros(self.n)
                )
                f_j = self.y - pred_other

                # Univariate GWR on partial residual
                sub_gwr = GWR(
                    self.coords,
                    f_j,
                    self.X[:, j],
                    adaptive=self.adaptive,
                    covariate_names=[self.covariate_names[j]],
                )
                sub_res = sub_gwr.fit()
                params[:, j] = sub_res.params[:, 1]  # Slope parameter
                bandwidths[self.covariate_names[j]] = sub_res.bandwidth

            diff = np.max(np.abs(params - old_params))
            if diff < tol:
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
# Spatial Autoregressive (SAR) / Spatial Lag Model
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

    # Instruments: W*X (Spatial lag of exogenous predictors)
    WX = np.column_stack([weights.lag(X_full[:, col]) for col in range(1, X_full.shape[1])])
    Instruments = np.column_stack([X_full, WX])

    # Stage 1: Regress Wy on Instruments
    gamma = np.linalg.lstsq(Instruments, Wy, rcond=None)[0]
    Wy_hat = np.dot(Instruments, gamma)

    # Stage 2: Regress y on Wy_hat and X_full
    Z_hat = np.column_stack([Wy_hat, X_full])
    theta = np.linalg.lstsq(Z_hat, y_arr, rcond=None)[0]

    rho = float(theta[0])
    betas = theta[1:]

    # Residuals & Metrics
    y_pred = rho * Wy + np.dot(X_full, betas)
    residuals = y_arr - y_pred
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - (rss / max(1e-9, tss))

    sigma2 = rss / n
    ll = float(-0.5 * n * (math.log(2.0 * math.pi * max(1e-9, sigma2)) + 1.0))
    k_params = len(theta)
    aic = -2.0 * ll + 2.0 * k_params

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
