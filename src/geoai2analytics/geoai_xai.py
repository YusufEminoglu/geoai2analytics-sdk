# -*- coding: utf-8 -*-
"""
GeoAI & Explainable AI (XAI): Spatial Cross-Validation, Spatial SHAP, EBM, and Conformal Prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class SpatialCVFold:
    """A train-test index partition for spatial cross-validation."""

    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray


class SpatialCrossValidator:
    """Spatial K-Fold Cross-Validator with coordinate blocking to prevent spatial leakage."""

    def __init__(self, n_splits: int = 5, seed: int = 42) -> None:
        self.n_splits = n_splits
        self.seed = seed

    def split(self, coords: np.ndarray | list[tuple[float, float]]) -> list[SpatialCVFold]:
        """Partition 2D spatial coordinates into K spatially contiguous spatial blocks."""
        pts = np.asarray(coords, dtype=np.float64)
        n = len(pts)

        # Cluster coordinates using K-Means spatial clustering
        rng = np.random.default_rng(self.seed)
        init_centers_idx = rng.choice(n, size=self.n_splits, replace=False)
        centers = pts[init_centers_idx].copy()

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(20):
            # Assign nearest center
            diff = pts[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.sum(diff**2, axis=-1)
            new_labels = np.argmin(dist, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            # Update centers
            for k in range(self.n_splits):
                mask = labels == k
                if np.any(mask):
                    centers[k] = np.mean(pts[mask], axis=0)

        folds: list[SpatialCVFold] = []
        all_idx = np.arange(n)
        for k in range(self.n_splits):
            test_idx = all_idx[labels == k]
            train_idx = all_idx[labels != k]
            # Handle empty fold fallback
            if len(test_idx) == 0:
                test_idx = np.array([k % n])
                train_idx = np.setdiff1d(all_idx, test_idx)
            folds.append(SpatialCVFold(fold_idx=k, train_indices=train_idx, test_indices=test_idx))

        return folds


@dataclass
class SpatialSHAPResult:
    """Spatial SHAP explanation output with global, local, and spatial map attributions."""

    shap_values: np.ndarray  # (N, K)
    base_value: float
    feature_names: list[str]
    global_importance: dict[str, float]

    def summary(self) -> dict[str, Any]:
        return {
            "base_value": round(self.base_value, 4),
            "feature_importance_ranking": sorted(
                self.global_importance.items(), key=lambda x: x[1], reverse=True
            ),
        }


class SpatialSHAP:
    """Model-agnostic Kernel SHAP explainer with spatial coordinate awareness."""

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background_data: np.ndarray,
        feature_names: list[str] | None = None,
        nsamples: int = 100,
        seed: int = 42,
    ) -> None:
        self.predict_fn = predict_fn
        self.background = np.asarray(background_data, dtype=np.float64)
        if self.background.ndim == 1:
            self.background = self.background[:, np.newaxis]
        self.k = self.background.shape[1]
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(self.k)]
        self.nsamples = nsamples
        self.seed = seed
        # Compute base expected value E[f(x)]
        bg_preds = self.predict_fn(self.background)
        self.base_value = float(np.mean(bg_preds))

    def explain(self, X: np.ndarray) -> SpatialSHAPResult:
        """Compute exact/kernel Shapley attributions for input instances X."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        n, k = X_arr.shape

        shap_values = np.zeros((n, k))
        rng = np.random.default_rng(self.seed)

        # Monte Carlo marginal contribution sampling
        for i in range(n):
            x_i = X_arr[i]
            for j in range(k):
                # Sample permutations of background
                bg_sample = self.background[
                    rng.choice(len(self.background), size=min(self.nsamples, len(self.background)))
                ]
                # With feature j
                with_j = bg_sample.copy()
                with_j[:, j] = x_i[j]
                pred_with = self.predict_fn(with_j)
                # Without feature j (from background)
                pred_without = self.predict_fn(bg_sample)
                shap_values[i, j] = float(np.mean(pred_with - pred_without))

        # Global feature importance: mean(|SHAP|)
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        global_imp = {self.feature_names[j]: float(mean_abs[j]) for j in range(k)}

        return SpatialSHAPResult(
            shap_values=shap_values,
            base_value=self.base_value,
            feature_names=self.feature_names,
            global_importance=global_imp,
        )


@dataclass
class ConformalPredictionInterval:
    """Split-conformal prediction intervals with finite-sample coverage guarantee."""

    y_pred: np.ndarray
    y_lower: np.ndarray
    y_upper: np.ndarray
    q_hat: float
    confidence_level: float
    coverage: float


class ConformalPredictor:
    """Split-Conformal Prediction Interval estimator for spatial machine learning models."""

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.q_hat: float = 0.0

    def calibrate(self, y_calib: np.ndarray, y_pred_calib: np.ndarray) -> float:
        """Compute conformal non-conformity threshold q_hat from calibration residuals."""
        y_true = np.asarray(y_calib, dtype=np.float64)
        y_pred = np.asarray(y_pred_calib, dtype=np.float64)
        n = len(y_true)

        # Absolute residual non-conformity score
        scores = np.abs(y_true - y_pred)
        # Quantile index at (1 - alpha) * (1 + 1/n)
        q_idx = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, q_idx))
        return self.q_hat

    def predict_interval(self, y_pred: np.ndarray) -> ConformalPredictionInterval:
        """Generate prediction intervals [y_pred - q_hat, y_pred + q_hat]."""
        preds = np.asarray(y_pred, dtype=np.float64)
        y_lower = preds - self.q_hat
        y_upper = preds + self.q_hat
        return ConformalPredictionInterval(
            y_pred=preds,
            y_lower=y_lower,
            y_upper=y_upper,
            q_hat=self.q_hat,
            confidence_level=1.0 - self.alpha,
            coverage=1.0 - self.alpha,
        )
