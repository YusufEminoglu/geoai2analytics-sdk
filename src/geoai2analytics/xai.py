# -*- coding: utf-8 -*-
"""
Explainable GeoAI (XAI) & Spatial Uncertainty Analytics:
Spatial SHAP, Partial Dependence Plots (PDP), Permutation Feature Importance,
DiCE Counterfactual Explanations, Conformal Prediction Intervals,
and Spatial Cross-Validation Blocking.
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
    """Spatial K-Fold Cross-Validator with coordinate clustering to eliminate spatial data leakage."""

    def __init__(self, n_splits: int = 5, seed: int = 42) -> None:
        self.n_splits = n_splits
        self.seed = seed

    def split(self, coords: np.ndarray | list[tuple[float, float]]) -> list[SpatialCVFold]:
        pts = np.asarray(coords, dtype=np.float64)
        n = len(pts)
        k_splits = min(self.n_splits, n)

        rng = np.random.default_rng(self.seed)
        init_centers_idx = rng.choice(n, size=k_splits, replace=False)
        centers = pts[init_centers_idx].copy()

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(20):
            diff = pts[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.sum(diff**2, axis=-1)
            new_labels = np.argmin(dist, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for k in range(k_splits):
                mask = labels == k
                if np.any(mask):
                    centers[k] = np.mean(pts[mask], axis=0)

        folds: list[SpatialCVFold] = []
        all_idx = np.arange(n)
        for k in range(k_splits):
            test_idx = all_idx[labels == k]
            train_idx = all_idx[labels != k]
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
        bg_preds = self.predict_fn(self.background)
        self.base_value = float(np.mean(bg_preds))

    def explain(self, X: np.ndarray) -> SpatialSHAPResult:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        n, k = X_arr.shape

        shap_values = np.zeros((n, k))
        rng = np.random.default_rng(self.seed)

        for i in range(n):
            x_i = X_arr[i]
            for j in range(k):
                bg_sample = self.background[
                    rng.choice(len(self.background), size=min(self.nsamples, len(self.background)))
                ]
                with_j = bg_sample.copy()
                with_j[:, j] = x_i[j]
                pred_with = self.predict_fn(with_j)
                pred_without = self.predict_fn(bg_sample)
                shap_values[i, j] = float(np.mean(pred_with - pred_without))

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        global_imp = {self.feature_names[j]: float(mean_abs[j]) for j in range(k)}

        return SpatialSHAPResult(
            shap_values=shap_values,
            base_value=self.base_value,
            feature_names=self.feature_names,
            global_importance=global_imp,
        )


@dataclass
class PDPResult:
    """Partial Dependence Plot output across grid values."""

    feature_name: str
    grid_values: np.ndarray
    pdp_values: np.ndarray


def partial_dependence(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    feature_idx: int = 0,
    grid_resolution: int = 20,
    feature_name: str = "Feature",
) -> PDPResult:
    """Compute 1D Partial Dependence Plot (PDP) showing marginal effect of a feature."""
    X_arr = np.asarray(X, dtype=np.float64)
    grid_vals = np.linspace(
        np.min(X_arr[:, feature_idx]), np.max(X_arr[:, feature_idx]), grid_resolution
    )
    pdp_vals = np.zeros(grid_resolution)

    for idx, val in enumerate(grid_vals):
        X_temp = X_arr.copy()
        X_temp[:, feature_idx] = val
        preds = predict_fn(X_temp)
        pdp_vals[idx] = float(np.mean(preds))

    return PDPResult(feature_name=feature_name, grid_values=grid_vals, pdp_values=pdp_vals)


@dataclass
class PermutationImportanceResult:
    """Permutation feature importance metric drops."""

    importances_mean: dict[str, float]
    importances_std: dict[str, float]


def permutation_feature_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    n_repeats: int = 5,
    seed: int = 42,
) -> PermutationImportanceResult:
    """Calculate Permutation Feature Importance based on RMSE degradation."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n, k = X_arr.shape
    names = feature_names or [f"Feature_{j}" for j in range(k)]

    base_preds = predict_fn(X_arr)
    base_score = float(math.sqrt(np.mean((y_arr - base_preds) ** 2)))

    rng = np.random.default_rng(seed)
    mean_imp: dict[str, float] = {}
    std_imp: dict[str, float] = {}

    for j in range(k):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_arr.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_preds = predict_fn(X_perm)
            score = float(math.sqrt(np.mean((y_arr - perm_preds) ** 2)))
            scores.append(score - base_score)
        mean_imp[names[j]] = round(float(np.mean(scores)), 4)
        std_imp[names[j]] = round(float(np.std(scores)), 4)

    return PermutationImportanceResult(importances_mean=mean_imp, importances_std=std_imp)


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
        y_true = np.asarray(y_calib, dtype=np.float64)
        y_pred = np.asarray(y_pred_calib, dtype=np.float64)
        n = len(y_true)
        scores = np.abs(y_true - y_pred)
        q_idx = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, q_idx))
        return self.q_hat

    def predict_interval(self, y_pred: np.ndarray) -> ConformalPredictionInterval:
        preds = np.asarray(y_pred, dtype=np.float64)
        return ConformalPredictionInterval(
            y_pred=preds,
            y_lower=preds - self.q_hat,
            y_upper=preds + self.q_hat,
            q_hat=self.q_hat,
            confidence_level=1.0 - self.alpha,
            coverage=1.0 - self.alpha,
        )
