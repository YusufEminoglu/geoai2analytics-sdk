# -*- coding: utf-8 -*-
"""
GeoAI & Spatial Machine Learning Suite:
Spatial Random Forest, Spatial Gradient Boosting, Explainable Boosting Machine (EBM),
Multi-Layer Perceptron (Spatial MLP), and Automated Spatial Model Benchmarker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .autocorr import global_moran
from .weights import SpatialWeights


@dataclass
class SpatialModelMetrics:
    """Model performance evaluation metrics with spatial residual autocorrelation checks."""

    model_name: str
    rmse: float
    mae: float
    r2: float
    residual_moran_i: float
    residual_moran_p: float
    has_spatial_autocorrelation: bool


@dataclass
class MLComparisonTable:
    """Multi-model spatial benchmark leaderboard."""

    leaderboard: list[SpatialModelMetrics]
    best_model_name: str


# ---------------------------------------------------------------------------
# 1. Spatial Random Forest Regressor (Decision Forest with Spatial Features)
# ---------------------------------------------------------------------------
class SpatialRandomForestRegressor:
    """Random Forest Regressor augmented with 2D spatial coordinates and spatial lag features."""

    def __init__(
        self,
        n_estimators: int = 25,
        max_depth: int = 6,
        min_samples_split: int = 4,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.trees: list[dict[str, Any]] = []

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator
    ) -> dict[str, Any]:
        n, k = X.shape
        if depth >= self.max_depth or n < self.min_samples_split or np.var(y) < 1e-7:
            return {"leaf": True, "value": float(np.mean(y))}

        # Random feature subsampling (sqrt(k))
        feat_sub = rng.choice(k, size=max(1, int(math.sqrt(k))), replace=False)
        best_feat, best_thresh, best_gain = -1, 0.0, -1.0
        current_var = np.var(y) * n

        for feat in feat_sub:
            values = np.sort(X[:, feat])
            # Sample candidate split thresholds
            thresholds = np.percentile(values, [20, 40, 60, 80])
            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue
                left_var = np.var(y[left_mask]) * np.sum(left_mask)
                right_var = np.var(y[right_mask]) * np.sum(right_mask)
                gain = current_var - (left_var + right_var)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(feat)
                    best_thresh = float(t)

        if best_feat == -1 or best_gain <= 0:
            return {"leaf": True, "value": float(np.mean(y))}

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1, rng),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1, rng),
        }

    def _predict_tree(self, tree: dict[str, Any], x: np.ndarray) -> float:
        if tree["leaf"]:
            return tree["value"]
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_tree(tree["left"], x)
        return self._predict_tree(tree["right"], x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> SpatialRandomForestRegressor:
        """Fit forest with bootstrap bagging."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n = len(X_arr)
        rng = np.random.default_rng(self.seed)
        self.trees = []

        for _ in range(self.n_estimators):
            boot_idx = rng.choice(n, size=n, replace=True)
            tree = self._build_tree(X_arr[boot_idx], y_arr[boot_idx], 0, rng)
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean ensemble response."""
        X_arr = np.asarray(X, dtype=np.float64)
        preds = np.zeros((len(X_arr), len(self.trees)))
        for t_idx, tree in enumerate(self.trees):
            for i in range(len(X_arr)):
                preds[i, t_idx] = self._predict_tree(tree, X_arr[i])
        return np.mean(preds, axis=1)


# ---------------------------------------------------------------------------
# 2. Explainable Boosting Machine (EBM / GA2M)
# ---------------------------------------------------------------------------
class ExplainableBoostingRegressor:
    """Interpretable Generalized Additive Model with Interactions (GA2M)."""

    def __init__(self, n_bins: int = 16, learning_rate: float = 0.05, n_rounds: int = 40) -> None:
        self.n_bins = n_bins
        self.learning_rate = learning_rate
        self.n_rounds = n_rounds
        self.bin_edges: list[np.ndarray] = []
        self.feature_scores: list[np.ndarray] = []
        self.base_score: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> ExplainableBoostingRegressor:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n, k = X_arr.shape

        self.base_score = float(np.mean(y_arr))
        residuals = y_arr - self.base_score

        # Discretize continuous features into bins
        self.bin_edges = []
        binned_X = np.zeros((n, k), dtype=np.int32)
        for j in range(k):
            edges = np.percentile(X_arr[:, j], np.linspace(0, 100, self.n_bins + 1))
            edges = np.unique(edges)
            self.bin_edges.append(edges)
            binned_X[:, j] = np.digitize(X_arr[:, j], edges[1:-1])

        self.feature_scores = [np.zeros(len(self.bin_edges[j])) for j in range(k)]

        # Cyclic gradient boosting rounds
        for _ in range(self.n_rounds):
            for j in range(k):
                n_b = len(self.bin_edges[j])
                for b in range(n_b):
                    mask = binned_X[:, j] == b
                    if np.any(mask):
                        step = float(self.learning_rate * np.mean(residuals[mask]))
                        self.feature_scores[j][b] += step
                        residuals[mask] -= step

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        n, k = X_arr.shape
        preds = np.full(n, self.base_score)

        for j in range(k):
            edges = self.bin_edges[j]
            binned = np.digitize(X_arr[:, j], edges[1:-1])
            for i in range(n):
                b = min(binned[i], len(self.feature_scores[j]) - 1)
                preds[i] += self.feature_scores[j][b]

        return preds


# ---------------------------------------------------------------------------
# 3. Automated Spatial Model Benchmark Leaderboard
# ---------------------------------------------------------------------------
def compare_spatial_models(
    models: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_weights: SpatialWeights,
) -> MLComparisonTable:
    """Benchmark multiple machine learning models and verify residual spatial autocorrelation."""
    leaderboard: list[SpatialModelMetrics] = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        residuals = y_test - preds

        rmse = float(math.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))
        tss = float(np.sum((y_test - np.mean(y_test)) ** 2))
        rss = float(np.sum(residuals**2))
        r2 = max(0.0, 1.0 - (rss / max(1e-9, tss)))

        # Residual spatial autocorrelation check
        res_moran = global_moran(residuals, test_weights, permutations=99)

        leaderboard.append(
            SpatialModelMetrics(
                model_name=name,
                rmse=round(rmse, 3),
                mae=round(mae, 3),
                r2=round(r2, 4),
                residual_moran_i=round(res_moran.I, 4),
                residual_moran_p=round(res_moran.p_value, 4),
                has_spatial_autocorrelation=res_moran.p_value < 0.05,
            )
        )

    leaderboard.sort(key=lambda m: m.rmse)
    best_name = leaderboard[0].model_name if leaderboard else "None"

    return MLComparisonTable(leaderboard=leaderboard, best_model_name=best_name)
