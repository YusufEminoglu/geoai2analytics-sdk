# -*- coding: utf-8 -*-
"""
Spatial Data Hygiene, Quality Auditing, Similarity Search, and Workflow Recommendation:
Data Readiness Audit, Mahalanobis Spatial Similarity Search, and GeoAI Workflow Advisor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .weights import SpatialWeights


@dataclass
class AuditIssue:
    """A detected spatial or numerical data hygiene issue."""

    severity: str  # Critical, Warning, Info
    issue_type: str
    message: str


@dataclass
class DataReadinessReport:
    """Comprehensive spatial dataset quality and readiness audit."""

    sample_size: int
    n_features: int
    has_critical_issues: bool
    vif_scores: dict[str, float]
    skewness_scores: dict[str, float]
    island_count: int
    issues: list[AuditIssue]


@dataclass
class SimilarityMatch:
    """A candidate feature matched by spatial attribute similarity."""

    candidate_index: int
    similarity_score: float  # 0.0 to 1.0 (1.0 = identical)
    distance: float


# ---------------------------------------------------------------------------
# 1. Spatial Data Readiness Audit
# ---------------------------------------------------------------------------
def audit_spatial_data(
    X: np.ndarray | list[list[float]],
    coords: np.ndarray | list[tuple[float, float]] | None = None,
    weights: SpatialWeights | None = None,
    feature_names: list[str] | None = None,
) -> DataReadinessReport:
    """Audit data quality, variance, multicollinearity (VIF), skewness, and spatial graph islands."""
    X_mat = np.asarray(X, dtype=np.float64)
    if X_mat.ndim == 1:
        X_mat = X_mat[:, np.newaxis]
    n, k = X_mat.shape
    names = feature_names or [f"Feature_{j}" for j in range(k)]

    issues: list[AuditIssue] = []
    vif_dict: dict[str, float] = {}
    skew_dict: dict[str, float] = {}

    # Check sample size
    if n < 30:
        issues.append(
            AuditIssue(
                "Warning",
                "Small Sample Size",
                f"Dataset contains only N={n} units; statistical power is limited.",
            )
        )

    # Check feature variances & skewness
    for j in range(k):
        col = X_mat[:, j]
        var = float(np.var(col))
        if var < 1e-9:
            issues.append(
                AuditIssue(
                    "Critical",
                    "Zero Variance",
                    f"Feature '{names[j]}' has constant value across all observations.",
                )
            )

        # Skewness
        std = math.sqrt(var) if var > 0 else 1.0
        skew = float(np.mean(((col - np.mean(col)) / std) ** 3))
        skew_dict[names[j]] = round(skew, 2)
        if abs(skew) > 2.0:
            issues.append(
                AuditIssue(
                    "Info",
                    "High Skewness",
                    f"Feature '{names[j]}' is strongly skewed (skewness={skew:.2f}); consider log/box-cox transform.",
                )
            )

    # Multicollinearity & Variance Inflation Factor (VIF)
    if k > 1:
        for j in range(k):
            y_j = X_mat[:, j]
            X_other = np.delete(X_mat, j, axis=1)
            X_other_full = np.column_stack([np.ones(n), X_other])
            try:
                betas = np.linalg.lstsq(X_other_full, y_j, rcond=None)[0]
                residuals = y_j - np.dot(X_other_full, betas)
                r2 = 1.0 - (np.sum(residuals**2) / max(1e-9, np.sum((y_j - np.mean(y_j)) ** 2)))
                vif = float(1.0 / max(1e-5, (1.0 - r2)))
            except Exception:
                vif = 1.0
            vif_dict[names[j]] = round(vif, 2)
            if vif > 7.5:
                issues.append(
                    AuditIssue(
                        "Warning",
                        "Multicollinearity",
                        f"Feature '{names[j]}' exhibits high multicollinearity (VIF={vif:.1f} > 7.5).",
                    )
                )

    # Spatial weights island check
    island_count = 0
    if weights is not None:
        row_sums = np.sum(weights.matrix, axis=1)
        island_count = int(np.sum(row_sums == 0))
        if island_count > 0:
            issues.append(
                AuditIssue(
                    "Warning",
                    "Spatial Islands",
                    f"Found {island_count} spatial units with zero connected neighbors.",
                )
            )

    has_crit = any(i.severity == "Critical" for i in issues)

    return DataReadinessReport(
        sample_size=n,
        n_features=k,
        has_critical_issues=has_crit,
        vif_scores=vif_dict,
        skewness_scores=skew_dict,
        island_count=island_count,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# 2. Similarity Search (Mahalanobis / Attribute Distance)
# ---------------------------------------------------------------------------
def similarity_search(
    target_profile: np.ndarray | list[float],
    candidate_features: np.ndarray | list[list[float]],
    top_k: int = 5,
) -> list[SimilarityMatch]:
    """Find candidate locations most similar to a target profile using standardized Euclidean distances."""
    target = np.asarray(target_profile, dtype=np.float64)
    candidates = np.asarray(candidate_features, dtype=np.float64)
    if candidates.ndim == 1:
        candidates = candidates[:, np.newaxis]

    # Standardize by candidate standard deviations
    stds = np.std(candidates, axis=0) + 1e-9
    norm_target = target / stds
    norm_cand = candidates / stds

    dists = np.sqrt(np.sum((norm_cand - norm_target) ** 2, axis=1))
    max_d = float(np.max(dists)) if len(dists) > 0 else 1.0

    sorted_idx = np.argsort(dists)[:top_k]
    matches: list[SimilarityMatch] = []

    for idx in sorted_idx:
        d = float(dists[idx])
        sim_score = max(0.0, 1.0 - (d / max(1e-9, max_d)))
        matches.append(
            SimilarityMatch(
                candidate_index=int(idx), similarity_score=round(sim_score, 4), distance=round(d, 4)
            )
        )

    return matches
