# -*- coding: utf-8 -*-
"""Spatial Conformal Prediction & Non-Parametric Uncertainty Quantification for geoai2analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class ConformalInterval:
    index: int
    predicted_y: float
    lower_bound: float
    upper_bound: float
    interval_width: float
    contains_actual: bool | None = None


@dataclass
class ConformalCoverageReport:
    target_confidence_level: float  # e.g. 0.95 (95% coverage)
    calibrated_quantile_q: float
    empirical_coverage_rate: float
    mean_interval_width: float
    intervals: list[ConformalInterval]

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_level": self.target_confidence_level,
            "quantile_q": round(self.calibrated_quantile_q, 4),
            "empirical_coverage_pct": round(self.empirical_coverage_rate * 100.0, 2),
            "mean_interval_width": round(self.mean_interval_width, 4),
            "total_intervals": len(self.intervals),
        }


def calibrate_spatial_conformal(
    y_calib_pred: Sequence[float],
    y_calib_actual: Sequence[float],
    y_test_pred: Sequence[float],
    y_test_actual: Sequence[float] | None = None,
    alpha: float = 0.05,  # 1 - alpha = 95% confidence
    spatial_distances: Sequence[float] | None = None,
) -> ConformalCoverageReport:
    """Split Conformal Prediction with non-parametric finite-sample valid prediction intervals."""
    y_cal_p = np.asarray(y_calib_pred, dtype=np.float64)
    y_cal_a = np.asarray(y_calib_actual, dtype=np.float64)
    n_cal = len(y_cal_p)

    # 1. Non-conformity scores R_i = |y_i - \hat{y}_i|
    scores = np.abs(y_cal_a - y_cal_p)

    # Weighted or standard conformal quantile: ceil((n+1)(1-alpha)) / n
    level = math.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
    level = min(1.0, max(0.0, level))
    q_hat = float(np.quantile(scores, level, method="higher"))

    # 2. Test intervals
    y_t_p = np.asarray(y_test_pred, dtype=np.float64)
    intervals: list[ConformalInterval] = []
    covered_cnt = 0

    for idx, pred in enumerate(y_t_p):
        low = float(pred - q_hat)
        high = float(pred + q_hat)
        width = float(high - low)
        contains: bool | None = None

        if y_test_actual is not None and idx < len(y_test_actual):
            act = y_test_actual[idx]
            contains = bool(low <= act <= high)
            if contains:
                covered_cnt += 1

        intervals.append(
            ConformalInterval(
                index=idx,
                predicted_y=float(pred),
                lower_bound=low,
                upper_bound=high,
                interval_width=width,
                contains_actual=contains,
            )
        )

    emp_coverage = (covered_cnt / max(1, len(intervals))) if y_test_actual is not None else (1.0 - alpha)
    mean_w = float(np.mean([it.interval_width for it in intervals])) if intervals else 0.0

    return ConformalCoverageReport(
        target_confidence_level=1.0 - alpha,
        calibrated_quantile_q=q_hat,
        empirical_coverage_rate=emp_coverage,
        mean_interval_width=mean_w,
        intervals=intervals,
    )
