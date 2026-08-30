# -*- coding: utf-8 -*-
"""Unit tests for geoai2analytics Round 4 features (Spatial Markov & Spatial Conformal)."""

from __future__ import annotations

import unittest

from geoai2analytics import (
    ConformalCoverageReport,
    SpatialMarkovResult,
    calibrate_spatial_conformal,
    simulate_landuse_transition,
)


class TestGeoAi2AnalyticsRound4(unittest.TestCase):
    def test_spatial_markov_chains(self) -> None:
        t0 = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]
        t1 = [0, 1, 1, 2, 2, 2, 0, 1, 1, 0]
        lags = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]

        res = simulate_landuse_transition(t0, t1, spatial_lag_states_t0=lags, class_names=["Forest", "Urban", "Agri"])
        self.assertIsInstance(res, SpatialMarkovResult)
        self.assertEqual(res.num_classes, 3)
        self.assertEqual(len(res.global_transition_matrix), 3)
        self.assertEqual(len(res.steady_state_distribution), 3)
        self.assertAlmostEqual(sum(res.steady_state_distribution), 1.0, delta=0.05)

        d = res.to_dict()
        self.assertIn("global_matrix", d)
        self.assertIn("steady_state", d)

    def test_spatial_conformal_prediction(self) -> None:
        y_cal_pred = [10.0, 20.0, 30.0, 40.0, 50.0]
        y_cal_act = [10.5, 19.2, 31.0, 39.5, 50.8]  # Errors around 0.5 - 1.0

        y_test_pred = [15.0, 25.0, 35.0]
        y_test_act = [15.2, 24.8, 35.4]

        report = calibrate_spatial_conformal(
            y_cal_pred,
            y_cal_act,
            y_test_pred,
            y_test_actual=y_test_act,
            alpha=0.10,
        )

        self.assertIsInstance(report, ConformalCoverageReport)
        self.assertGreaterEqual(report.empirical_coverage_rate, 0.80)
        self.assertGreater(report.calibrated_quantile_q, 0.0)
        self.assertEqual(len(report.intervals), 3)
        self.assertTrue(all(it.contains_actual for it in report.intervals))

        d = report.to_dict()
        self.assertIn("empirical_coverage_pct", d)
