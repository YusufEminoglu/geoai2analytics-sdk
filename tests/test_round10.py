# -*- coding: utf-8 -*-
"""Unit tests for geoai2analytics Round 10 features (ST Diffusion & Spatial Causal SHAP)."""

from __future__ import annotations

import unittest

from geoai2analytics import (
    SpatialShapleyReport,
    STDiffusionResult,
    explain_spatial_counterfactual_shap,
    predict_spatiotemporal_flow,
)


class TestGeoAi2AnalyticsRound10(unittest.TestCase):
    def test_spatiotemporal_graph_diffusion(self) -> None:
        coords = [(0.0, 0.0), (500.0, 0.0), (1000.0, 0.0)]
        signals = [
            [25.0, 30.0, 35.0],
            [28.0, 32.0, 38.0],
            [30.0, 35.0, 42.0],
        ]

        res = predict_spatiotemporal_flow(signals, coords, forecast_horizon_steps=3)

        self.assertIsInstance(res, STDiffusionResult)
        self.assertEqual(res.num_sensor_nodes, 3)
        self.assertEqual(res.forecast_horizon_steps, 3)
        self.assertEqual(len(res.forecasted_matrix), 3)
        self.assertEqual(len(res.forecasted_matrix[0]), 3)

        d = res.to_dict()
        self.assertIn("nodes", d)
        self.assertIn("horizon_steps", d)
        self.assertIn("mae", d)

    def test_spatial_causal_counterfactual_shap(self) -> None:
        feats = [12.5, 4.2, 0.85]
        names = ["density", "accessibility", "air_quality"]

        report = explain_spatial_counterfactual_shap(
            feature_values=feats,
            feature_names=names,
            target_prediction=78.5,
            base_expected_value=50.0,
            spatial_distance_to_core_m=350.0,
        )

        self.assertIsInstance(report, SpatialShapleyReport)
        self.assertEqual(len(report.feature_attributions), 3)
        self.assertGreater(report.minimal_counterfactual_distance_m, 0.0)
        self.assertIn("transit hub", report.recommended_spatial_intervention)

        d = report.to_dict()
        self.assertIn("prediction", d)
        self.assertIn("spatial_shap", d)
        self.assertIn("counterfactual_dist_m", d)
