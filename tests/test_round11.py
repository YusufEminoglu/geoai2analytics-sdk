# -*- coding: utf-8 -*-
"""Unit tests for geoai2analytics Round 11 features (ST Kriging & Spatial Density Anomaly Isolation)."""

from __future__ import annotations

import unittest

from geoai2analytics import (
    STKrigingResult,
    STVariogramParams,
    SpatialAnomalyReport,
    SpatialIsolationForestConfig,
    detect_spatial_density_anomalies,
    interpolate_spatiotemporal_kriging,
)


class TestGeoAi2AnalyticsRound11(unittest.TestCase):
    def test_spatiotemporal_kriging(self) -> None:
        samples = [
            (0.0, 0.0, 0.0, 15.0),
            (100.0, 0.0, 1.0, 18.0),
            (0.0, 100.0, 2.0, 20.0),
            (100.0, 100.0, 3.0, 24.0),
        ]
        target = (50.0, 50.0, 1.5)
        params = STVariogramParams(spatial_range_m=1000.0, temporal_range_hours=12.0)

        res = interpolate_spatiotemporal_kriging(samples, target, params=params)

        self.assertIsInstance(res, STKrigingResult)
        self.assertGreater(res.predicted_value, 10.0)
        self.assertGreater(res.kriging_estimation_variance, 0.0)
        self.assertEqual(res.num_spatiotemporal_neighbors_used, 4)

        d = res.to_dict()
        self.assertIn("predicted_val", d)
        self.assertIn("variance", d)

    def test_spatial_density_anomaly_isolation(self) -> None:
        # Cluster of normal points + 1 isolated distant outlier
        pts = [(10.0 + i * 2, 10.0 + i * 2) for i in range(15)] + [(5000.0, 5000.0)]
        cfg = SpatialIsolationForestConfig(contamination_fraction=0.10, kernel_bandwidth_m=200.0)

        report = detect_spatial_density_anomalies(pts, config=cfg)

        self.assertIsInstance(report, SpatialAnomalyReport)
        self.assertEqual(report.total_samples_evaluated, 16)
        self.assertGreater(report.anomalies_detected_count, 0)
        self.assertIn(15, report.outlier_point_indices)  # The 16th point (index 15) is outlier

        d = report.to_dict()
        self.assertIn("evaluated_samples", d)
        self.assertIn("outliers_count", d)
