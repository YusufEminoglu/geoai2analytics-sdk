# -*- coding: utf-8 -*-
"""Unit tests for geoai2analytics Round 9 features (Spatial Surrogate & Point Process Intensity)."""

from __future__ import annotations

import unittest

from geoai2analytics import (
    PointProcessIntensityResult,
    SpatialEmbeddingMatrix,
    SpatialSurrogateResult,
    estimate_inhomogeneous_intensity,
    fit_spatial_autoregressive_surrogate,
)


class TestGeoAi2AnalyticsRound9(unittest.TestCase):
    def test_spatial_autoregressive_deep_surrogate(self) -> None:
        coords = [(x * 10.0, y * 10.0) for x in range(4) for y in range(4)]
        x = [[i * 1.5, (i % 3) * 2.0] for i in range(len(coords))]
        y = [10.0 + x[i][0] * 2.0 + (coords[i][0] + coords[i][1]) * 0.1 for i in range(len(coords))]

        res = fit_spatial_autoregressive_surrogate(x, y, coords, embedding_dim=3)

        self.assertIsInstance(res, SpatialSurrogateResult)
        self.assertGreater(res.r_squared, 0.0)
        self.assertEqual(res.geo_embeddings.embedding_dimension, 3)
        self.assertEqual(len(res.predicted_values), len(coords))

        d = res.to_dict()
        self.assertIn("r_squared", d)
        self.assertIn("spatial_lag_rho", d)

    def test_inhomogeneous_point_process_intensity(self) -> None:
        pts = [(100.0, 100.0), (120.0, 110.0), (105.0, 130.0), (500.0, 500.0), (520.0, 480.0)]
        res = estimate_inhomogeneous_intensity(pts, k_nearest_neighbors=3)

        self.assertIsInstance(res, PointProcessIntensityResult)
        self.assertEqual(res.total_events_count, 5)
        self.assertGreater(res.mean_intensity_per_km2, 0.0)
        self.assertGreater(res.max_hotspot_intensity_per_km2, 0.0)
        self.assertEqual(len(res.bandwidth_profiles), 5)

        d = res.to_dict()
        self.assertIn("events_count", d)
        self.assertIn("mean_intensity", d)
