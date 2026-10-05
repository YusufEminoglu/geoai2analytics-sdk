# -*- coding: utf-8 -*-
"""Unit tests for geoai2analytics Round 5 features (GW-GLM Poisson & Anisotropic DBSCAN)."""

from __future__ import annotations

import unittest

import numpy as np

from geoai2analytics import (
    GWGLMResult,
    SpatialClusterResult,
    cluster_spatial_points_dbscan,
    fit_gw_poisson_regression,
)


class TestGeoAi2AnalyticsRound5(unittest.TestCase):
    def test_gw_poisson_regression(self) -> None:
        np.random.seed(42)
        coords = [(float(x), float(y)) for x in range(5) for y in range(4)]
        n = len(coords)
        x_feat = [[float(i * 0.1), float((i % 3) * 0.5)] for i in range(n)]
        y_counts = [int(1 + (c[0] + c[1]) * 0.8 + np.random.poisson(1)) for c in coords]

        res = fit_gw_poisson_regression(coords, x_feat, y_counts, bandwidth_distance=3.5)
        self.assertIsInstance(res, GWGLMResult)
        self.assertEqual(res.model_family, "poisson")
        self.assertEqual(res.n_observations, n)
        self.assertEqual(len(res.predicted_means), n)
        self.assertEqual(len(res.local_coefficients), n)
        self.assertGreater(res.aic_score, 0.0)

        d = res.to_dict()
        self.assertIn("aic_score", d)
        self.assertIn("residual_deviance", d)

    def test_anisotropic_spatial_dbscan(self) -> None:
        # Cluster 1: tight group around (10, 10)
        c1 = [(10.0 + dx, 10.0 + dy) for dx in (-1.0, 0.0, 1.0) for dy in (-1.0, 0.0, 1.0)]
        # Cluster 2: tight group around (100, 100)
        c2 = [(100.0 + dx, 100.0 + dy) for dx in (-1.0, 0.0, 1.0) for dy in (-1.0, 0.0, 1.0)]
        # Noise point
        noise = [(500.0, 500.0)]

        all_pts = c1 + c2 + noise
        res = cluster_spatial_points_dbscan(all_pts, eps_distance=5.0, min_samples=4)

        self.assertIsInstance(res, SpatialClusterResult)
        self.assertEqual(res.total_points, len(all_pts))
        self.assertEqual(res.num_clusters, 2)
        self.assertEqual(res.noise_points_count, 1)
        self.assertEqual(res.cluster_labels[-1], -1)

        d = res.to_dict()
        self.assertIn("num_clusters", d)
        self.assertIn("noise_points", d)
