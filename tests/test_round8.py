# -*- coding: utf-8 -*-
"""Unit tests for geoai2analytics Round 8 features (Spatial HDBSCAN & Multi-Scale Entropy)."""

from __future__ import annotations

import unittest

from geoai2analytics import (
    HDBSCANResult,
    SpatialEntropyReport,
    calculate_spatial_information_entropy,
    cluster_spatial_hdbscan,
)


class TestGeoAi2AnalyticsRound8(unittest.TestCase):
    def test_spatial_hdbscan_clustering(self) -> None:
        # Two tight clusters + noise
        c1 = [(10.0 + i * 0.1, 10.0 + i * 0.1) for i in range(8)]
        c2 = [(100.0 + i * 0.1, 100.0 + i * 0.1) for i in range(8)]
        noise = [(50.0, 50.0), (0.0, 100.0)]
        pts = c1 + c2 + noise

        res = cluster_spatial_hdbscan(pts, min_cluster_size=5)

        self.assertIsInstance(res, HDBSCANResult)
        self.assertEqual(res.total_points_count, len(pts))
        self.assertGreaterEqual(res.num_clusters, 1)
        self.assertEqual(len(res.cluster_labels), len(pts))

        d = res.to_dict()
        self.assertIn("total_points", d)
        self.assertIn("clusters_count", d)

    def test_multiscale_spatial_entropy(self) -> None:
        # Clustered points (low entropy) vs uniform grid
        pts = [(x * 50.0, y * 50.0) for x in range(6) for y in range(6)]
        rep = calculate_spatial_information_entropy(pts, scale_grid_resolutions_m=(50.0, 100.0, 200.0))

        self.assertIsInstance(rep, SpatialEntropyReport)
        self.assertEqual(rep.total_features_count, 36)
        self.assertEqual(len(rep.scale_profiles), 3)
        self.assertGreater(rep.mean_normalized_entropy, 0.0)

        d = rep.to_dict()
        self.assertIn("total_features", d)
        self.assertIn("mean_entropy", d)
        self.assertIn("scales_evaluated", d)
