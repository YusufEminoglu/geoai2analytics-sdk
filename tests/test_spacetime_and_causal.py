# -*- coding: utf-8 -*-
"""Unit tests for Spatio-Temporal and Spatial Causal Inference modules in geoai2analytics."""

from __future__ import annotations

import unittest

import numpy as np

from geoai2analytics import (
    SpaceTimeCube,
    emerging_hotspot_analysis,
    knn_weights,
    mann_kendall_test,
    spatial_difference_in_differences,
    spatial_propensity_score_matching,
    spatiotemporal_moran,
)


class TestSpaceTimeAndCausal(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        self.n_locs = 30
        self.n_time = 8
        self.coords = np.random.uniform(0.0, 100.0, size=(self.n_locs, 2))
        self.weights = knn_weights(self.coords, k=4)

        # Generate spatio-temporal matrix with increasing trend for first 5 units
        self.st_matrix = np.random.normal(10.0, 2.0, size=(self.n_locs, self.n_time))
        for t in range(self.n_time):
            self.st_matrix[:5, t] += (t + 1) * 3.0

        self.cube = SpaceTimeCube(
            coords=self.coords,
            time_series_matrix=self.st_matrix,
            time_labels=[f"Year_{2018+t}" for t in range(self.n_time)],
        )

    def test_mann_kendall_test(self) -> None:
        increasing_series = [1.0, 2.5, 3.2, 4.8, 6.0, 7.5, 9.1]
        res = mann_kendall_test(increasing_series)
        self.assertEqual(res.trend, "increasing")
        self.assertGreater(res.sen_slope, 0.0)
        self.assertLess(res.p_value, 0.05)

        flat_series = [5.0, 5.1, 4.9, 5.0, 5.2, 4.8]
        res_flat = mann_kendall_test(flat_series)
        self.assertEqual(res_flat.trend, "no trend")

    def test_spatiotemporal_moran_and_emerging_hotspots(self) -> None:
        st_moran = spatiotemporal_moran(self.cube, self.weights)
        self.assertIsInstance(st_moran, float)

        esta_res = emerging_hotspot_analysis(self.cube, self.weights)
        self.assertEqual(esta_res.n_locations, self.n_locs)
        self.assertEqual(esta_res.n_time_steps, self.n_time)
        self.assertIsInstance(esta_res.pattern_summary, dict)

        d = esta_res.to_dict()
        self.assertIn("pattern_summary", d)
        self.assertEqual(len(d["patterns"]), self.n_locs)

    def test_spatial_propensity_score_matching(self) -> None:
        N = 50
        X = np.random.uniform(10, 50, size=(N, 3))
        coords = np.random.uniform(0, 100, size=(N, 2))

        # Binary treatment based on first covariate + noise
        prob = 1.0 / (1.0 + np.exp(-(0.1 * X[:, 0] - 3.0)))
        treatment = (np.random.uniform(0, 1, size=N) < prob).astype(int)
        # Ensure at least 5 in each group
        treatment[:10] = 1
        treatment[10:20] = 0

        # Outcome with positive true treatment effect = 5.0
        outcome = 2.0 * X[:, 0] + 5.0 * treatment + np.random.normal(0, 1, size=N)

        res = spatial_propensity_score_matching(
            treatment=treatment,
            outcome=outcome,
            covariates=X,
            coords=coords,
            spatial_penalty_weight=0.2,
        )

        self.assertGreater(len(res.matched_pairs), 0)
        self.assertIsInstance(res.att, float)
        d = res.to_dict()
        self.assertIn("att", d)
        self.assertIn("covariate_balance_after", d)

    def test_spatial_difference_in_differences(self) -> None:
        N = 40
        coords = np.random.uniform(0, 100, size=(N, 2))
        w = knn_weights(coords, k=4)

        t = np.zeros(N, dtype=int)
        t[:15] = 1

        y0 = np.random.normal(50, 5, size=N)
        y1 = y0 + 10.0 * t + np.random.normal(2, 1, size=N)

        sdid_res = spatial_difference_in_differences(
            y_pre=y0,
            y_post=y1,
            treatment=t,
            spatial_weights=w,
        )

        self.assertIsInstance(sdid_res.att_direct, float)
        self.assertGreater(sdid_res.att_direct, 5.0)
        d = sdid_res.to_dict()
        self.assertIn("att_direct", d)
        self.assertIn("rho_spatial_spillover", d)
