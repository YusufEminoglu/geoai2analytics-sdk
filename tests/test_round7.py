# -*- coding: utf-8 -*-
"""Unit tests for geoai2analytics Round 7 features (Bivariate Ripley's K & Gravity Accessibility)."""

from __future__ import annotations

import unittest

from geoai2analytics import (
    DistanceDecayType,
    GravityAccessibilityReport,
    RipleysKCrossResult,
    calculate_ripleys_k_bivariate,
    compute_gravity_accessibility_matrix,
)


class TestGeoAi2AnalyticsRound7(unittest.TestCase):
    def test_bivariate_ripleys_k(self) -> None:
        # Colocated clusters
        pts_a = [(10.0, 10.0), (12.0, 11.0), (100.0, 100.0)]
        pts_b = [(10.5, 10.5), (11.8, 11.2), (101.0, 99.0)]

        res = calculate_ripleys_k_bivariate(pts_a, pts_b, study_area_m2=10000.0, max_radius_r_m=50.0, radius_steps_count=5)

        self.assertIsInstance(res, RipleysKCrossResult)
        self.assertEqual(len(res.radius_distances_r), 5)
        self.assertEqual(len(res.observed_k_cross_values), 5)
        self.assertEqual(len(res.besag_l_cross_values), 5)

        d = res.to_dict()
        self.assertIn("radii_count", d)
        self.assertIn("attraction", d)

    def test_gravity_accessibility_matrix(self) -> None:
        origins = [(0.0, 0.0), (10.0, 0.0), (50.0, 0.0)]
        facilities = [(0.0, 1.0), (10.0, 1.0)]
        capacities = [100.0, 50.0]

        rep = compute_gravity_accessibility_matrix(
            origin_coordinates=origins,
            facility_coordinates=facilities,
            facility_capacities=capacities,
            decay_type=DistanceDecayType.EXPONENTIAL,
            decay_parameter_beta=0.01,
        )

        self.assertIsInstance(rep, GravityAccessibilityReport)
        self.assertEqual(rep.total_origins_analyzed, 3)
        self.assertGreater(rep.mean_accessibility_score, 0.0)
        self.assertGreaterEqual(rep.equity_gini_coefficient, 0.0)
        self.assertLessEqual(rep.equity_gini_coefficient, 1.0)
        self.assertEqual(len(rep.origin_accessibility_scores), 3)

        d = rep.to_dict()
        self.assertIn("mean_accessibility", d)
        self.assertIn("gini_index", d)
