# -*- coding: utf-8 -*-
"""Unit tests for geoai2analytics Round 6 features (Spatial Hedonic Pricing & Urban Sprawl CA)."""

from __future__ import annotations

import unittest

from geoai2analytics import (
    HedonicValuationResult,
    UrbanCAGrowthResult,
    fit_spatial_hedonic_model,
    simulate_urban_growth_ca,
)


class TestGeoAi2AnalyticsRound6(unittest.TestCase):
    def test_spatial_hedonic_pricing_model(self) -> None:
        coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 2.0)]
        props = [
            {"area_m2": 100.0, "age_years": 5.0, "dist_cbd_km": 1.2},
            {"area_m2": 150.0, "age_years": 2.0, "dist_cbd_km": 1.5},
            {"area_m2": 80.0, "age_years": 15.0, "dist_cbd_km": 3.0},
            {"area_m2": 120.0, "age_years": 1.0, "dist_cbd_km": 2.1},
            {"area_m2": 200.0, "age_years": 8.0, "dist_cbd_km": 4.5},
        ]
        prices = [350000.0, 520000.0, 220000.0, 410000.0, 600000.0]

        res = fit_spatial_hedonic_model(coords, props, prices, spatial_lag_rho=0.3)

        self.assertIsInstance(res, HedonicValuationResult)
        self.assertGreaterEqual(res.r_squared, 0.0)
        self.assertIn("area_m2", res.coefficients)
        self.assertIn("dist_cbd_km", res.coefficients)
        self.assertEqual(len(res.predicted_prices), 5)
        self.assertGreater(res.spatial_spillover_multiplier, 1.0)

        d = res.to_dict()
        self.assertIn("r_squared", d)
        self.assertIn("spatial_spillover_multiplier", d)

    def test_urban_sprawl_ca_simulation(self) -> None:
        init_grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        ca_res = simulate_urban_growth_ca(
            init_grid,
            spread_coefficient=0.8,
            steps=3,
        )

        self.assertIsInstance(ca_res, UrbanCAGrowthResult)
        self.assertEqual(ca_res.initial_urban_cells, 4)
        self.assertGreaterEqual(ca_res.final_urban_cells, 4)
        self.assertEqual(len(ca_res.expansion_history), 4)

        d = ca_res.to_dict()
        self.assertIn("growth_rate_pct", d)
        self.assertIn("final_urban_cells", d)
