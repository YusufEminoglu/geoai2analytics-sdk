# -*- coding: utf-8 -*-
"""Unit tests for geoai2analytics Rounds 2 and 3 features."""

from __future__ import annotations

import unittest
import numpy as np

from geoai2analytics import (
    EpidemicSpreadResult,
    GEVFitResult,
    HuffModelResult,
    NetKDEResult,
    ReillyResult,
    SpatialSEIRSimulator,
    WilsonFlowResult,
    huff_model,
    network_cross_k_function,
    network_kernel_density_estimation,
    reilly_law_breaking_point,
    spatial_gev_fit,
    spatial_return_period_map,
    wilson_spatial_interaction,
)


class TestGeoAI2AnalyticsRounds2And3(unittest.TestCase):
    def test_huff_and_gravity_interaction_models(self) -> None:
        origins = [(0.0, 0.0), (100.0, 100.0)]
        demands = [5000.0, 3000.0]
        destinations = [(50.0, 50.0), (200.0, 200.0)]
        attract = [10000.0, 25000.0]

        huff = huff_model(origins, demands, destinations, attract)
        self.assertIsInstance(huff, HuffModelResult)
        self.assertEqual(len(huff.expected_patrons_by_destination), 2)
        self.assertAlmostEqual(sum(huff.expected_patrons_by_destination), 8000.0, delta=10.0)

        # Reilly's Law
        reilly = reilly_law_breaking_point(population_a=100000.0, population_b=400000.0, distance_km=60.0)
        self.assertIsInstance(reilly, ReillyResult)
        self.assertEqual(reilly.distance_km, 60.0)
        self.assertAlmostEqual(reilly.breaking_point_from_a_km, 20.0, delta=0.5)

        # Wilson doubly-constrained flow
        cost_mat = [[5.0, 20.0], [25.0, 10.0]]
        w_flow = wilson_spatial_interaction(origins_supply=[100.0, 200.0], destinations_demand=[150.0, 150.0], cost_matrix=cost_mat)
        self.assertIsInstance(w_flow, WilsonFlowResult)
        self.assertAlmostEqual(w_flow.total_flow, 300.0, delta=1.0)

    def test_extreme_value_gev_return_period(self) -> None:
        series = [45.2, 58.1, 72.4, 61.3, 89.0, 54.2, 67.8, 95.3, 78.1, 84.5]
        fit = spatial_gev_fit(series)
        self.assertIsInstance(fit, GEVFitResult)
        self.assertGreater(fit.return_level_100yr, fit.return_level_10yr)

        stations = [(10.0, 10.0), (20.0, 20.0)]
        mat = [series, [s * 1.2 for s in series]]
        r_map = spatial_return_period_map(stations, mat, target_return_years=100.0)
        self.assertEqual(len(r_map), 2)
        self.assertIn("return_level", r_map[0])

    def test_network_kernel_density_estimation(self) -> None:
        events = [(10.0, 10.0), (15.0, 12.0), (120.0, 110.0)]
        segments = [(12.0, 11.0), (50.0, 50.0), (115.0, 108.0)]

        net_kde = network_kernel_density_estimation(events, segments, bandwidth=50.0)
        self.assertIsInstance(net_kde, NetKDEResult)
        self.assertEqual(len(net_kde.segment_densities), 3)
        self.assertGreater(net_kde.segment_densities[0], 0.0)
        self.assertEqual(net_kde.segment_densities[1], 0.0)  # Beyond bandwidth

        cross_k = network_cross_k_function(events[:2], events[2:], distances=[50.0, 200.0])
        self.assertEqual(cross_k[50.0], 0.0)
        self.assertEqual(cross_k[200.0], 1.0)

    def test_spatial_seir_simulator(self) -> None:
        pops = [50000, 30000, 20000]
        coords = [(0.0, 0.0), (10.0, 0.0), (0.0, 15.0)]
        sim = SpatialSEIRSimulator(pops, coords)
        res = sim.simulate(initial_infected_zone=0, initial_infected_count=20, days=30)

        self.assertIsInstance(res, EpidemicSpreadResult)
        self.assertEqual(res.timesteps, 30)
        self.assertGreater(res.peak_infectious_count, 0)
        self.assertGreater(res.final_attack_rate, 0.0)
