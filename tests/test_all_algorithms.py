# -*- coding: utf-8 -*-
"""Comprehensive test suite for geoai2analytics."""

import unittest

import numpy as np

from geoai2analytics import (
    GWR,
    MGWR,
    ConformalPredictor,
    SpatialCrossValidator,
    SpatialSHAP,
    bivariate_moran,
    distance_band_weights,
    fit_spatial_lag,
    generate_synthetic_spatial_dataset,
    global_moran,
    grid_weights,
    knn_weights,
    local_getis_ord_gi_star,
    local_moran,
    spatial_gini,
)
from geoai2analytics.cli import main as cli_main


class TestGeoAi2Analytics(unittest.TestCase):
    def setUp(self) -> None:
        self.data, self.weights = generate_synthetic_spatial_dataset(n=60, seed=42)
        self.coords = np.column_stack([self.data["x_coord"], self.data["y_coord"]])
        self.y = np.array(self.data["y"])
        self.X = np.column_stack([self.data["X1"], self.data["X2"]])

    def test_spatial_weights(self) -> None:
        # KNN
        w_knn = knn_weights(self.coords, k=4)
        self.assertEqual(w_knn.n, 60)
        self.assertTrue(w_knn.is_row_standardized)
        self.assertAlmostEqual(float(np.sum(w_knn.matrix[0])), 1.0, places=5)

        # Spatial Lag
        lag_y = w_knn.lag(self.y)
        self.assertEqual(len(lag_y), 60)

        # Grid weights
        w_grid = grid_weights(5, 5, contiguity="queen")
        self.assertEqual(w_grid.n, 25)

        # Distance band weights
        w_dist = distance_band_weights(self.coords, threshold=30.0, kernel="gaussian")
        self.assertEqual(w_dist.n, 60)

    def test_global_and_local_moran(self) -> None:
        res = global_moran(self.y, self.weights, permutations=99)
        self.assertIsInstance(res.I, float)
        self.assertGreater(res.I, 0.2)  # Positive spatial autocorrelation
        self.assertLess(res.p_sim, 0.05)
        self.assertEqual(res.permutations, 99)

        summary = res.summary()
        self.assertIn("morans_i", summary)
        self.assertEqual(summary["spatial_pattern"], "Clustered (Positive Autocorrelation)")

        # Local Moran (LISA)
        lisa = local_moran(self.y, self.weights, permutations=99)
        self.assertEqual(len(lisa.quadrants), 60)
        self.assertGreater(lisa.high_high_count + lisa.low_low_count, 0)
        lisa_sum = lisa.summary()
        self.assertEqual(lisa_sum["total_units"], 60)

    def test_getis_ord_and_spatial_gini(self) -> None:
        # Getis-Ord Gi*
        gi_res = local_getis_ord_gi_star(self.y, self.weights)
        self.assertEqual(len(gi_res.gi_star), 60)
        self.assertEqual(len(gi_res.cluster_bins), 60)

        # Spatial Gini
        gini_res = spatial_gini(self.y, self.weights)
        self.assertGreater(gini_res.gini, 0.0)
        self.assertGreater(gini_res.spatial_gini, 0.0)

        # Bivariate Moran
        biv_res = bivariate_moran(self.data["X1"], self.y, self.weights, permutations=99)
        self.assertIsInstance(biv_res.I_xy, float)

    def test_gwr_and_mgwr(self) -> None:
        # GWR with fixed bandwidth
        gwr_model = GWR(self.coords, self.y, self.X, bandwidth=20, adaptive=True)
        gwr_res = gwr_model.fit()

        self.assertEqual(gwr_res.params.shape, (60, 3))
        self.assertEqual(len(gwr_res.local_r2), 60)
        self.assertGreater(gwr_res.global_r2, 0.5)

        summary = gwr_res.summary()
        self.assertEqual(summary["sample_size"], 60)

        # MGWR
        mgwr_model = MGWR(self.coords, self.y, self.X, adaptive=True)
        mgwr_res = mgwr_model.fit(max_iter=3)
        self.assertEqual(mgwr_res.params.shape, (60, 3))
        self.assertEqual(len(mgwr_res.bandwidths), 3)

    def test_sar_spatial_lag(self) -> None:
        sar_res = fit_spatial_lag(self.y, self.X, self.weights)
        self.assertIsInstance(sar_res.rho, float)
        self.assertGreater(sar_res.r2, 0.4)
        self.assertEqual(len(sar_res.betas), 3)

    def test_spatial_cv_and_shap(self) -> None:
        # Spatial K-Fold
        scv = SpatialCrossValidator(n_splits=4, seed=42)
        folds = scv.split(self.coords)
        self.assertEqual(len(folds), 4)
        for f in folds:
            self.assertGreater(len(f.train_indices), 0)
            self.assertGreater(len(f.test_indices), 0)

        # Spatial SHAP
        def dummy_predict(X_in: np.ndarray) -> np.ndarray:
            return 2.0 + 1.5 * X_in[:, 0] - 2.5 * X_in[:, 1]

        shap_explainer = SpatialSHAP(
            predict_fn=dummy_predict,
            background_data=self.X[:20],
            feature_names=["X1", "X2"],
            nsamples=20,
        )
        shap_res = shap_explainer.explain(self.X[:10])
        self.assertEqual(shap_res.shap_values.shape, (10, 2))
        self.assertIn("X1", shap_res.global_importance)

    def test_conformal_prediction(self) -> None:
        cp = ConformalPredictor(alpha=0.10)
        q_hat = cp.calibrate(self.y[:40], self.y[:40] + np.random.normal(0, 0.5, 40))
        self.assertGreater(q_hat, 0.0)

        intervals = cp.predict_interval(self.y[40:])
        self.assertEqual(len(intervals.y_lower), 20)
        self.assertEqual(len(intervals.y_upper), 20)
        self.assertTrue(np.all(intervals.y_upper >= intervals.y_lower))

    def test_cli(self) -> None:
        res1 = cli_main(["info"])
        self.assertEqual(res1, 0)

        res2 = cli_main(["test-moran"])
        self.assertEqual(res2, 0)


if __name__ == "__main__":
    unittest.main()
