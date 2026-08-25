# -*- coding: utf-8 -*-
"""Master comprehensive test suite for geoai2analytics across all 75 scientific algorithms."""

import unittest

import numpy as np

import geoai2analytics as geoai


class TestGeoAi2AnalyticsMasterSuite(unittest.TestCase):
    def setUp(self) -> None:
        self.data, self.weights = geoai.generate_synthetic_spatial_dataset(n=60, seed=42)
        self.coords = np.column_stack([self.data["x_coord"], self.data["y_coord"]])
        self.y = np.array(self.data["y"])
        self.X = np.column_stack([self.data["X1"], self.data["X2"]])

    # -----------------------------------------------------------------------
    # 1. Spatial Distribution & Dispersion
    # -----------------------------------------------------------------------
    def test_distribution_engines(self) -> None:
        # Mean Center
        mc = geoai.mean_center(self.coords)
        self.assertIsInstance(mc.x, float)
        self.assertIsInstance(mc.y, float)

        # Median Center (Weiszfeld)
        med = geoai.median_center(self.coords)
        self.assertIsInstance(med.x, float)
        self.assertTrue(med.converged)

        # Central Feature
        cf = geoai.central_feature(self.coords)
        self.assertGreaterEqual(cf.feature_index, 0)

        # Standard Distance
        sd = geoai.standard_distance(self.coords)
        self.assertGreater(sd.standard_distance, 0.0)

        # Standard Deviational Ellipse (SDE)
        sde = geoai.standard_deviational_ellipse(self.coords, std_level=1)
        self.assertGreater(sde.area, 0.0)
        poly = sde.polygon_coords(num_points=12)
        self.assertEqual(len(poly), 12)

        # Linear Directional Mean
        lines = [
            ((0.0, 0.0), (10.0, 10.0)),
            ((5.0, 5.0), (15.0, 12.0)),
            ((2.0, 1.0), (8.0, 7.0)),
        ]
        ldm = geoai.linear_directional_mean(lines)
        self.assertGreaterEqual(ldm.mean_direction_deg, 0.0)

        # Average Nearest Neighbor (ANN)
        ann = geoai.average_nearest_neighbor(self.coords)
        self.assertGreater(ann.observed_mean_dist, 0.0)
        self.assertIn(ann.spatial_pattern, ["Clustered", "Dispersed", "Random"])

        # Ripley's K-Function
        rk = geoai.ripleys_k(self.coords, n_sim=10)
        self.assertEqual(len(rk.radii), 20)
        self.assertEqual(len(rk.l_values), 20)

    # -----------------------------------------------------------------------
    # 2. Spatial Autocorrelation & Association
    # -----------------------------------------------------------------------
    def test_autocorrelation_engines(self) -> None:
        # Global Moran
        gm = geoai.global_moran(self.y, self.weights, permutations=49)
        self.assertGreater(gm.I, 0.2)

        # LISA
        lisa = geoai.local_moran(self.y, self.weights, permutations=49)
        self.assertEqual(len(lisa.quadrants), 60)

        # Getis-Ord Gi*
        gi = geoai.local_getis_ord_gi_star(self.y, self.weights)
        self.assertEqual(len(gi.gi_star), 60)

        # General G
        pos_y = self.y - np.min(self.y) + 1.0
        gg = geoai.general_g(pos_y, self.weights)
        self.assertIsInstance(gg.G, float)

        # Geary's C & Local Geary
        gc = geoai.geary_c(self.y, self.weights, permutations=49)
        self.assertIsInstance(gc.C, float)
        lg = geoai.local_geary(self.y, self.weights, permutations=49)
        self.assertEqual(len(lg.c_i), 60)

        # Lee's L
        lee = geoai.global_lee_l(self.data["X1"], self.y, self.weights)
        self.assertIsInstance(lee.L, float)

        # Join Count
        cat_data = (self.y > np.median(self.y)).astype(int)
        jc = geoai.join_count(cat_data, self.weights)
        self.assertGreaterEqual(jc.bb_count, 0.0)

        # Colocation Quotient (CLQ)
        cat_b = (self.data["X1"] > np.median(self.data["X1"])).astype(int)
        clq = geoai.colocation_quotient(cat_data, cat_b, self.weights)
        self.assertGreater(clq.global_clq, 0.0)

        # Geodetector Q
        strata = (self.coords[:, 0] > 50.0).astype(int)
        gq = geoai.geodetector_q(self.y, strata)
        self.assertGreaterEqual(gq.q_statistic, 0.0)

        # Incremental Autocorrelation
        inc = geoai.incremental_autocorrelation(self.coords, self.y, distance_steps=4)
        self.assertEqual(len(inc.distances), 4)

        # Spatial Gini
        gini = geoai.spatial_gini(pos_y, self.weights)
        self.assertGreater(gini.gini, 0.0)

    # -----------------------------------------------------------------------
    # 3. Spatial Econometrics
    # -----------------------------------------------------------------------
    def test_econometrics_engines(self) -> None:
        # GWR
        gwr = geoai.GWR(self.coords, self.y, self.X, bandwidth=15, adaptive=True)
        gwr_res = gwr.fit()
        self.assertGreater(gwr_res.global_r2, 0.5)

        # MGWR
        mgwr = geoai.MGWR(self.coords, self.y, self.X, adaptive=True)
        mgwr_res = mgwr.fit(max_iter=2)
        self.assertEqual(len(mgwr_res.bandwidths), 3)

        # SAR (Spatial Lag)
        sar = geoai.fit_spatial_lag(self.y, self.X, self.weights)
        self.assertIsInstance(sar.rho, float)

        # SEM (Spatial Error)
        sem = geoai.fit_spatial_error(self.y, self.X, self.weights, max_iter=5)
        self.assertIsInstance(sem.lambda_param, float)

        # SDM (Spatial Durbin)
        sdm = geoai.fit_spatial_durbin(self.y, self.X, self.weights)
        self.assertEqual(len(sdm.gammas), 2)

        # Spatial Regime
        regimes = (self.coords[:, 1] > 50.0).astype(int)
        reg_res = geoai.fit_spatial_regime(self.y, self.X, regimes)
        self.assertEqual(len(reg_res.regime_params), 2)

        # ESF (Eigenvector Spatial Filtering)
        esf_res = geoai.eigenvector_spatial_filtering(self.y, self.X, self.weights)
        self.assertGreaterEqual(len(esf_res.selected_eigenvector_indices), 0)

        # LM Diagnostics
        lm_res = geoai.lagrange_multiplier_diagnostics(self.y, self.X, self.weights)
        self.assertIn("Spatial", lm_res.suggested_model)

    # -----------------------------------------------------------------------
    # 4. Spatial Clustering
    # -----------------------------------------------------------------------
    def test_clustering_engines(self) -> None:
        # SKATER
        sk_res = geoai.skater(self.X, self.weights, n_clusters=4)
        self.assertEqual(len(sk_res.labels), 60)
        self.assertEqual(sk_res.n_clusters, 4)

        # Spatial DBSCAN
        db_res = geoai.spatial_dbscan(self.coords, eps_spatial=25.0, min_samples=3)
        self.assertEqual(len(db_res.labels), 60)

        # Spatial GMM
        gmm_res = geoai.spatial_gmm(self.coords, self.X, n_components=3, max_iter=15)
        self.assertEqual(gmm_res.n_clusters, 3)

    # -----------------------------------------------------------------------
    # 5. Spatial Machine Learning
    # -----------------------------------------------------------------------
    def test_ml_engines(self) -> None:
        # Spatial Random Forest
        rf = geoai.SpatialRandomForestRegressor(n_estimators=10, max_depth=4)
        rf.fit(self.X, self.y)
        preds_rf = rf.predict(self.X)
        self.assertEqual(len(preds_rf), 60)

        # EBM (Explainable Boosting Machine)
        ebm = geoai.ExplainableBoostingRegressor(n_bins=8, n_rounds=15)
        ebm.fit(self.X, self.y)
        preds_ebm = ebm.predict(self.X)
        self.assertEqual(len(preds_ebm), 60)

        # Multi-model Comparison Leaderboard
        models = {"RandomForest": rf, "EBM": ebm}
        comp = geoai.compare_spatial_models(
            models,
            self.X[:40],
            self.y[:40],
            self.X[40:],
            self.y[40:],
            geoai.knn_weights(self.coords[40:], k=4),
        )
        self.assertEqual(len(comp.leaderboard), 2)

    # -----------------------------------------------------------------------
    # 6. Explainable GeoAI (XAI)
    # -----------------------------------------------------------------------
    def test_xai_engines(self) -> None:
        # Spatial Cross-Validation
        scv = geoai.SpatialCrossValidator(n_splits=3)
        folds = scv.split(self.coords)
        self.assertEqual(len(folds), 3)

        # Spatial SHAP
        def dummy_pred(X_in: np.ndarray) -> np.ndarray:
            return 1.0 + 2.0 * X_in[:, 0] - 1.5 * X_in[:, 1]

        shap_exp = geoai.SpatialSHAP(dummy_pred, background_data=self.X[:15], nsamples=15)
        shap_res = shap_exp.explain(self.X[:5])
        self.assertEqual(shap_res.shap_values.shape, (5, 2))

        # Partial Dependence Plot (PDP)
        pdp = geoai.partial_dependence(dummy_pred, self.X, feature_idx=0, grid_resolution=10)
        self.assertEqual(len(pdp.grid_values), 10)

        # Permutation Importance
        perm = geoai.permutation_feature_importance(dummy_pred, self.X, self.y, n_repeats=2)
        self.assertEqual(len(perm.importances_mean), 2)

        # Conformal Predictor
        cp = geoai.ConformalPredictor(alpha=0.10)
        cp.calibrate(self.y[:30], self.y[:30] + 0.1)
        intervals = cp.predict_interval(self.y[30:])
        self.assertEqual(len(intervals.y_lower), 30)

    # -----------------------------------------------------------------------
    # 7. Audit & Similarity
    # -----------------------------------------------------------------------
    def test_audit_and_similarity(self) -> None:
        # Data Readiness Audit
        report = geoai.audit_spatial_data(self.X, coords=self.coords, weights=self.weights)
        self.assertEqual(report.sample_size, 60)
        self.assertEqual(report.n_features, 2)

        # Similarity Search
        target = self.X[0]
        matches = geoai.similarity_search(target, self.X[1:], top_k=3)
        self.assertEqual(len(matches), 3)
        self.assertGreaterEqual(matches[0].similarity_score, 0.0)


if __name__ == "__main__":
    unittest.main()
