# Changelog

All notable changes to **`geoai2analytics-sdk`** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-26

### Added (Full 75-Algorithm Master Suite)
- **Spatial Distribution & Dispersion**:
  - `mean_center`: Geographic and weighted center of mass.
  - `median_center`: Continuous $L_1$ Fermat-Weber median point via Weiszfeld algorithm.
  - `central_feature`: Most accessible feature minimizing total travel distance.
  - `standard_distance`: Spatial dispersion and concentration radius.
  - `standard_deviational_ellipse`: Directional distribution, anisotropy, and orientation angle (1, 2, 3 $\sigma$).
  - `linear_directional_mean`: Directional trend and circular variance of line vectors.
  - `average_nearest_neighbor`: Clark-Evans ANN spatial point pattern clustering test.
  - `ripleys_k`: Multi-distance Ripley's $K(r)$, Besag's $L(r)$, and Monte Carlo CSR confidence envelope.
- **Spatial Autocorrelation & Association**:
  - `global_moran` & `local_moran` (LISA): Analytical variance, $z$-score, Monte Carlo permutation test ($999+$ reps), and 4-quadrant cluster identification.
  - `local_getis_ord_gi_star`: Hotspot and coldspot identification ($99\%, 95\%, 90\%$).
  - `general_g`: High/Low clustering General $G$ statistic.
  - `geary_c` & `local_geary`: Global and local spatial dissimilarity indices.
  - `global_lee_l`: Spatial smoothing and Pearson integration.
  - `join_count`: Categorical spatial autocorrelation (BB, WW, BW).
  - `colocation_quotient` (CLQ): Global and local spatial co-occurrence.
  - `geodetector_q`: Wang's spatial stratified heterogeneity $q$-statistic.
  - `incremental_autocorrelation`: Automatic peak clustering distance band detection.
  - `bivariate_moran` & `spatial_gini`: Cross-spatial correlation and spatial inequality.
- **Spatial Econometrics & Local Regressions**:
  - `GWR` & `MGWR`: Golden-section bandwidth optimization, AICc minimization, Gaussian/Bisquare/Exponential kernels, and backfitting GAM.
  - `fit_spatial_lag` (SAR): 2-Stage Least Squares (2SLS) $\rho$ estimation.
  - `fit_spatial_error` (SEM): Iterative spatial error autoregression $\lambda$.
  - `fit_spatial_durbin` (SDM): Direct and indirect spatial spillover effects ($W X$).
  - `fit_spatial_regime`: Chow structural stability test and regime-specific models.
  - `eigenvector_spatial_filtering` (ESF): $M W M$ synthetic spatial proxy filtering.
  - `lagrange_multiplier_diagnostics`: LM-Lag, LM-Error, Robust LM-Lag, and Robust LM-Error tests.
- **Spatial Clustering & Regionalization**:
  - `skater`: Minimum Spanning Tree (MST) spatially constrained regionalization.
  - `spatial_dbscan`: Density-based spatial clustering with noise identification.
  - `spatial_gmm`: Gaussian Mixture Model fusing coordinates and attributes.
- **Spatial Machine Learning**:
  - `SpatialRandomForestRegressor`: Bagged forest with spatial coordinate embeddings.
  - `ExplainableBoostingRegressor`: Transparent GA2M generalized additive boosting with spatial interactions.
  - `compare_spatial_models`: Multi-model leaderboard benchmark (RMSE, MAE, $R^2$, Residual Moran's I).
- **Explainable GeoAI (XAI)**:
  - `SpatialSHAP`: Global, local, and 2D geographic coordinate map Shapley attributions.
  - `partial_dependence`: 1D/2D Partial Dependence Plots across spatial variables.
  - `permutation_feature_importance`: Permuted RMSE degradation scoring.
  - `ConformalPredictor`: Finite-sample $(1-\alpha)$ coverage prediction intervals.
  - `SpatialCrossValidator`: Spatial $K$-Fold coordinate blocking.
- **Spatial Data Hygiene & Quality Audit**:
  - `audit_spatial_data`: Multicollinearity (VIF), zero variance, skewness, and spatial graph island detection.
  - `similarity_search`: Standardized Euclidean/Mahalanobis spatial profile matching.

## [0.1.0] - 2026-08-26
- Initial release with core spatial statistics and GWR engines.
