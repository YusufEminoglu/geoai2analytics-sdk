# Changelog

All notable changes to this project will be documented in this file.

## [0.10.0] - 2026-08-30
### Added
- **Spatial Autoregressive Deep Surrogate & Geo-Embeddings (`spatial_lag_deep_surrogate.py`)**: Added `fit_spatial_autoregressive_surrogate` combining spatial lag weight matrices with deep neural embeddings.
- **Inhomogeneous Poisson Point Process Adaptive Intensity (`point_process_k_nearest_intensity.py`)**: Added `estimate_inhomogeneous_intensity` computing variable-bandwidth adaptive point pattern intensities.

## [0.9.0] - 2026-08-30
### Added
- **Hierarchical Density-Based Spatial Clustering (`spatial_hdbscan_clustering.py`)**: Added `cluster_spatial_hdbscan` using Mutual Reachability Distance graphs and Minimum Spanning Trees without arbitrary radius parameters.
- **Multi-Scale Spatial Information Entropy (`multiscale_spatial_entropy.py`)**: Added `calculate_spatial_information_entropy` evaluating Batty & Shannon spatial disorder across hierarchical grid scales.

## [0.8.0] - 2026-08-30
### Added
- **Bivariate Ripley's K Point Pattern Cross-Function (`ripleys_k_cross_function.py`)**: Added `calculate_ripleys_k_bivariate` and Besag L transform testing spatial co-location attraction vs repulsion.
- **Continuous Gravity Accessibility Potential Engine (`accessibility_isochrone_gravity.py`)**: Added `compute_gravity_accessibility_matrix` supporting Exponential, Gaussian, and Power distance decay models.

## [0.7.0] - 2026-08-30
### Added
- **Spatial Hedonic Real Estate Valuation Engine (`spatial_hedonic_pricing.py`)**: Added `fit_spatial_hedonic_model` modeling SAR spatial autocorrelation, structural attributes, and amenity elasticities.
- **SLEUTH-style Urban Sprawl Growth Cellular Automata (`cellular_automata_growth.py`)**: Added `simulate_urban_growth_ca` simulating diffusion, breed, spread, and road-influenced urban expansion.

## [0.6.0] - 2026-08-30
### Added
- **Geographically Weighted GLM (GW-GLM Poisson & Logistic) (`geographically_weighted_glm.py`)**: Added `fit_gw_poisson_regression` with spatial distance kernel weighting, local McFadden pseudo-$R^2$, and AICc optimization.
- **Directional Anisotropic Spatial DBSCAN (`spatial_density_dbscan.py`)**: Added `cluster_spatial_points_dbscan` supporting directional elliptical spatial density distance metrics.

## [0.5.0] - 2026-08-30
### Added
- **Spatial Markov Chains & Cellular Transition Engine (`spatial_markov.py`)**: Added `simulate_landuse_transition` with spatial lag neighborhood conditional transition matrices, ergodic steady state distributions, and chi-square spatial independence tests.
- **Spatial Conformal Prediction & Uncertainty Quantification (`spatial_conformal.py`)**: Added `calibrate_spatial_conformal` producing distribution-free finite-sample valid prediction intervals.

## [0.4.0] - 2026-08-30
### Added
- **Spatial Interaction & Gravity Model Suite (`spatial_interaction.py`)**: Added `huff_model`, `reilly_law_breaking_point`, and `wilson_spatial_interaction` (doubly-constrained entropy-maximizing matrix).
- **Spatial Extreme Value & Return Period Risk Engine (`extreme_value.py`)**: Added L-moment `spatial_gev_fit` and `spatial_return_period_map` (10, 50, 100, 500-year return level estimation).
- **Network-Constrained Spatial Statistics (`network_stats.py`)**: Added `network_kernel_density_estimation` (Equal Split continuous NetKDE) and `network_cross_k_function`.
- **Spatially Explicit SEIR Epidemic & Hazard Diffusion Simulator (`spatial_abm.py`)**: Added multi-zone agent-based SEIR model (`SpatialSEIRSimulator`).

## [0.3.0] - 2026-08-30

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Added
- **Spatio-Temporal Autocorrelation & Space-Time Emerging Hotspots (`spacetime.py`)**:
  - `SpaceTimeCube`: Multi-period 3D space-time grid container.
  - `emerging_hotspot_analysis`: Full ESRI-compatible space-time pattern classification (New Hotspot, Consecutive Hotspot, Intensifying Hotspot, Persistent Hotspot, Sporadic Hotspot, Oscillating Hotspot, Diminishing Hotspot, Historical Hotspot) using local Getis-Ord Gi* across time slices and the Mann-Kendall non-parametric trend test (`mann_kendall_test`) with Sen's slope estimator.
  - `spatiotemporal_moran`: Space-time joint autocorrelation estimator with space-time lag matrix product.
- **Spatial Causal Inference & Propensity Score Matching (`causal.py`)**:
  - `spatial_propensity_score_matching` (SPSM): Propensity score estimation combined with spatial distance caliper penalties to eliminate geographic selection bias, computing ATT and covariate balance improvements.
  - `spatial_difference_in_differences` (SDID): Spatial DiD with autoregressive spatial spillover lag ($W \Delta y$), computing direct treatment effect, indirect spillover effect, and robust standard errors.

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
