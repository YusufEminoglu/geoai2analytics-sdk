# Changelog

All notable changes to **`geoai2analytics-sdk`** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-26

### Added
- **Spatial Autocorrelation Engines**:
  - Global Moran's I with analytical variance under randomization and Monte Carlo permutations ($999$ iterations).
  - Local Moran's I (LISA) decomposition with quadrant clusters (High-High, Low-Low, Low-High, High-Low).
  - Getis-Ord Local $G_i^*$ hotspot and coldspot analysis.
  - Geary's $C$ global/local spatial dissimilarity indices.
  - Bivariate Moran's I cross-spatial correlation.
  - Spatial Gini inequality and spatial disparity ratio.
- **Spatial Econometrics & Local Regressions**:
  - Geographically Weighted Regression (GWR) with Golden-Section search bandwidth optimization, Hurvich AICc, Gaussian/Bisquare/Exponential spatial kernels.
  - Multiscale GWR (MGWR) with backfitting Generalized Additive Model (GAM) for variable-specific spatial bandwidths.
  - Spatial Autoregressive (SAR / Spatial Lag) estimation via 2-Stage Least Squares (2SLS).
- **Explainable GeoAI (XAI)**:
  - Spatial Cross-Validation (Spatial $K$-Fold) coordinate clustering.
  - Spatial SHAP with continuous 2D coordinate map attributions.
  - Split-Conformal Spatial Uncertainty intervals.
- **Packaging & Infrastructure**:
  - Comprehensive unit test suite with 86.8% coverage.
  - Interactive academic documentation site with live Spatial Autocorrelation & LISA simulator sandbox.
  - Multi-platform CI testing matrix (Python 3.9 - 3.13).
  - PyPI Trusted Publishing via GitHub Actions.
