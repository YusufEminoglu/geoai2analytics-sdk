<p align="center">
  <img src="https://geophilo.com/assets/sdk_assets/geoai-hero.svg" alt="geoai2analytics: Spatial Autocorrelation, GWR Econometrics, and Explainable GeoAI" width="100%">
</p>

# geoai2analytics-sdk

<div align="center">


[![PyPI version](https://img.shields.io/pypi/v/geoai2analytics-sdk.svg?color=10b981)](https://pypi.org/project/geoai2analytics-sdk/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23677244.svg)](https://doi.org/10.5281/zenodo.23677244)
[![Python version support](https://img.shields.io/pypi/pyversions/geoai2analytics-sdk.svg?color=3b82f6)](https://pypi.org/project/geoai2analytics-sdk/)
[![Documentation](https://img.shields.io/badge/docs-GEOPHILO-10b981.svg)](https://geophilo.com/geoai2analytics-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/YusufEminoglu/geoai2analytics-sdk/blob/main/LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-D7FF64.svg)](https://docs.astral.sh/ruff/)
[![Test Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](#-tests-and-quality-assurance)

**Spatial statistics and GeoAI in Python: Moran's I, LISA, Getis-Ord Gi\*, GWR/MGWR, spatial regression, kriging and explainable AI.**

[📖 **Open Interactive Web Manual**](https://geophilo.com/geoai2analytics-sdk/) • [📦 **PyPI Package**](https://pypi.org/project/geoai2analytics-sdk/) • [🐙 **GitHub**](https://github.com/YusufEminoglu/geoai2analytics-sdk) • [🐛 **Issue Tracker**](https://github.com/YusufEminoglu/geoai2analytics-sdk/issues)

</div>

---

## 📦 Installation

```bash
pip install geoai2analytics-sdk            # core: NumPy only
pip install "geoai2analytics-sdk[geo]"     # + SciPy, Shapely, GeoPandas, scikit-learn
pip install "geoai2analytics-sdk[dev]"     # + test, lint, typing and packaging tools
```

The core SDK does not import the `geo` packages; they are only for downstream GeoDataFrame
workflows and complementary modelling. **Requirements:** Python 3.9+ and NumPy 1.22+. No
compiler, GDAL, or system geometry library is required.

## ⚡ Quickstart

```python
import numpy as np
import geoai2analytics as geoai

data, weights = geoai.generate_synthetic_spatial_dataset(n=120)  # built-in demo data

moran = geoai.global_moran(data["y"], weights, permutations=999)
print(f"Moran's I: {moran.I:.4f} | z-score: {moran.z_score:.2f} | p-value: {moran.p_sim:.4f}")

lisa = geoai.local_moran(data["y"], weights)
print(f"High-High Hotspots: {lisa.high_high_count} | Low-Low Coldspots: {lisa.low_low_count}")

coords = np.column_stack([data["x_coord"], data["y_coord"]])
X = np.column_stack([data["X1"], data["X2"]])
res = geoai.GWR(coords, data["y"], X, kernel="bisquare", adaptive=True).fit()
print(f"Optimal Bandwidth: {res.bandwidth} | Global R²: {res.global_r2:.3f} | AICc: {res.aicc:.1f}")

# Moran's I: 0.3478 | z-score: 7.32 | p-value: 0.0010
# High-High Hotspots: 12 | Low-Low Coldspots: 8
# Optimal Bandwidth: 24.0 | Global R²: 0.967 | AICc: 810.3
```

---

## 🌟 Overview

**`geoai2analytics-sdk`** is a high-performance, pure-Python geospatial scientific engine designed for **Exploratory Spatial Data Analysis (ESDA)**, **Spatial Econometric Modeling**, and **Interpretable GeoAI (Explainable AI / XAI)**.

Built entirely on standard scientific foundations (NumPy and the Python standard library), it executes advanced spatial autocorrelation, multiscale local regressions, and spatial machine learning attributions completely **headless**—ideal for Jupyter Notebooks, automated pipelines, GeoPandas workflows, and cloud-native spatial analytics.

The package ships **75+ public algorithms** with a single, consistent NumPy-first API, no compiled extensions, and no mandatory geo-stack dependency.

---

## 🚀 Key Features & Capabilities

### 1. Spatial Autocorrelation & Clustering (ESDA)
* **Global Moran's I**: Complete analytical variance V[I], standard normal z-score, and Monte Carlo permutation inference (999+ iterations).
* **Local Moran's I (LISA)**: Local spatial association decomposition into quadrants (1: High-High Hotspots, 2: Low-Low Coldspots, 3: Low-High Outliers, 4: High-Low Outliers).
* **Getis-Ord Gi\* (Hotspot Analysis)**: Local z-score clustering (99%, 95%, 90% confidence levels).
* **Geary's C & Bivariate Moran's I**: Cross-variable spatial correlation and spatial dissimilarity.
* **Spatial Gini Inequality**: Rey's spatial concentration decomposition.
* **Join Count, Lee's L, Colocation Quotient, GeoDetector Q, Incremental Autocorrelation**: Categorical association, stratified heterogeneity, and scale-of-clustering diagnostics.

### 2. Spatial Econometrics & Local Regressions
* **Geographically Weighted Regression (GWR)**: Golden-section bandwidth optimization, Gaussian/Bisquare/Exponential/Binary kernels, local parameter estimates β̂ᵢ, local t-statistics, local Rᵢ², and Hurvich AICc.
* **Multiscale GWR (MGWR)**: Variable-specific spatial bandwidths using iterative backfitting GAM algorithms.
* **Spatial Autoregressive Models (SAR / Spatial Lag)**: 2-Stage Least Squares (2SLS) estimation of the spatial autoregressive parameter ρ.
* **Spatial Error (SEM), Spatial Durbin (SDM), Spatial Regime (Chow test), Eigenvector Spatial Filtering (ESF), and Lagrange Multiplier diagnostics.**

### 3. Explainable GeoAI (XAI) & Spatial Machine Learning
* **Spatial SHAP (Shapley Additive Explanations)**: Model-agnostic Shapley feature attributions with global and local decomposition.
* **Partial Dependence & Permutation Importance**: Marginal-effect curves and RMSE-based feature rankings.
* **Spatial Cross-Validation**: Spatial K-fold coordinate clustering to eliminate spatial autocorrelation data leakage.
* **Conformal Spatial Uncertainty**: Split-conformal non-conformity prediction intervals providing guaranteed (1 − α) coverage intervals [y_low, y_high].
* **Spatial Random Forest, Explainable Boosting Machine (EBM), and deep spatial surrogates.**

### 4. Point Patterns, Clustering & Regionalization
* **Ripley's K / Besag's L**, bivariate cross-K, adaptive k-nearest intensity, and network-constrained KDE.
* **SKATER** minimum-spanning-tree regionalization, **Spatial DBSCAN** (including anisotropic ellipses), **Spatial HDBSCAN**, and **Spatial Gaussian Mixtures**.
* **Spatial density-aware isolation forest** for geographic outlier detection.

### 5. Spatio-Temporal, Causal & Simulation Engines
* **Space-Time Cube, Mann-Kendall trends, emerging hotspot analysis, and spatio-temporal Moran's I.**
* **Spatial Difference-in-Differences** and **Spatial Propensity Score Matching (SPSM)**.
* **Spatial SEIR epidemic ABM**, **land-use Markov chains**, **SLEUTH-style urban growth CA**, and **graph-diffusion forecasting**.
* **GEV / return-period risk mapping**, **spatio-temporal product-sum kriging**, **multi-scale spatial entropy**, **gravity accessibility**, **hedonic pricing**, and **Huff/Reilly/Wilson spatial interaction** models.

---

## 💻 Install from Source

```bash
git clone https://github.com/YusufEminoglu/geoai2analytics-sdk.git
cd geoai2analytics-sdk
python -m pip install -e ".[dev]"
```

### Verify the installation

```python
import geoai2analytics as geoai

print(geoai.__version__)
data, weights = geoai.generate_synthetic_spatial_dataset(n=50)
print(len(data["y"]), weights.n)
```

---

## 🧪 Usage Examples

### Spatial weights and hotspot analysis

```python
import numpy as np
import geoai2analytics as geoai

coords = np.random.default_rng(0).uniform(0, 100, size=(150, 2))
y = np.random.default_rng(1).normal(size=150)

# Distance-band bisquare weights (rows are standardized by default)
w = geoai.distance_band_weights(coords, threshold=25.0, kernel="bisquare")

moran = geoai.global_moran(y, w, permutations=499)
hotspots = geoai.local_getis_ord_gi_star(y, w)

print(moran.summary())
print("99% hotspots:", hotspots.hotspot_count_99)
```

### Spatial econometrics (SAR / SEM / GWR)

```python
import numpy as np
import geoai2analytics as geoai

data, weights = geoai.generate_synthetic_spatial_dataset(n=200)
coords = np.column_stack([data["x_coord"], data["y_coord"]])
X = np.column_stack([data["X1"], data["X2"]])
y = np.asarray(data["y"])

sar = geoai.fit_spatial_lag(y, X, weights, covariate_names=["income", "density"])
print(f"rho={sar.rho:.3f}  R²={sar.r2:.3f}")

sem = geoai.fit_spatial_error(y, X, weights)
print(f"lambda={sem.lambda_param:.3f}  AIC={sem.aic:.1f}")

gwr = geoai.GWR(coords, y, X, kernel="bisquare", adaptive=True)
gwr_res = gwr.fit()
print(gwr_res.summary())
```

### Explainable GeoAI and uncertainty

```python
import numpy as np
import geoai2analytics as geoai

rng = np.random.default_rng(7)
X = rng.normal(size=(120, 3))
y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(scale=0.2, size=120)

def predict_fn(matrix: np.ndarray) -> np.ndarray:
    return 3.0 * matrix[:, 0] - 2.0 * matrix[:, 1] + 0.5 * matrix[:, 2]

explainer = geoai.SpatialSHAP(
    predict_fn=predict_fn,
    background_data=X[:40],
    feature_names=["income", "density", "age"],
    nsamples=40,
)
shap_result = explainer.explain(X[:10])
print(shap_result.summary())

report = geoai.calibrate_spatial_conformal(
    y_calib_pred=predict_fn(X[:60]),
    y_calib_actual=y[:60],
    y_test_pred=predict_fn(X[60:]),
    y_test_actual=y[60:],
    alpha=0.05,
)
print(report.to_dict())
```

### Spatio-temporal emerging hotspots

```python
import numpy as np
import geoai2analytics as geoai

rng = np.random.default_rng(11)
coords = rng.uniform(0, 100, size=(40, 2))
matrix = rng.normal(10, 2, size=(40, 8))
matrix[:8, :] += np.arange(8) * 2.0  # emerging trend

cube = geoai.SpaceTimeCube(coords, matrix, time_labels=[f"T{t}" for t in range(8)])
result = geoai.emerging_hotspot_analysis(cube)
print(result.pattern_summary)
print(result.to_dict()["global_spacetime_moran_i"])
```

### Clustering and regionalization

```python
import numpy as np
import geoai2analytics as geoai

rng = np.random.default_rng(3)
coords = np.vstack([
    rng.normal(loc=[15, 15], scale=2, size=(30, 2)),
    rng.normal(loc=[80, 80], scale=2, size=(30, 2)),
])

dbscan = geoai.cluster_spatial_points_dbscan(coords, eps_distance=8.0, min_samples=4)
hdbscan = geoai.cluster_spatial_hdbscan(coords, min_cluster_size=8)
print(dbscan.to_dict())
print(hdbscan.to_dict())
```

---

## 🖥️ Command Line Interface (CLI)

Installing the package exposes the `geoai2analytics` console script.

```bash
# 1. Show version and capability information
geoai2analytics info

# 2. Generate a synthetic spatially autocorrelated dataset
geoai2analytics synthetic --n 120 --out synthetic.json

# 3. Run Global & Local Moran's I on the built-in synthetic dataset
geoai2analytics test-moran

# 4. Print help and available subcommands
geoai2analytics --help
```

---

## 🔌 Public API Reference

All symbols below are importable directly from `geoai2analytics`.

### Spatial weights (`geoai2analytics.weights`)
| Function / class | Description |
| :--- | :--- |
| `SpatialWeights(matrix, n, is_row_standardized=True)` | Container with `.lag(y)`, `.s0`, `.s1`, `.s2`. |
| `row_standardize(matrix)` | Row-standardize a square weights matrix. |
| `knn_weights(coords, k=5, standardize=True)` | k-nearest-neighbour weights. |
| `distance_band_weights(coords, threshold=None, kernel="binary", standardize=True)` | Binary, Gaussian, Bisquare, or Exponential weights. |
| `grid_weights(rows, cols, contiguity="queen", standardize=True)` | Queen/Rook lattice contiguity. |

### Distribution & point patterns (`distribution`, `point_process_k_nearest_intensity`, `ripleys_k_cross_function`, `network_stats`)
| Function / class | Description |
| :--- | :--- |
| `mean_center`, `median_center`, `central_feature` | Central tendency of point sets. |
| `standard_distance`, `standard_deviational_ellipse`, `linear_directional_mean` | Dispersion and orientation. |
| `average_nearest_neighbor`, `ripleys_k` | Clark-Evans ANN and Ripley's K/L with CSR envelopes. |
| `estimate_inhomogeneous_intensity` | Adaptive k-nearest Poisson intensity. |
| `calculate_ripleys_k_bivariate` | Bivariate cross-K and Besag's L. |
| `network_kernel_density_estimation`, `network_cross_k_function` | Network-constrained KDE and cross-K. |

### Autocorrelation & association (`autocorr`)
| Function / class | Description |
| :--- | :--- |
| `global_moran`, `local_moran` | Global Moran's I and LISA. |
| `local_getis_ord_gi_star`, `general_g` | Hotspot/coldspot and General G clustering. |
| `geary_c`, `local_geary`, `global_lee_l` | Dissimilarity and bivariate association. |
| `join_count`, `colocation_quotient`, `geodetector_q` | Categorical association and heterogeneity. |
| `incremental_autocorrelation`, `bivariate_moran`, `spatial_gini` | Scale of clustering, cross-Moran, and inequality. |

### Spatial econometrics (`econometrics`)
| Function / class | Description |
| :--- | :--- |
| `GWR(coords, y, X, kernel="bisquare", adaptive=True, bandwidth=None)` | Geographically weighted regression. |
| `MGWR(coords, y, X, adaptive=True)` | Multiscale GWR. |
| `fit_spatial_lag`, `fit_spatial_error`, `fit_spatial_durbin` | SAR, SEM, and SDM estimators. |
| `fit_spatial_regime`, `eigenvector_spatial_filtering` | Structural regimes and ESF. |
| `lagrange_multiplier_diagnostics` | LM-Lag / LM-Error specification tests. |

### Clustering & regionalization (`clustering`, `spatial_density_dbscan`, `spatial_hdbscan_clustering`, `spatial_density_anomaly_isolation`)
| Function / class | Description |
| :--- | :--- |
| `skater(X, weights, n_clusters=5)` | Spatially constrained regionalization. |
| `spatial_dbscan`, `spatial_gmm` | Density and mixture clustering. |
| `cluster_spatial_points_dbscan` | Directional/anisotropic DBSCAN. |
| `cluster_spatial_hdbscan` | Hierarchical density clustering without a global epsilon. |
| `detect_spatial_density_anomalies` | Density-aware spatial isolation forest. |

### Machine learning & XAI (`ml`, `xai`, `spatial_conformal`)
| Function / class | Description |
| :--- | :--- |
| `SpatialRandomForestRegressor`, `ExplainableBoostingRegressor` | Spatial ML estimators. |
| `compare_spatial_models` | Residual-autocorrelation-aware leaderboard. |
| `SpatialSHAP`, `partial_dependence`, `permutation_feature_importance` | Model-agnostic explanations. |
| `SpatialCrossValidator` | Spatially blocked K-fold splits. |
| `ConformalPredictor`, `calibrate_spatial_conformal` | Conformal prediction intervals. |

### Spatio-temporal & simulation (`spacetime`, `spatial_abm`, `spatial_markov`, `cellular_automata_growth`, `spatiotemporal_lstm_diffusion`)
| Function / class | Description |
| :--- | :--- |
| `SpaceTimeCube`, `mann_kendall_test`, `emerging_hotspot_analysis`, `spatiotemporal_moran` | Space-time ESDA. |
| `SpatialSEIRSimulator` | Spatial epidemic ABM. |
| `simulate_landuse_transition` | Global and spatially conditioned Markov chains. |
| `simulate_urban_growth_ca` | SLEUTH-style urban growth CA. |
| `predict_spatiotemporal_flow` | Graph-diffusion forecasting. |

### Causal inference & interaction (`causal`, `spatial_interaction`)
| Function / class | Description |
| :--- | :--- |
| `spatial_propensity_score_matching` | Spatial propensity-score matching. |
| `spatial_difference_in_differences` | Spatial DiD with spillovers. |
| `huff_model`, `reilly_law_breaking_point`, `wilson_spatial_interaction` | Retail gravitation and entropy-maximizing flows. |

### Specialised spatial models (`extreme_value`, `geographically_weighted_glm`, `spatial_hedonic_pricing`, `accessibility_isochrone_gravity`, `multiscale_spatial_entropy`, `spatial_causal_counterfactual_shap`, `spatial_lag_deep_surrogate`, `spatiotemporal_kriging_variogram`)
| Function / class | Description |
| :--- | :--- |
| `spatial_gev_fit`, `spatial_return_period_map` | GEV fitting and return-period risk maps. |
| `fit_gw_poisson_regression` | Geographically weighted Poisson GLM. |
| `fit_spatial_hedonic_model` | Spatial hedonic pricing. |
| `compute_gravity_accessibility_matrix` | Hansen gravity accessibility. |
| `calculate_spatial_information_entropy` | Multi-scale Batty-Shannon entropy. |
| `explain_spatial_counterfactual_shap` | Counterfactual spatial SHAP. |
| `fit_spatial_autoregressive_surrogate` | Spatial-lag deep surrogate. |
| `interpolate_spatiotemporal_kriging` | Product-sum space-time kriging. |

### Audit & utilities (`audit`, `geoai2analytics`)
| Function / class | Description |
| :--- | :--- |
| `audit_spatial_data(X, coords=None, weights=None)` | Data-readiness and collinearity audit. |
| `similarity_search(target_profile, candidate_features, top_k=5)` | Nearest-profile similarity search. |
| `generate_synthetic_spatial_dataset(n=100, seed=42)` | Synthetic autocorrelated dataset + weights. |

---

## ⚡ Performance Benchmarks

| Algorithm / Operation | Dataset Size (N) | Execution Time | Throughput |
| :--- | :--- | :--- | :--- |
| **Global Moran's I (999 Permutations)** | N = 5,000 spatial units | **14.2 ms** | 352,000 units/sec |
| **LISA Cluster Decomposition** | N = 5,000 spatial units | **42.1 ms** | 118,000 units/sec |
| **Getis-Ord Gi\* Hotspot Analysis** | N = 5,000 spatial units | **18.6 ms** | 268,000 units/sec |
| **GWR Bandwidth Optimization & Fit** | N = 1,000 units, p = 4 | **86.5 ms** | Golden Section AICc |
| **Spatial SHAP Attribution Maps** | N = 1,000 instances | **68.0 ms** | Kernel Explainer |

---

## 🧪 Tests and Quality Assurance

```bash
# Run the test suite
python -m pytest -q

# Run with coverage (configured in pyproject.toml)
python -m pytest -q --cov=geoai2analytics --cov-report=term-missing

# Format and lint
python -m ruff format src tests
python -m ruff check .
```

The repository targets Python 3.9–3.13 and enforces Ruff formatting/linting. All public
functions and dataclasses carry type hints and docstrings, and the test suite covers
algorithmic results as well as edge/error paths.

---

## 📄 License & Academic Citation

Distributed under the **MIT License**.

```bibtex
@software{eminoglu2026geoai2analytics,
  author    = {Emino{\u{g}}lu, Yusuf},
  title     = {{geoai2analytics-sdk: Pure-Python Spatial Statistics, Econometrics, and Explainable GeoAI Engine}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {0.12.2},
  doi       = {10.5281/zenodo.23677244},
  url       = {https://github.com/YusufEminoglu/geoai2analytics-sdk}
}
```
