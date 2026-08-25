# -*- coding: utf-8 -*-
"""
geoai2analytics — Pure-Python Spatial Statistics, Econometrics, and Explainable GeoAI Engine.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Yusuf Eminoğlu"

from .autocorr import (
    BivariateMoranResult,
    GearyCResult,
    GlobalMoranResult,
    LocalGetisOrdResult,
    LocalMoranResult,
    SpatialGiniResult,
    bivariate_moran,
    global_moran,
    local_getis_ord_gi_star,
    local_moran,
    spatial_gini,
)
from .econometrics import (
    GWR,
    MGWR,
    GWRResult,
    MGWRResult,
    SpatialLagResult,
    fit_spatial_lag,
)
from .geoai_xai import (
    ConformalPredictionInterval,
    ConformalPredictor,
    SpatialCrossValidator,
    SpatialCVFold,
    SpatialSHAP,
    SpatialSHAPResult,
)
from .weights import (
    SpatialWeights,
    distance_band_weights,
    grid_weights,
    knn_weights,
    row_standardize,
)


def generate_synthetic_spatial_dataset(
    n: int = 100,
    seed: int = 42,
) -> tuple[dict[str, list[float]], SpatialWeights]:
    """Generate synthetic spatial coordinates and autocorrelated spatial values for quick testing."""
    import numpy as np

    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 100.0, size=(n, 2))
    weights = knn_weights(coords, k=6)

    # Generate spatial lag autoregressive signal
    e = rng.normal(0, 1, size=n)
    X1 = rng.uniform(10, 50, size=n)
    X2 = rng.uniform(1, 5, size=n)
    # y = 2.5 + 0.8 * W*y + 1.2 * X1 - 2.0 * X2 + e
    I_n = np.eye(n)
    inv_mat = np.linalg.inv(I_n - 0.65 * weights.matrix)
    signal = 2.0 + 1.5 * X1 - 3.0 * X2 + e
    y = np.dot(inv_mat, signal)

    data = {
        "x_coord": coords[:, 0].tolist(),
        "y_coord": coords[:, 1].tolist(),
        "y": y.tolist(),
        "X1": X1.tolist(),
        "X2": X2.tolist(),
    }
    return data, weights


__all__ = [
    "__version__",
    "SpatialWeights",
    "row_standardize",
    "knn_weights",
    "distance_band_weights",
    "grid_weights",
    "GlobalMoranResult",
    "LocalMoranResult",
    "LocalGetisOrdResult",
    "GearyCResult",
    "BivariateMoranResult",
    "SpatialGiniResult",
    "global_moran",
    "local_moran",
    "local_getis_ord_gi_star",
    "bivariate_moran",
    "spatial_gini",
    "GWR",
    "MGWR",
    "GWRResult",
    "MGWRResult",
    "SpatialLagResult",
    "fit_spatial_lag",
    "SpatialCrossValidator",
    "SpatialCVFold",
    "SpatialSHAP",
    "SpatialSHAPResult",
    "ConformalPredictor",
    "ConformalPredictionInterval",
    "generate_synthetic_spatial_dataset",
]
