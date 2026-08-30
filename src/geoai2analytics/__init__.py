# -*- coding: utf-8 -*-
"""
geoai2analytics — Pure-Python Spatial Statistics, Econometrics, and Explainable GeoAI Master Suite.
"""

from __future__ import annotations

__version__ = "0.9.0"
__author__ = "Yusuf Eminoğlu"

from .accessibility_isochrone_gravity import (
    DistanceDecayType,
    GravityAccessibilityReport,
    compute_gravity_accessibility_matrix,
)
from .audit import (
    AuditIssue,
    DataReadinessReport,
    SimilarityMatch,
    audit_spatial_data,
    similarity_search,
)
from .autocorr import (
    BivariateMoranResult,
    ColocationQuotientResult,
    GearyCResult,
    GeneralGResult,
    GeodetectorQResult,
    GlobalMoranResult,
    IncrementalAutocorrResult,
    JoinCountResult,
    LeeLResult,
    LocalGearyResult,
    LocalGetisOrdResult,
    LocalMoranResult,
    SpatialGiniResult,
    bivariate_moran,
    colocation_quotient,
    geary_c,
    general_g,
    geodetector_q,
    global_lee_l,
    global_moran,
    incremental_autocorrelation,
    join_count,
    local_geary,
    local_getis_ord_gi_star,
    local_moran,
    spatial_gini,
)
from .causal import (
    SDIDResult,
    SPSMMatchPair,
    SPSMResult,
    spatial_difference_in_differences,
    spatial_propensity_score_matching,
)
from .cellular_automata_growth import (
    UrbanCAGrowthResult,
    simulate_urban_growth_ca,
)
from .clustering import (
    SKATERResult,
    SpatialClusteringResult,
    skater,
    spatial_dbscan,
    spatial_gmm,
)
from .distribution import (
    ANNResult,
    CentralFeatureResult,
    LinearDirectionalMeanResult,
    MeanCenterResult,
    MedianCenterResult,
    RipleysKResult,
    SDEResult,
    StandardDistanceResult,
    average_nearest_neighbor,
    central_feature,
    linear_directional_mean,
    mean_center,
    median_center,
    ripleys_k,
    standard_deviational_ellipse,
    standard_distance,
)
from .econometrics import (
    GWR,
    MGWR,
    ESFResult,
    GWRResult,
    LMDiagnosticsResult,
    MGWRResult,
    SpatialDurbinResult,
    SpatialErrorResult,
    SpatialLagResult,
    SpatialRegimeResult,
    eigenvector_spatial_filtering,
    fit_spatial_durbin,
    fit_spatial_error,
    fit_spatial_lag,
    fit_spatial_regime,
    lagrange_multiplier_diagnostics,
)
from .extreme_value import (
    GEVFitResult,
    spatial_gev_fit,
    spatial_return_period_map,
)
from .geographically_weighted_glm import (
    GWGLMResult,
    fit_gw_poisson_regression,
)
from .multiscale_spatial_entropy import (
    ScaleEntropyProfile,
    SpatialEntropyReport,
    calculate_spatial_information_entropy,
)
from .network_stats import (
    NetKDEResult,
    network_cross_k_function,
    network_kernel_density_estimation,
)
from .ripleys_k_cross_function import (
    RipleysKCrossResult,
    calculate_ripleys_k_bivariate,
)
from .spatial_abm import (
    EpidemicSpreadResult,
    SpatialSEIRSimulator,
)
from .spatial_conformal import (
    ConformalCoverageReport,
    ConformalInterval,
    calibrate_spatial_conformal,
)
from .spatial_density_dbscan import (
    SpatialClusterResult,
    cluster_spatial_points_dbscan,
)
from .spatial_hdbscan_clustering import (
    ClusterCondensedTree,
    HDBSCANResult,
    cluster_spatial_hdbscan,
)
from .spatial_hedonic_pricing import (
    HedonicValuationResult,
    fit_spatial_hedonic_model,
)
from .spatial_interaction import (
    HuffModelResult,
    ReillyResult,
    WilsonFlowResult,
    huff_model,
    reilly_law_breaking_point,
    wilson_spatial_interaction,
)
from .spatial_markov import (
    SpatialMarkovResult,
    simulate_landuse_transition,
)
from .spacetime import (
    EmergingHotspotResult,
    MannKendallResult,
    SpaceTimeAnalysisResult,
    SpaceTimeCube,
    emerging_hotspot_analysis,
    mann_kendall_test,
    spatiotemporal_moran,
)
from .ml import (
    ExplainableBoostingRegressor,
    MLComparisonTable,
    SpatialModelMetrics,
    SpatialRandomForestRegressor,
    compare_spatial_models,
)
from .weights import (
    SpatialWeights,
    distance_band_weights,
    grid_weights,
    knn_weights,
    row_standardize,
)
from .xai import (
    ConformalPredictionInterval,
    ConformalPredictor,
    PDPResult,
    PermutationImportanceResult,
    SpatialCrossValidator,
    SpatialCVFold,
    SpatialSHAP,
    SpatialSHAPResult,
    partial_dependence,
    permutation_feature_importance,
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

    e = rng.normal(0, 1, size=n)
    X1 = rng.uniform(10, 50, size=n)
    X2 = rng.uniform(1, 5, size=n)
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
    # Weights
    "SpatialWeights",
    "row_standardize",
    "knn_weights",
    "distance_band_weights",
    "grid_weights",
    # Distribution & Dispersion
    "MeanCenterResult",
    "MedianCenterResult",
    "CentralFeatureResult",
    "StandardDistanceResult",
    "SDEResult",
    "LinearDirectionalMeanResult",
    "ANNResult",
    "RipleysKResult",
    "mean_center",
    "median_center",
    "central_feature",
    "standard_distance",
    "standard_deviational_ellipse",
    "linear_directional_mean",
    "average_nearest_neighbor",
    "ripleys_k",
    # Autocorrelation & Association
    "GlobalMoranResult",
    "LocalMoranResult",
    "LocalGetisOrdResult",
    "GeneralGResult",
    "GearyCResult",
    "LocalGearyResult",
    "LeeLResult",
    "JoinCountResult",
    "ColocationQuotientResult",
    "GeodetectorQResult",
    "IncrementalAutocorrResult",
    "BivariateMoranResult",
    "SpatialGiniResult",
    "global_moran",
    "local_moran",
    "local_getis_ord_gi_star",
    "general_g",
    "geary_c",
    "local_geary",
    "global_lee_l",
    "join_count",
    "colocation_quotient",
    "geodetector_q",
    "incremental_autocorrelation",
    "bivariate_moran",
    "spatial_gini",
    # Econometrics
    "GWR",
    "MGWR",
    "GWRResult",
    "MGWRResult",
    "SpatialLagResult",
    "SpatialErrorResult",
    "SpatialDurbinResult",
    "SpatialRegimeResult",
    "ESFResult",
    "LMDiagnosticsResult",
    "fit_spatial_lag",
    "fit_spatial_error",
    "fit_spatial_durbin",
    "fit_spatial_regime",
    "eigenvector_spatial_filtering",
    "lagrange_multiplier_diagnostics",
    # Clustering
    "SKATERResult",
    "SpatialClusteringResult",
    "skater",
    "spatial_dbscan",
    "spatial_gmm",
    # Machine Learning
    "SpatialRandomForestRegressor",
    "ExplainableBoostingRegressor",
    "SpatialModelMetrics",
    "MLComparisonTable",
    "compare_spatial_models",
    # Explainable GeoAI
    "SpatialCrossValidator",
    "SpatialCVFold",
    "SpatialSHAP",
    "SpatialSHAPResult",
    "PDPResult",
    "PermutationImportanceResult",
    "ConformalPredictor",
    "ConformalPredictionInterval",
    "partial_dependence",
    "permutation_feature_importance",
    # Audit & Similarity
    "AuditIssue",
    "DataReadinessReport",
    "SimilarityMatch",
    "audit_spatial_data",
    "similarity_search",
    # Spatio-Temporal & Emerging Hotspots
    "SpaceTimeCube",
    "emerging_hotspot_analysis",
    "mann_kendall_test",
    "spatiotemporal_moran",
    "MannKendallResult",
    "EmergingHotspotResult",
    "SpaceTimeAnalysisResult",
    # Spatial Causal Inference
    "spatial_propensity_score_matching",
    "spatial_difference_in_differences",
    "SPSMResult",
    "SPSMMatchPair",
    "SDIDResult",
    # Spatial Interaction & Gravity Models
    "huff_model",
    "reilly_law_breaking_point",
    "wilson_spatial_interaction",
    "HuffModelResult",
    "ReillyResult",
    "WilsonFlowResult",
    # Extreme Value Risk
    "spatial_gev_fit",
    "spatial_return_period_map",
    "GEVFitResult",
    # NetKDE & Network Stats
    "network_kernel_density_estimation",
    "network_cross_k_function",
    "NetKDEResult",
    # Spatial ABM & SEIR
    "SpatialSEIRSimulator",
    "EpidemicSpreadResult",
    # Spatial Markov Chains
    "simulate_landuse_transition",
    "SpatialMarkovResult",
    # Spatial Conformal Prediction
    "calibrate_spatial_conformal",
    "ConformalCoverageReport",
    "ConformalInterval",
    # Geographically Weighted GLM (Poisson / Logistic)
    "fit_gw_poisson_regression",
    "GWGLMResult",
    # Directional Anisotropic Spatial DBSCAN
    "cluster_spatial_points_dbscan",
    "SpatialClusterResult",
    # Spatial Hedonic Real Estate Valuation
    "fit_spatial_hedonic_model",
    "HedonicValuationResult",
    # SLEUTH-style Urban Sprawl Cellular Automata
    "simulate_urban_growth_ca",
    "UrbanCAGrowthResult",
    # Bivariate Ripley's K Point Pattern Cross-Function
    "calculate_ripleys_k_bivariate",
    "RipleysKCrossResult",
    # Continuous Gravity Accessibility Potential
    "compute_gravity_accessibility_matrix",
    "GravityAccessibilityReport",
    "DistanceDecayType",
    # Hierarchical Density-Based Spatial Clustering (Spatial HDBSCAN)
    "cluster_spatial_hdbscan",
    "HDBSCANResult",
    "ClusterCondensedTree",
    # Multi-Scale Spatial Information Entropy & Sprawl Dispersion
    "calculate_spatial_information_entropy",
    "SpatialEntropyReport",
    "ScaleEntropyProfile",
]
