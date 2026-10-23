# -*- coding: utf-8 -*-
"""Edge-case and error-path tests for under-covered public geoai2analytics APIs."""

from __future__ import annotations

import numpy as np
import pytest

import geoai2analytics as geoai
from geoai2analytics.cli import main as cli_main


def _small_weights(n: int = 6) -> geoai.SpatialWeights:
    """Build a small row-standardized KNN weights object for validation tests."""
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    return geoai.knn_weights(coords, k=2)


def test_spatial_weights_validation() -> None:
    """SpatialWeights and lag reject mismatched shapes."""
    with pytest.raises(ValueError):
        geoai.SpatialWeights(matrix=np.zeros((3, 2)), n=3)

    weights = _small_weights()
    with pytest.raises(ValueError):
        weights.lag(np.zeros(2))


def test_distance_band_unknown_kernel_and_defaults() -> None:
    """Unknown kernels raise and omitted thresholds still build a graph."""
    coords = np.column_stack([np.arange(6, dtype=float), np.zeros(6)])

    with pytest.raises(ValueError):
        geoai.distance_band_weights(coords, threshold=5.0, kernel="not-a-kernel")

    weights = geoai.distance_band_weights(coords, kernel="bisquare")
    assert weights.n == 6
    assert weights.s0 > 0.0


def test_knn_k_exceeds_n_and_unstandardized_grid() -> None:
    """KNN caps k at n-1 and grid weights support rook contiguity without scaling."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    weights = geoai.knn_weights(coords, k=5, standardize=False)
    assert weights.n == 2
    assert not weights.is_row_standardized

    grid = geoai.grid_weights(3, 3, contiguity="rook", standardize=False)
    assert grid.n == 9
    assert float(np.sum(grid.matrix[0])) == 2.0

    standardized = geoai.row_standardize(np.zeros((2, 2)))
    assert np.all(standardized == 0.0)


def test_global_and_local_moran_constant_inputs() -> None:
    """Constant response vectors yield zero statistics without raising."""
    weights = _small_weights()

    global_res = geoai.global_moran(np.ones(weights.n), weights, permutations=9)
    assert global_res.I == 0.0
    assert global_res.p_sim == 1.0

    with pytest.raises(ValueError):
        geoai.global_moran(np.ones(weights.n + 1), weights, permutations=9)

    lisa = geoai.local_moran(np.ones(weights.n), weights, permutations=9)
    assert np.all(lisa.quadrants == 0)
    assert lisa.high_high_count == 0


def test_join_count_bivariate_moran_and_gini_errors() -> None:
    """Categorical, length, and zero-mean edge cases are handled explicitly."""
    weights = _small_weights()

    with pytest.raises(ValueError):
        geoai.join_count(np.zeros(weights.n, dtype=int), weights)

    with pytest.raises(ValueError):
        geoai.bivariate_moran(np.ones(weights.n + 1), np.ones(weights.n), weights)

    gini = geoai.spatial_gini(np.zeros(weights.n), weights)
    assert gini.gini == 0.0
    assert gini.spatial_gini == 0.0
    assert gini.spatial_disparity_ratio == 1.0


def test_gravity_accessibility_decay_models() -> None:
    """Empty origins short-circuit and Gaussian/Power decays remain positive."""
    empty = geoai.compute_gravity_accessibility_matrix([], [(0.0, 0.0)])
    assert empty.origin_accessibility_scores == []
    assert empty.to_dict()["origins_count"] == 0

    for decay in (geoai.DistanceDecayType.GAUSSIAN, geoai.DistanceDecayType.POWER):
        report = geoai.compute_gravity_accessibility_matrix(
            [(0.0, 0.0)],
            [(10.0, 0.0)],
            facility_capacities=[2.0],
            decay_type=decay,
        )
        assert report.total_origins_analyzed == 1
        assert report.origin_accessibility_scores[0] > 0.0
        assert report.to_dict()["origins_count"] == 1


def test_gev_requires_three_observations_and_return_period_map() -> None:
    """GEV fitting validates sample size and maps return periods per station."""
    with pytest.raises(ValueError):
        geoai.spatial_gev_fit([1.0, 2.0])

    series = np.array(
        [
            [5.0, 6.0, 7.5, 8.0, 9.5, 10.0],
            [4.0, 5.5, 6.0, 7.5, 8.0, 9.0],
        ]
    )
    records = geoai.spatial_return_period_map(
        [(0.0, 0.0), (1.0, 1.0)], series, target_return_years=50.0
    )

    assert len(records) == 2
    assert records[0]["return_period_years"] == 50.0
    assert records[0]["return_level"] > 0.0

    fit = geoai.spatial_gev_fit(series[0])
    assert fit.to_dict()["return_10yr"] > 0.0


def test_conformal_without_actual_values() -> None:
    """Missing test actuals fall back to the nominal coverage level."""
    report = geoai.calibrate_spatial_conformal(
        y_calib_pred=[1.0, 2.0, 3.0, 4.0],
        y_calib_actual=[1.1, 2.2, 3.1, 4.3],
        y_test_pred=[1.5, 2.5],
    )

    assert report.empirical_coverage_rate == pytest.approx(0.95)
    assert all(interval.contains_actual is None for interval in report.intervals)
    assert report.to_dict()["total_intervals"] == 2


def test_spacetime_cube_validation_and_short_series() -> None:
    """Cube shape mismatches raise; short/constant series are handled."""
    with pytest.raises(ValueError):
        geoai.SpaceTimeCube(coords=np.zeros((3, 2)), time_series_matrix=np.zeros((4, 2)))

    short = geoai.mann_kendall_test([1.0, 2.0])
    assert short.trend == "no trend"
    assert short.sen_slope == 0.0

    cube = geoai.SpaceTimeCube(coords=np.zeros((3, 2)), time_series_matrix=np.zeros((3, 4)))
    assert geoai.spatiotemporal_moran(cube) == 0.0


def test_cli_default_synthetic_and_file_output(tmp_path, capsys) -> None:
    """The CLI handles no subcommand, stdout output, and JSON file output."""
    assert cli_main([]) == 0
    assert cli_main(["synthetic", "--n", "8"]) == 0

    out = tmp_path / "synthetic.json"
    assert cli_main(["synthetic", "--n", "8", "--out", str(out)]) == 0
    assert out.is_file()
    assert "Generated" in capsys.readouterr().out


def test_network_kde_and_cross_k_edge_cases() -> None:
    """Empty event sets and Gaussian kernels are supported."""
    empty = geoai.network_kernel_density_estimation([], [(0.0, 0.0)], bandwidth=100.0)
    assert empty.total_events == 0
    assert empty.segment_densities == [0.0]
    assert empty.to_dict()["total_events"] == 0

    gaussian = geoai.network_kernel_density_estimation(
        [(0.0, 0.0)], [(10.0, 0.0)], bandwidth=100.0, kernel="gaussian"
    )
    assert gaussian.segment_densities[0] > 0.0

    cross = geoai.network_cross_k_function([], [(0.0, 0.0)], distances=(10.0, 20.0))
    assert cross == {10.0: 0.0, 20.0: 0.0}


def test_point_process_and_ripleys_empty_inputs() -> None:
    """Empty point patterns and oversized k are handled gracefully."""
    empty = geoai.estimate_inhomogeneous_intensity([])
    assert empty.total_events_count == 0
    assert empty.to_dict()["events_count"] == 0

    capped = geoai.estimate_inhomogeneous_intensity(
        [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], k_nearest_neighbors=10
    )
    assert capped.k_neighbors_used == 2

    k_cross = geoai.calculate_ripleys_k_bivariate([], [(0.0, 0.0)], study_area_m2=100.0)
    assert k_cross.radius_distances_r == []
    assert k_cross.to_dict()["attraction"] is False


def test_clustering_edge_cases() -> None:
    """Both density-based clusterers return empty partitions for no input."""
    dbscan = geoai.cluster_spatial_points_dbscan([])
    assert dbscan.total_points == 0

    hdbscan = geoai.cluster_spatial_hdbscan([])
    assert hdbscan.total_points_count == 0


def test_stream_engines_empty_inputs() -> None:
    """Hedonic, surrogate, kriging, and diffusion engines tolerate empty input."""
    assert geoai.fit_spatial_hedonic_model([], [], []).r_squared == 0.0
    assert geoai.fit_spatial_autoregressive_surrogate([], [], []).learned_weights_count == 0
    assert (
        geoai.interpolate_spatiotemporal_kriging(
            [], (0.0, 0.0, 0.0)
        ).num_spatiotemporal_neighbors_used
        == 0
    )
    assert geoai.predict_spatiotemporal_flow([], []).num_sensor_nodes == 0


def test_entropy_and_interaction_edge_cases() -> None:
    """Entropy handles empty/single inputs and interaction models stay normalized."""
    empty = geoai.calculate_spatial_information_entropy([])
    assert empty.total_features_count == 0
    assert empty.to_dict()["total_features"] == 0

    single = geoai.calculate_spatial_information_entropy([(0.0, 0.0)])
    assert single.mean_normalized_entropy == 0.0

    reilly = geoai.reilly_law_breaking_point(1000.0, 4000.0, 10.0)
    assert 0.0 < reilly.breaking_point_from_a_km < 10.0

    demand = geoai.huff_model([(0.0, 0.0)], [100.0], [(10.0, 0.0), (20.0, 0.0)], [2.0, 1.0])
    assert sum(demand.probabilities[0]) == pytest.approx(1.0)
    assert demand.to_dict()["n_origins"] == 1

    flow = geoai.wilson_spatial_interaction(
        origins_supply=[100.0, 200.0],
        destinations_demand=[150.0, 150.0],
        cost_matrix=[[5.0, 20.0], [25.0, 10.0]],
    )
    assert flow.total_flow > 0.0


def test_markov_with_spatial_lag_and_to_dict() -> None:
    """Spatially conditioned transitions populate conditional matrices."""
    result = geoai.simulate_landuse_transition(
        states_t0=[0, 1, 1, 0, 1, 0],
        states_t1=[1, 1, 0, 0, 1, 1],
        spatial_lag_states_t0=[0, 1, 1, 0, 0, 1],
        class_names=["Low", "High"],
    )

    assert result.num_classes == 2
    assert result.class_labels == ["Low", "High"]
    assert result.spatial_conditional_matrices
    payload = result.to_dict()
    assert payload["num_classes"] == 2
    assert "global_matrix" in payload


def test_seir_simulation_and_to_dict() -> None:
    """The SEIR simulator returns one history entry per simulated day."""
    simulator = geoai.SpatialSEIRSimulator([1000.0, 2000.0], [(0.0, 0.0), (100.0, 0.0)])
    result = simulator.simulate(days=10)

    assert result.timesteps == 10
    assert len(result.infectious_history) == 10
    assert result.to_dict()["timesteps"] == 10


def test_mean_center_validation_and_linear_mean_empty() -> None:
    """Mean center validates shapes/weights and empty line sets are safe."""
    with pytest.raises(ValueError):
        geoai.mean_center(np.zeros((3, 3)))

    with pytest.raises(ValueError):
        geoai.mean_center([(0.0, 0.0), (1.0, 1.0)], weights=[1.0])

    with pytest.raises(ValueError):
        geoai.mean_center([(0.0, 0.0), (1.0, 1.0)], weights=[0.0, 0.0])

    empty = geoai.linear_directional_mean([])
    assert empty.directional_strength == "None"


def test_average_nearest_neighbor_requires_three_points() -> None:
    """ANN rejects point sets too small for a meaningful test."""
    with pytest.raises(ValueError):
        geoai.average_nearest_neighbor([(0.0, 0.0), (1.0, 1.0)])


def test_compare_spatial_models_empty_leaderboard() -> None:
    """An empty model mapping yields an empty leaderboard."""
    coords = np.column_stack([np.arange(5, dtype=float), np.zeros(5)])
    weights = geoai.knn_weights(coords, k=2)
    X = np.arange(10, dtype=float).reshape(5, 2)
    y = np.arange(5, dtype=float)

    table = geoai.compare_spatial_models({}, X, y, X, y, weights)
    assert table.leaderboard == []
    assert table.best_model_name == "None"


def test_gwr_fixed_gaussian_and_spatial_error_1d() -> None:
    """Fixed Gaussian GWR and 1-D covariate SEM branches execute correctly."""
    data, weights = geoai.generate_synthetic_spatial_dataset(n=20, seed=1)
    coords = np.column_stack([data["x_coord"], data["y_coord"]])
    y = np.asarray(data["y"])
    x1 = np.asarray(data["X1"])
    X = np.column_stack([x1, data["X2"]])

    gwr = geoai.GWR(coords, y, X, kernel="gaussian", adaptive=False, bandwidth=40.0)
    result = gwr.fit()
    assert result.params.shape == (20, 3)

    sem = geoai.fit_spatial_error(y, x1, weights, max_iter=3)
    assert len(sem.betas) == 2


def test_spatial_cross_validator_more_splits_than_points() -> None:
    """Requested folds are capped at the number of observations."""
    folds = geoai.SpatialCrossValidator(n_splits=5, seed=1).split([(0.0, 0.0), (1.0, 1.0)])

    assert len(folds) == 2
    assert all(len(fold.test_indices) >= 1 for fold in folds)
