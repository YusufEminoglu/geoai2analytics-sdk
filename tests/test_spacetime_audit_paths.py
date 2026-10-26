# -*- coding: utf-8 -*-
"""Branch coverage for the emerging-hotspot classifier and the data readiness audit."""

from __future__ import annotations

import numpy as np
import pytest

from geoai2analytics import spacetime
from geoai2analytics.audit import audit_spatial_data, similarity_search
from geoai2analytics.autocorr import LocalGetisOrdResult
from geoai2analytics.spacetime import (
    SpaceTimeCube,
    emerging_hotspot_analysis,
    mann_kendall_test,
    spatiotemporal_moran,
)
from geoai2analytics.weights import SpatialWeights

H, C, N = 3.0, -3.0, 0.0  # hot, cold and neutral Gi* z-scores

# One row per location: designed Gi* z-score series over T = 10 steps.
_SERIES = {
    "New Hotspot": [N] * 9 + [H],
    "Intensifying Hotspot": [N, N, N, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "Persistent Hotspot": [H] * 10,
    "Sporadic Hotspot": [H, N, N, N, N, N, N, N, N, H],
    "New Coldspot": [N] * 9 + [C],
    "Intensifying Coldspot": [N, N, N, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0],
    "Persistent Coldspot": [C] * 10,
    "Consecutive Coldspot": [C, N, C, N, C, N, N, C, C, C],
    "Oscillating Coldspot": [C, H, N, N, N, N, N, N, H, C],
    "Sporadic Coldspot": [C, N, N, N, N, N, N, N, N, C],
    "Oscillating Hotspot": [H, C, N, N, N, N, N, N, C, H],
    "Historical Hotspot": [N, H, N, H, H, N, H, H, N, N],
    "Diminishing Hotspot": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 1.0, 0.5],
    "Historical Coldspot": [N, C, N, C, C, N, C, C, N, N],
    "Diminishing Coldspot": [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.5, -1.0, -0.5],
}


def test_emerging_hotspot_pattern_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each designed z-score series lands in its emerging-hotspot category."""
    names = list(_SERIES)
    z = np.array([_SERIES[k] for k in names])

    def fake_gi(values: np.ndarray, weights: SpatialWeights) -> LocalGetisOrdResult:
        t = fake_gi.calls  # type: ignore[attr-defined]
        fake_gi.calls += 1  # type: ignore[attr-defined]
        col = z[:, t]
        return LocalGetisOrdResult(col, np.full(col.shape, 0.01), np.zeros(col.shape))

    fake_gi.calls = 0  # type: ignore[attr-defined]
    monkeypatch.setattr(spacetime, "local_getis_ord_gi_star", fake_gi)

    n = len(names)
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    cube = SpaceTimeCube(coords, np.random.default_rng(0).random((n, 10)))
    res = emerging_hotspot_analysis(cube)  # default KNN weights path
    got = [r.pattern_type for r in res.results]
    assert got == names
    assert sum(res.pattern_summary.values()) == n


def test_spatiotemporal_moran_zero_weights_and_constant_cube() -> None:
    """Zero weights or a constant cube give I = 0 instead of dividing by zero."""
    coords = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    cube = SpaceTimeCube(coords, [[1.0, 2.0], [3.0, 1.0], [2.0, 5.0]], time_labels=["a", "b"])
    assert cube.time_labels == ["a", "b"]
    assert spatiotemporal_moran(cube, SpatialWeights(np.zeros((3, 3)), 3)) == 0.0
    flat = SpaceTimeCube(coords, np.ones((3, 2)))
    assert spatiotemporal_moran(flat) == 0.0
    with pytest.raises(ValueError):
        SpaceTimeCube(coords[:2], np.ones((3, 2)))


def test_mann_kendall_short_and_decreasing() -> None:
    """Short series are neutral; a monotone decline is a significant decrease."""
    assert mann_kendall_test([1.0, 2.0]).trend == "no trend"
    mk = mann_kendall_test([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
    assert mk.trend == "decreasing"
    assert mk.sen_slope == pytest.approx(-1.0)


def test_audit_flags_every_issue_type() -> None:
    """Small sample, constant, skewed, collinear and island issues are all reported."""
    rng = np.random.default_rng(1)
    a = rng.random(20)
    skewed = np.zeros(20)
    skewed[0] = 100.0
    X = np.column_stack([a, a * 2.0 + 1e-6 * rng.random(20), np.full(20, 5.0), skewed])
    w = np.zeros((20, 20))
    w[0, 1] = w[1, 0] = 1.0
    rep = audit_spatial_data(X, weights=SpatialWeights(w, 20), feature_names=list("abcd"))
    kinds = {i.issue_type for i in rep.issues}
    assert kinds == {
        "Small Sample Size",
        "Zero Variance",
        "High Skewness",
        "Multicollinearity",
        "Spatial Islands",
    }
    assert rep.has_critical_issues
    assert rep.island_count == 18
    assert rep.vif_scores["a"] > 7.5


def test_audit_single_feature_vector() -> None:
    """A 1-D input is audited as one feature with no VIF scores."""
    rep = audit_spatial_data(np.arange(40, dtype=float))
    assert rep.n_features == 1
    assert rep.vif_scores == {}
    assert rep.issues == []


def test_similarity_search_one_dimensional_candidates() -> None:
    """1-D candidates are treated as a single attribute; identical match scores 1."""
    matches = similarity_search([2.0], [1.0, 2.0, 5.0], top_k=2)
    assert [m.candidate_index for m in matches] == [1, 0]
    assert matches[0].similarity_score == 1.0
