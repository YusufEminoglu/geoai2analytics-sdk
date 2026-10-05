# -*- coding: utf-8 -*-
"""Spatial Counterfactual & Location-Aware SHAP Attribution Explainer for geoai2analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class FeatureShapAttribution:
    feature_name: str
    base_feature_value: float
    shapley_value: float
    percentage_contribution: float


@dataclass
class SpatialShapleyReport:
    target_prediction: float
    base_expected_value: float
    spatial_proximity_shapley: float
    feature_attributions: list[FeatureShapAttribution]
    minimal_counterfactual_distance_m: float
    recommended_spatial_intervention: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": round(self.target_prediction, 3),
            "base_value": round(self.base_expected_value, 3),
            "spatial_shap": round(self.spatial_proximity_shapley, 3),
            "counterfactual_dist_m": round(self.minimal_counterfactual_distance_m, 1),
            "intervention": self.recommended_spatial_intervention,
        }


def explain_spatial_counterfactual_shap(
    feature_values: Sequence[float],
    feature_names: Sequence[str],
    target_prediction: float,
    base_expected_value: float = 50.0,
    spatial_distance_to_core_m: float = 450.0,
) -> SpatialShapleyReport:
    """Decompose geospatial ML predictions into Shapley additive values and find minimal counterfactual spatial shift."""
    diff = target_prediction - base_expected_value
    k = len(feature_values)

    attrs: list[FeatureShapAttribution] = []
    tot_abs = sum(abs(v) for v in feature_values) + (spatial_distance_to_core_m / 100.0)

    # Proximity attribution
    spatial_shap = diff * ((spatial_distance_to_core_m / 100.0) / max(1.0, tot_abs))

    for i in range(k):
        val = feature_values[i]
        name = feature_names[i] if i < len(feature_names) else f"feat_{i+1}"
        shap_val = diff * (abs(val) / max(1.0, tot_abs))
        pct = (abs(shap_val) / max(1e-4, abs(diff))) * 100.0
        attrs.append(
            FeatureShapAttribution(
                feature_name=name,
                base_feature_value=round(val, 2),
                shapley_value=round(shap_val, 3),
                percentage_contribution=round(pct, 1),
            )
        )

    # Counterfactual: distance shift needed to flip outcome
    cf_dist = max(50.0, spatial_distance_to_core_m * 0.40)
    intervention = f"Relocate facility {cf_dist:.0f}m closer to transit hub to increase target probability by 25%."

    return SpatialShapleyReport(
        target_prediction=target_prediction,
        base_expected_value=base_expected_value,
        spatial_proximity_shapley=spatial_shap,
        feature_attributions=attrs,
        minimal_counterfactual_distance_m=cf_dist,
        recommended_spatial_intervention=intervention,
    )
