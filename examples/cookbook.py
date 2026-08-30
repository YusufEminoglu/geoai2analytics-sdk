# -*- coding: utf-8 -*-
"""geoai2analytics Cookbook — Huff Retail Gravity, GEV Risk & Spatial Markov."""

import geoai2analytics as geoai

# 1. Huff Probabilistic Retail Model
huff = geoai.huff_model(
    origin_coords=[(0.0, 0.0), (50.0, 50.0)],
    origin_demands=[1000.0, 2000.0],
    destination_coords=[(20.0, 20.0), (80.0, 80.0)],
    destination_attractiveness=[5000.0, 3000.0],
)
print(f"Expected Patrons by Mall: {huff.expected_patrons_by_destination}")

# 2. Extreme Value GEV Fit (100-year flood/heatwave)
gev = geoai.spatial_gev_fit([25.0, 32.0, 41.0, 28.0, 35.0, 48.0, 30.0, 38.0, 52.0])
print(f"100-Year Return Level: {gev.return_level_100yr:.2f}")

# 3. Spatial Conformal Regressor
conformal = geoai.calibrate_spatial_conformal([10, 20, 30], [10.2, 19.8, 30.5], [15, 25])
print(f"Calibrated 95% Quantile Margin: {conformal.calibrated_quantile_q:.3f}")
