# -*- coding: utf-8 -*-
"""SLEUTH-style Constrained Urban Sprawl Growth Cellular Automata for geoai2analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class UrbanCAGrowthResult:
    initial_urban_cells: int
    final_urban_cells: int
    new_urbanized_cells: int
    urban_growth_rate_pct: float
    simulation_steps: int
    urban_grid: list[list[int]]  # 0: non-urban, 1: urbanized
    expansion_history: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_urban_cells": self.initial_urban_cells,
            "final_urban_cells": self.final_urban_cells,
            "new_urbanized_cells": self.new_urbanized_cells,
            "growth_rate_pct": round(self.urban_growth_rate_pct, 2),
            "steps": self.simulation_steps,
        }


def simulate_urban_growth_ca(
    initial_urban_mask: Sequence[Sequence[int]],
    slope_resistance_matrix: Sequence[Sequence[float]] | None = None,
    road_proximity_matrix: Sequence[Sequence[float]] | None = None,
    exclusion_mask: Sequence[Sequence[int]] | None = None,
    diffusion_coefficient: float = 0.25,
    breed_coefficient: float = 0.35,
    spread_coefficient: float = 0.50,
    road_gravity_coefficient: float = 0.40,
    steps: int = 5,
) -> UrbanCAGrowthResult:
    """Simulate spatial-temporal urban sprawl diffusion using constrained multi-criteria Cellular Automata."""
    grid = np.asarray(initial_urban_mask, dtype=np.int32)
    h, w = grid.shape

    slope = np.asarray(slope_resistance_matrix, dtype=np.float64) if slope_resistance_matrix is not None else np.zeros((h, w))
    road_prox = np.asarray(road_proximity_matrix, dtype=np.float64) if road_proximity_matrix is not None else np.ones((h, w)) * 0.5
    excluded = np.asarray(exclusion_mask, dtype=np.int32) if exclusion_mask is not None else np.zeros((h, w), dtype=np.int32)

    init_cnt = int(np.sum(grid == 1))
    history = [init_cnt]

    curr_grid = grid.copy()

    for _step in range(steps):
        next_grid = curr_grid.copy()

        # Count 8-neighborhood urban neighbors
        padded = np.pad(curr_grid, 1, mode="constant", constant_values=0)
        neighbors = np.zeros((h, w), dtype=np.int32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbors += padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]

        for r in range(h):
            for c in range(w):
                if curr_grid[r, c] == 1 or excluded[r, c] == 1:
                    continue

                nb_count = neighbors[r, c]
                slope_factor = max(0.0, 1.0 - slope[r, c])
                road_factor = road_prox[r, c]

                # 4 SLEUTH Growth Rules:
                # 1. Spontaneous Growth (Diffusion)
                # 2. New Spreading Centers (Breed)
                # 3. Edge Growth (Organic Spread)
                # 4. Road-Influenced Growth
                p_spontaneous = diffusion_coefficient * 0.05 * slope_factor
                p_breed = breed_coefficient * 0.10 * slope_factor if nb_count >= 2 else 0.0
                p_spread = spread_coefficient * (nb_count / 8.0) * slope_factor
                p_road = road_gravity_coefficient * road_factor * slope_factor if nb_count >= 1 else 0.0

                p_urbanize = max(p_spontaneous, p_breed, p_spread, p_road)
                p_urbanize = min(1.0, max(0.0, p_urbanize))

                # Deterministic threshold for stable repeatable test runs or probabilistic
                if p_urbanize >= 0.30:
                    next_grid[r, c] = 1

        curr_grid = next_grid
        history.append(int(np.sum(curr_grid == 1)))

    final_cnt = int(np.sum(curr_grid == 1))
    growth_rate = ((final_cnt - init_cnt) / max(1, init_cnt)) * 100.0

    return UrbanCAGrowthResult(
        initial_urban_cells=init_cnt,
        final_urban_cells=final_cnt,
        new_urbanized_cells=final_cnt - init_cnt,
        urban_growth_rate_pct=growth_rate,
        simulation_steps=steps,
        urban_grid=curr_grid.tolist(),
        expansion_history=history,
    )
