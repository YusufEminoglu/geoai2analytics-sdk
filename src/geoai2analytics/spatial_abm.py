# -*- coding: utf-8 -*-
"""Spatially Explicit SEIR Epidemic & Hazard Diffusion Simulator for geoai2analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class EpidemicSpreadResult:
    """Simulation trajectory of epidemic diffusion across spatial zones."""

    timesteps: int
    susceptible_history: list[int]
    exposed_history: list[int]
    infectious_history: list[int]
    recovered_history: list[int]
    peak_infectious_day: int
    peak_infectious_count: int
    final_attack_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timesteps": self.timesteps,
            "peak_infectious_day": self.peak_infectious_day,
            "peak_infectious_count": self.peak_infectious_count,
            "final_attack_rate": round(self.final_attack_rate, 4),
        }


class SpatialSEIRSimulator:
    """Agent-based spatial SEIR model running across connected regional zones."""

    def __init__(
        self,
        zone_populations: Sequence[int],
        zone_coords: Sequence[tuple[float, float]],
        beta_transmission: float = 0.35,
        sigma_incubation: float = 0.20,  # 1/incubation_period (5 days)
        gamma_recovery: float = 0.10,    # 1/infectious_period (10 days)
        mobility_rate: float = 0.05,     # Inter-zone daily commuting fraction
    ) -> None:
        self.populations = np.asarray(zone_populations, dtype=float)
        self.coords = list(zone_coords)
        self.n_zones = len(self.populations)
        self.beta = beta_transmission
        self.sigma = sigma_incubation
        self.gamma = gamma_recovery
        self.mobility_rate = mobility_rate

        # Spatial distance gravity mobility matrix
        self.mobility_matrix = np.zeros((self.n_zones, self.n_zones))
        for i in range(self.n_zones):
            for j in range(self.n_zones):
                if i != j:
                    dist = max(100.0, math.hypot(self.coords[i][0] - self.coords[j][0], self.coords[i][1] - self.coords[j][1]))
                    self.mobility_matrix[i, j] = self.populations[j] / (dist ** 1.5)
            row_sum = np.sum(self.mobility_matrix[i, :])
            if row_sum > 0:
                self.mobility_matrix[i, :] = (self.mobility_matrix[i, :] / row_sum) * self.mobility_rate

    def simulate(self, initial_infected_zone: int = 0, initial_infected_count: int = 10, days: int = 60) -> EpidemicSpreadResult:
        """Run day-by-day spatial ODE/difference step simulation."""
        S = self.populations.copy()
        E = np.zeros(self.n_zones)
        I = np.zeros(self.n_zones)
        R = np.zeros(self.n_zones)

        init_z = max(0, min(self.n_zones - 1, initial_infected_zone))
        I[init_z] = min(S[init_z], float(initial_infected_count))
        S[init_z] -= I[init_z]

        s_hist, e_hist, i_hist, r_hist = [], [], [], []

        for day in range(days):
            s_hist.append(int(np.sum(S)))
            e_hist.append(int(np.sum(E)))
            i_hist.append(int(np.sum(I)))
            r_hist.append(int(np.sum(R)))

            # Effective infectious force including mobile contacts
            effective_I = I + np.dot(self.mobility_matrix, I)

            new_exposed = (self.beta * S * effective_I) / np.maximum(1.0, self.populations)
            new_exposed = np.minimum(S, new_exposed)

            new_infectious = self.sigma * E
            new_infectious = np.minimum(E, new_infectious)

            new_recovered = self.gamma * I
            new_recovered = np.minimum(I, new_recovered)

            S -= new_exposed
            E += new_exposed - new_infectious
            I += new_infectious - new_recovered
            R += new_recovered

        peak_day = int(np.argmax(i_hist))
        peak_cnt = int(np.max(i_hist))
        tot_pop = float(np.sum(self.populations))
        tot_inf = float(r_hist[-1] + i_hist[-1] + e_hist[-1])
        attack_rate = tot_inf / max(1.0, tot_pop)

        return EpidemicSpreadResult(
            timesteps=days,
            susceptible_history=s_hist,
            exposed_history=e_hist,
            infectious_history=i_hist,
            recovered_history=r_hist,
            peak_infectious_day=peak_day,
            peak_infectious_count=peak_cnt,
            final_attack_rate=attack_rate,
        )
