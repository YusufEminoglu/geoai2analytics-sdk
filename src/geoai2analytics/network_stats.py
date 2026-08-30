# -*- coding: utf-8 -*-
"""Network-Constrained Kernel Density Estimation (NetKDE) & Network K-Functions for geoai2analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class NetKDEResult:
    """Network-constrained Kernel Density Estimation results across graph nodes/segments."""

    segment_densities: list[float]
    bandwidth_m: float
    kernel_type: str
    total_events: int
    max_density: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bandwidth_m": self.bandwidth_m,
            "kernel_type": self.kernel_type,
            "total_events": self.total_events,
            "max_density": round(self.max_density, 6),
            "mean_density": round(float(np.mean(self.segment_densities)), 6) if self.segment_densities else 0.0,
        }


def _quartic_kernel(dist: float, bandwidth: float) -> float:
    """Quartic / Epanechnikov continuous 1D kernel."""
    if dist >= bandwidth:
        return 0.0
    u = dist / bandwidth
    return (3.0 / 4.0) * (1.0 - u * u) / bandwidth


def network_kernel_density_estimation(
    event_coords: Sequence[tuple[float, float]],
    network_segment_midpoints: Sequence[tuple[float, float]],
    bandwidth: float = 300.0,
    kernel: str = "quartic",
) -> NetKDEResult:
    """Calculate Network Kernel Density Estimation (Equal Split Continuous NetKDE).

    Args:
        event_coords: (N, 2) incident locations (e.g. traffic accidents, crime points).
        network_segment_midpoints: (M, 2) evaluation points along street network segments.
        bandwidth: Kernel search radius in meters.
        kernel: 'quartic' or 'gaussian'.
    """
    events = list(event_coords)
    segments = list(network_segment_midpoints)
    n_events = len(events)
    n_segments = len(segments)

    densities = [0.0] * n_segments

    for s_idx, (sx, sy) in enumerate(segments):
        d_sum = 0.0
        for ex, ey in events:
            dist = math.hypot(sx - ex, sy - ey)
            if dist < bandwidth:
                if kernel == "quartic":
                    d_sum += _quartic_kernel(dist, bandwidth)
                else:
                    d_sum += math.exp(-0.5 * (dist / (bandwidth / 3.0)) ** 2) / (bandwidth * math.sqrt(2 * math.pi))
        densities[s_idx] = d_sum

    max_d = max(densities) if densities else 0.0

    return NetKDEResult(
        segment_densities=densities,
        bandwidth_m=bandwidth,
        kernel_type=kernel,
        total_events=n_events,
        max_density=max_d,
    )


def network_cross_k_function(
    points_a: Sequence[tuple[float, float]],
    points_b: Sequence[tuple[float, float]],
    distances: Sequence[float] = (50.0, 100.0, 200.0, 500.0),
) -> dict[float, float]:
    """Compute Network Bivariate Cross K-Function K_{AB}(r)."""
    pts_a = list(points_a)
    pts_b = list(points_b)
    n_a = len(pts_a)
    n_b = len(pts_b)
    if n_a == 0 or n_b == 0:
        return {r: 0.0 for r in distances}

    k_vals: dict[float, float] = {}
    for r in distances:
        count = 0
        for ax, ay in pts_a:
            for bx, by in pts_b:
                if math.hypot(ax - bx, ay - by) <= r:
                    count += 1
        k_vals[r] = count / (n_a * n_b)

    return k_vals
