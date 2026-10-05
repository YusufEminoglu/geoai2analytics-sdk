# -*- coding: utf-8 -*-
"""Directional Anisotropic Spatial DBSCAN Clustering for geoai2analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class SpatialClusterResult:
    total_points: int
    num_clusters: int
    noise_points_count: int
    cluster_labels: list[int]  # -1 for noise, 0..k for clusters
    cluster_centroids: dict[int, tuple[float, float]]
    cluster_sizes: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_points": self.total_points,
            "num_clusters": self.num_clusters,
            "noise_points": self.noise_points_count,
            "cluster_sizes": self.cluster_sizes,
        }


def cluster_spatial_points_dbscan(
    coords: Sequence[tuple[float, float]],
    eps_distance: float = 50.0,
    min_samples: int = 4,
    anisotropy_ratio: float = 1.0,  # 1.0 = isotropic circular, >1.0 = stretched along primary axis
    anisotropy_angle_deg: float = 0.0,
) -> SpatialClusterResult:
    """Perform density-based spatial clustering (DBSCAN) with optional directional elliptical metrics."""
    pts = np.asarray(coords, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return SpatialClusterResult(0, 0, 0, [], {}, {})

    # Transform coordinates according to anisotropy ellipse
    theta = math.radians(anisotropy_angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Rotation matrix
    rot_pts = np.zeros_like(pts)
    rot_pts[:, 0] = pts[:, 0] * cos_t + pts[:, 1] * sin_t
    rot_pts[:, 1] = -pts[:, 0] * sin_t + pts[:, 1] * cos_t

    # Scale Y axis by anisotropy ratio
    rot_pts[:, 1] *= anisotropy_ratio

    # Compute Euclidean distance matrix in transformed space
    diff = rot_pts[:, np.newaxis, :] - rot_pts[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    curr_cluster = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = np.where(dist_matrix[i] <= eps_distance)[0].tolist()

        if len(neighbors) < min_samples:
            labels[i] = -1  # Noise
        else:
            labels[i] = curr_cluster
            # Expand cluster
            queue = [nb for nb in neighbors if nb != i]
            while queue:
                nb_idx = queue.pop(0)
                if not visited[nb_idx]:
                    visited[nb_idx] = True
                    nb_neighbors = np.where(dist_matrix[nb_idx] <= eps_distance)[0].tolist()
                    if len(nb_neighbors) >= min_samples:
                        for item in nb_neighbors:
                            if item not in queue and not visited[item]:
                                queue.append(item)

                if labels[nb_idx] == -1:
                    labels[nb_idx] = curr_cluster

            curr_cluster += 1

    # Compute centroids and cluster sizes
    unique_lbls = [lbl for lbl in set(labels) if lbl != -1]
    centroids: dict[int, tuple[float, float]] = {}
    sizes: dict[int, int] = {}

    for cl_id in unique_lbls:
        idx_mask = (labels == cl_id)
        c_pts = pts[idx_mask]
        mean_x = float(np.mean(c_pts[:, 0]))
        mean_y = float(np.mean(c_pts[:, 1]))
        centroids[int(cl_id)] = (round(mean_x, 3), round(mean_y, 3))
        sizes[int(cl_id)] = int(np.sum(idx_mask))

    noise_cnt = int(np.sum(labels == -1))

    return SpatialClusterResult(
        total_points=n,
        num_clusters=len(unique_lbls),
        noise_points_count=noise_cnt,
        cluster_labels=labels.tolist(),
        cluster_centroids=centroids,
        cluster_sizes=sizes,
    )
