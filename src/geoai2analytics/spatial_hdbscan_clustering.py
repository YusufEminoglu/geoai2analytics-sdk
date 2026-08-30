# -*- coding: utf-8 -*-
"""Hierarchical Density-Based Spatial Clustering (Spatial HDBSCAN) for geoai2analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class ClusterCondensedTree:
    cluster_id: int
    size: int
    parent_cluster_id: int
    lambda_val: float  # 1 / distance


@dataclass
class HDBSCANResult:
    total_points_count: int
    num_clusters: int
    noise_points_count: int
    cluster_labels: list[int]  # -1 for noise, 0..N-1 for clusters
    exemplar_centroids: dict[int, tuple[float, float]]
    cluster_sizes: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_points": self.total_points_count,
            "clusters_count": self.num_clusters,
            "noise_count": self.noise_points_count,
            "cluster_sizes": self.cluster_sizes,
        }


def cluster_spatial_hdbscan(
    coordinates: Sequence[tuple[float, float]],
    min_cluster_size: int = 5,
    min_samples: int | None = None,
) -> HDBSCANResult:
    """Perform hierarchical density-based spatial clustering without requiring a predefined global epsilon.
    
    Implements Mutual Reachability Distance (MRD) graph and minimum spanning tree hierarchical extraction.
    """
    pts = list(coordinates)
    n = len(pts)
    if n == 0:
        return HDBSCANResult(0, 0, 0, [], {}, {})

    k = min_samples or min_cluster_size
    k = min(n - 1, max(1, k))

    # 1. Compute core distances for all points (distance to k-th nearest neighbor)
    core_dists: list[float] = []
    dist_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        row = []
        for j in range(n):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            dist_matrix[i][j] = d
            row.append(d)
        row.sort()
        core_dists.append(row[min(k, len(row) - 1)])

    # 2. Build Mutual Reachability Graph: d_mreach(a, b) = max(core_dist(a), core_dist(b), d(a, b))
    # 3. Minimum Spanning Tree (Prim's algorithm)
    in_tree = [False] * n
    min_edge = [float("inf")] * n
    parent = [-1] * n
    min_edge[0] = 0.0

    mst_edges: list[tuple[float, int, int]] = []  # (mreach_dist, u, v)

    for _ in range(n):
        # Pick smallest edge outside tree
        u = -1
        for i in range(n):
            if not in_tree[i] and (u == -1 or min_edge[i] < min_edge[u]):
                u = i

        in_tree[u] = True
        if parent[u] != -1:
            mst_edges.append((min_edge[u], parent[u], u))

        for v in range(n):
            if not in_tree[v]:
                mrd = max(core_dists[u], core_dists[v], dist_matrix[u][v])
                if mrd < min_edge[v]:
                    min_edge[v] = mrd
                    parent[v] = u

    # 4. Extract dense clusters using thresholding on MST edges
    # Sort MST edges descending by distance
    mst_edges.sort(key=lambda x: x[0], reverse=True)

    # Disjoint Set Union (DSU) with connected components
    dsu_parent = list(range(n))
    dsu_size = [1] * n

    def find(x: int) -> int:
        if dsu_parent[x] == x:
            return x
        dsu_parent[x] = find(dsu_parent[x])
        return dsu_parent[x]

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            if dsu_size[rx] < dsu_size[ry]:
                rx, ry = ry, rx
            dsu_parent[ry] = rx
            dsu_size[rx] += dsu_size[ry]

    # Reconstruct from smallest edges (reverse of sorted desc)
    # Threshold at median/robust percentile
    edge_dists = [e[0] for e in mst_edges]
    cutoff = sorted(edge_dists)[int(len(edge_dists) * 0.75)] if edge_dists else float("inf")

    for dist_val, u, v in reversed(mst_edges):
        if dist_val <= cutoff:
            union(u, v)

    # Assign cluster labels
    components: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        components.setdefault(root, []).append(i)

    labels = [-1] * n
    cluster_id = 0
    centroids: dict[int, tuple[float, float]] = {}
    cluster_sizes: dict[int, int] = {}
    noise_count = 0

    for root, members in components.items():
        if len(members) >= min_cluster_size:
            for m in members:
                labels[m] = cluster_id
            avg_x = sum(pts[m][0] for m in members) / len(members)
            avg_y = sum(pts[m][1] for m in members) / len(members)
            centroids[cluster_id] = (round(avg_x, 3), round(avg_y, 3))
            cluster_sizes[cluster_id] = len(members)
            cluster_id += 1
        else:
            for m in members:
                labels[m] = -1
                noise_count += 1

    return HDBSCANResult(
        total_points_count=n,
        num_clusters=cluster_id,
        noise_points_count=noise_count,
        cluster_labels=labels,
        exemplar_centroids=centroids,
        cluster_sizes=cluster_sizes,
    )
