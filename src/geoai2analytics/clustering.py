# -*- coding: utf-8 -*-
"""
Spatial Clustering, Regionalization, and Segmentation:
SKATER (Minimum Spanning Tree Spatial Regionalization), Spatial DBSCAN,
Spatial Gaussian Mixture Models (GMM), and Multivariate Spatial Clustering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .weights import SpatialWeights


@dataclass
class SKATERResult:
    """SKATER spatially constrained regionalization output."""

    labels: np.ndarray
    n_clusters: int
    total_ssd: float
    between_ssd: float


@dataclass
class SpatialClusteringResult:
    """Spatial clustering and regionalization partition output."""

    labels: np.ndarray
    n_clusters: int
    n_noise: int
    cluster_sizes: dict[int, int]


# ---------------------------------------------------------------------------
# 1. SKATER (Spatial 'K'luster Analysis by Tree Edge Removal)
# ---------------------------------------------------------------------------
def skater(
    X: np.ndarray | list[list[float]],
    weights: SpatialWeights,
    n_clusters: int = 5,
) -> SKATERResult:
    """Perform spatially constrained regionalization using Minimum Spanning Tree edge cutting (Assunção et al.)."""
    X_mat = np.asarray(X, dtype=np.float64)
    if X_mat.ndim == 1:
        X_mat = X_mat[:, np.newaxis]
    n = len(X_mat)
    if n_clusters >= n:
        n_clusters = max(1, n - 1)

    # Compute attribute dissimilarities along spatial graph edges
    w_mat = weights.matrix
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if w_mat[i, j] > 0 or w_mat[j, i] > 0:
                dist_attr = float(np.linalg.norm(X_mat[i] - X_mat[j]))
                edges.append((dist_attr, i, j))

    # Kruskal's Algorithm to build Minimum Spanning Tree (MST)
    edges.sort()
    parent = list(range(n))

    def find(u: int) -> int:
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u: int, v: int) -> bool:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            return True
        return False

    mst_adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    mst_edges: list[tuple[float, int, int]] = []

    for d, u, v in edges:
        if union(u, v):
            mst_adj[u].append((v, d))
            mst_adj[v].append((u, d))
            mst_edges.append((d, u, v))

    # Prune top (n_clusters - 1) highest-cost edges in the MST to form clusters
    mst_edges.sort(reverse=True)
    cut_edges = set(mst_edges[: (n_clusters - 1)])

    # Breadth-first search on pruned MST to assign cluster labels
    pruned_adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for d, u, v in mst_edges:
        if (d, u, v) not in cut_edges:
            pruned_adj[u].append(v)
            pruned_adj[v].append(u)

    labels = -np.ones(n, dtype=np.int32)
    cur_label = 0

    for i in range(n):
        if labels[i] == -1:
            queue = [i]
            labels[i] = cur_label
            while queue:
                curr = queue.pop(0)
                for neighbor in pruned_adj[curr]:
                    if labels[neighbor] == -1:
                        labels[neighbor] = cur_label
                        queue.append(neighbor)
            cur_label += 1

    # Total Sum of Squares Decomposition
    grand_mean = np.mean(X_mat, axis=0)
    total_ssd = float(np.sum((X_mat - grand_mean) ** 2))

    between_ssd = 0.0
    for k in range(cur_label):
        mask = labels == k
        if np.any(mask):
            c_mean = np.mean(X_mat[mask], axis=0)
            between_ssd += float(np.sum(mask) * np.sum((c_mean - grand_mean) ** 2))

    return SKATERResult(
        labels=labels,
        n_clusters=cur_label,
        total_ssd=round(total_ssd, 2),
        between_ssd=round(between_ssd, 2),
    )


# ---------------------------------------------------------------------------
# 2. Spatial DBSCAN
# ---------------------------------------------------------------------------
def spatial_dbscan(
    coords: np.ndarray | list[tuple[float, float]],
    attributes: np.ndarray | list[list[float]] | None = None,
    eps_spatial: float = 10.0,
    min_samples: int = 4,
) -> SpatialClusteringResult:
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN)."""
    pts = np.asarray(coords, dtype=np.float64)
    n = len(pts)

    # Pairwise spatial distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))

    labels = -np.ones(n, dtype=np.int32)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue

        neighbors = list(np.where(dist[i] <= eps_spatial)[0])
        if len(neighbors) < min_samples:
            labels[i] = -1  # Noise
            continue

        labels[i] = cluster_id
        seed_set = [idx for idx in neighbors if idx != i]

        while seed_set:
            curr = seed_set.pop(0)
            if labels[curr] == -1:
                labels[curr] = cluster_id

            if labels[curr] != -1:
                continue

            labels[curr] = cluster_id
            curr_neighbors = list(np.where(dist[curr] <= eps_spatial)[0])
            if len(curr_neighbors) >= min_samples:
                seed_set.extend(
                    [idx for idx in curr_neighbors if idx not in seed_set and labels[idx] == -1]
                )

        cluster_id += 1

    noise_count = int(np.sum(labels == -1))
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    sizes = {int(c): int(np.sum(labels == c)) for c in unique_clusters}

    return SpatialClusteringResult(
        labels=labels,
        n_clusters=len(unique_clusters),
        n_noise=noise_count,
        cluster_sizes=sizes,
    )


# ---------------------------------------------------------------------------
# 3. Spatial Gaussian Mixture Model (Spatial GMM)
# ---------------------------------------------------------------------------
def spatial_gmm(
    coords: np.ndarray | list[tuple[float, float]],
    attributes: np.ndarray | list[list[float]],
    n_components: int = 4,
    coord_weight: float = 0.5,
    max_iter: int = 100,
    seed: int = 42,
) -> SpatialClusteringResult:
    """Gaussian Mixture Model fusing continuous spatial coordinates with multidimensional attributes."""
    pts = np.asarray(coords, dtype=np.float64)
    attrs = np.asarray(attributes, dtype=np.float64)
    if attrs.ndim == 1:
        attrs = attrs[:, np.newaxis]

    # Normalize coordinates and attributes to unit scale
    pts_norm = (pts - np.mean(pts, axis=0)) / (np.std(pts, axis=0) + 1e-9)
    attrs_norm = (attrs - np.mean(attrs, axis=0)) / (np.std(attrs, axis=0) + 1e-9)

    X_fused = np.column_stack([coord_weight * pts_norm, (1.0 - coord_weight) * attrs_norm])
    n, d = X_fused.shape

    rng = np.random.default_rng(seed)
    # Initialize means with random sample
    means = X_fused[rng.choice(n, size=n_components, replace=False)].copy()
    covs = [np.eye(d) for _ in range(n_components)]
    weights = np.ones(n_components) / n_components

    labels = np.zeros(n, dtype=np.int32)

    # EM Algorithm
    for _ in range(max_iter):
        # E-step: Responsibilities
        resp = np.zeros((n, n_components))
        for k in range(n_components):
            diff = X_fused - means[k]
            inv_cov = np.linalg.pinv(covs[k] + np.eye(d) * 1e-5)
            exp_term = np.exp(-0.5 * np.sum(np.dot(diff, inv_cov) * diff, axis=1))
            det = max(1e-9, float(np.linalg.det(covs[k] + np.eye(d) * 1e-5)))
            resp[:, k] = weights[k] * exp_term / math.sqrt(det)

        resp_sum = np.sum(resp, axis=1, keepdims=True) + 1e-12
        resp /= resp_sum

        # M-step: Update parameters
        Nk = np.sum(resp, axis=0)
        weights = Nk / n
        for k in range(n_components):
            if Nk[k] > 0:
                means[k] = np.sum(resp[:, k, np.newaxis] * X_fused, axis=0) / Nk[k]
                diff = X_fused - means[k]
                covs[k] = np.dot((resp[:, k, np.newaxis] * diff).T, diff) / Nk[k]

    labels = np.argmax(resp, axis=1)
    sizes = {int(c): int(np.sum(labels == c)) for c in range(n_components)}

    return SpatialClusteringResult(
        labels=labels,
        n_clusters=n_components,
        n_noise=0,
        cluster_sizes=sizes,
    )
