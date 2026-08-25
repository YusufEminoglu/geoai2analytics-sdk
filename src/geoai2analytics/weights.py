# -*- coding: utf-8 -*-
"""
Spatial Weights Matrices (Contiguity, Distance Bands, KNN, and Kernel Decays).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpatialWeights:
    """Spatial Weights Matrix container with spatial lag and standardization operators."""

    matrix: np.ndarray  # Shape (N, N)
    n: int
    is_row_standardized: bool = True

    def __post_init__(self) -> None:
        if self.matrix.shape != (self.n, self.n):
            raise ValueError(
                f"Matrix shape {self.matrix.shape} must match (n, n) = ({self.n}, {self.n})"
            )

    def lag(self, y: np.ndarray | list[float]) -> np.ndarray:
        """Compute the spatial lag W*y for vector y."""
        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != self.n:
            raise ValueError(f"Array length {arr.shape} must match spatial units n={self.n}")
        return np.dot(self.matrix, arr)

    @property
    def s0(self) -> float:
        """Sum of all weights in the matrix."""
        return float(np.sum(self.matrix))

    @property
    def s1(self) -> float:
        """S1 measure for analytical Moran variance."""
        return float(0.5 * np.sum((self.matrix + self.matrix.T) ** 2))

    @property
    def s2(self) -> float:
        """S2 measure for analytical Moran variance."""
        row_sums = np.sum(self.matrix, axis=1)
        col_sums = np.sum(self.matrix, axis=0)
        return float(np.sum((row_sums + col_sums) ** 2))


def row_standardize(matrix: np.ndarray) -> np.ndarray:
    """Row-standardize a square weights matrix (each row sums to 1.0; zero-sum rows remain 0)."""
    mat = matrix.astype(np.float64, copy=True)
    np.fill_diagonal(mat, 0.0)
    row_sums = np.sum(mat, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = np.where(row_sums > 0, mat / row_sums, 0.0)
    return standardized


def knn_weights(
    coords: np.ndarray | list[tuple[float, float]], k: int = 5, standardize: bool = True
) -> SpatialWeights:
    """Construct k-Nearest Neighbor (KNN) spatial weights matrix."""
    pts = np.asarray(coords, dtype=np.float64)
    n = pts.shape[0]
    if n <= k:
        k = max(1, n - 1)

    # Compute Euclidean distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)

    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        nearest_idx = np.argsort(dist[i])[:k]
        matrix[i, nearest_idx] = 1.0

    if standardize:
        matrix = row_standardize(matrix)

    return SpatialWeights(matrix=matrix, n=n, is_row_standardized=standardize)


def distance_band_weights(
    coords: np.ndarray | list[tuple[float, float]],
    threshold: float | None = None,
    kernel: str = "binary",  # binary, gaussian, bisquare, exponential
    standardize: bool = True,
) -> SpatialWeights:
    """Construct distance-band or continuous distance-decay spatial weights."""
    pts = np.asarray(coords, dtype=np.float64)
    n = pts.shape[0]

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))

    if threshold is None:
        # Default to max 1-NN distance to ensure no islands
        dist_no_diag = dist.copy()
        np.fill_diagonal(dist_no_diag, np.inf)
        min_dists = np.min(dist_no_diag, axis=1)
        threshold = float(np.max(min_dists) * 1.05)

    matrix = np.zeros((n, n), dtype=np.float64)

    if kernel == "binary":
        matrix = np.where((dist <= threshold) & (dist > 0), 1.0, 0.0)
    elif kernel == "gaussian":
        matrix = np.exp(-0.5 * (dist / threshold) ** 2)
        np.fill_diagonal(matrix, 0.0)
    elif kernel == "bisquare":
        mask = (dist <= threshold) & (dist > 0)
        matrix = np.where(mask, (1.0 - (dist / threshold) ** 2) ** 2, 0.0)
    elif kernel == "exponential":
        matrix = np.exp(-dist / threshold)
        np.fill_diagonal(matrix, 0.0)
    else:
        raise ValueError(
            f"Unknown kernel '{kernel}'. Choose from 'binary', 'gaussian', 'bisquare', 'exponential'."
        )

    if standardize:
        matrix = row_standardize(matrix)

    return SpatialWeights(matrix=matrix, n=n, is_row_standardized=standardize)


def grid_weights(
    rows: int, cols: int, contiguity: str = "queen", standardize: bool = True
) -> SpatialWeights:
    """Construct regular lattice grid spatial weights (Queen or Rook contiguity)."""
    n = rows * cols
    matrix = np.zeros((n, n), dtype=np.float64)

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            # Rook neighbors (North, South, East, West)
            rook = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            for nr, nc in rook:
                if 0 <= nr < rows and 0 <= nc < cols:
                    j = nr * cols + nc
                    matrix[i, j] = 1.0

            # Queen diagonals
            if contiguity.lower() == "queen":
                diag = [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]
                for nr, nc in diag:
                    if 0 <= nr < rows and 0 <= nc < cols:
                        j = nr * cols + nc
                        matrix[i, j] = 1.0

    if standardize:
        matrix = row_standardize(matrix)

    return SpatialWeights(matrix=matrix, n=n, is_row_standardized=standardize)
