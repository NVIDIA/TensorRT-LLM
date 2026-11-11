# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Data-oblivious TurboQuant parameters: rotations, QJL projections, and Lloyd-Max codebooks.

TurboQuant (Zandieh et al., arXiv 2504.19874) rotates every vector with a random orthogonal
matrix, after which each coordinate of a unit vector in R^n follows the density
``f(u) ~ (1 - u^2)^((n - 3) / 2)`` on [-1, 1] whatever the input was. Each rotated coordinate is
then quantized with the Lloyd-Max codebook of that density. Codebooks here are expressed in
``sqrt(n)``-scaled units, where the density is close to N(0, 1).

Every matrix is a pure function of (seed, dimension, purpose), generated in float64 on the CPU,
so equal configurations produce equal cold-page bytes in every process.
"""

import functools
import math

import torch

ROTATION = 0
"""Purpose of the Haar rotation of a group with a single instance (or of the outlier group)."""
REGULAR_ROTATION = 1
"""Purpose of the Haar rotation of the regular-channel group of a fractional-bit setting."""
QJL_PROJECTION = 2
"""Purpose of the i.i.d. Gaussian QJL projection of TurboQuant-prod."""

_MAX_SUPPORT = 12.0
_GRID_POINTS = 1 << 18
_MAX_ITERATIONS = 2000


def _generator(seed: int, dim: int, purpose: int) -> torch.Generator:
    mixed = (
        seed * 0x9E3779B97F4A7C15 + dim * 0xBF58476D1CE4E5B9 + purpose * 0x94D049BB133111EB
    ) % (1 << 63)
    return torch.Generator(device="cpu").manual_seed(mixed)


@functools.lru_cache(maxsize=None)
def haar_rotation(seed: int, dim: int, purpose: int = ROTATION) -> torch.Tensor:
    """Return a Haar-distributed orthogonal ``dim x dim`` float64 matrix."""

    gaussian = torch.randn(dim, dim, generator=_generator(seed, dim, purpose), dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    # Fixing the signs of R's diagonal makes Q exactly Haar distributed.
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


@functools.lru_cache(maxsize=None)
def qjl_projection(seed: int, dim: int) -> torch.Tensor:
    """Return the ``dim x dim`` float64 QJL matrix with i.i.d. N(0, 1) entries."""

    return torch.randn(
        dim, dim, generator=_generator(seed, dim, QJL_PROJECTION), dtype=torch.float64
    )


@functools.lru_cache(maxsize=None)
def lloyd_max_codebook(dim: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (centroids, thresholds) of the optimal ``bits``-bit scalar quantizer.

    The quantizer minimizes the mean squared error of one coordinate ``v = sqrt(dim) * u`` of a
    uniformly random unit vector ``u`` in R^dim. ``centroids`` holds ``2**bits`` ascending float64
    values; ``thresholds`` holds the ``2**bits - 1`` decision boundaries between them.
    """

    if dim < 4 or not 1 <= bits <= 8:
        raise ValueError(f"unsupported TurboQuant codebook: dim={dim}, bits={bits}")
    support = min(math.sqrt(dim), _MAX_SUPPORT)
    # Midpoint rule on a fine grid of the (unnormalized) coordinate density.
    edges = torch.linspace(-support, support, _GRID_POINTS + 1, dtype=torch.float64)
    grid = 0.5 * (edges[1:] + edges[:-1])
    log_density = 0.5 * (dim - 3) * torch.log1p(-(grid * grid) / dim)
    density = torch.exp(log_density - log_density.max())
    cumulative_mass = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(density, 0)])
    cumulative_moment = torch.cat(
        [torch.zeros(1, dtype=torch.float64), torch.cumsum(density * grid, 0)]
    )

    levels = 1 << bits
    # Start from Gaussian quantiles, then run Lloyd iterations to convergence.
    probabilities = (torch.arange(levels, dtype=torch.float64) + 0.5) / levels
    centroids = math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)
    for _ in range(_MAX_ITERATIONS):
        thresholds = 0.5 * (centroids[1:] + centroids[:-1])
        cuts = torch.searchsorted(grid, thresholds)
        bounds = torch.cat([torch.zeros(1, dtype=torch.long), cuts, torch.tensor([grid.numel()])])
        mass = cumulative_mass[bounds[1:]] - cumulative_mass[bounds[:-1]]
        moment = cumulative_moment[bounds[1:]] - cumulative_moment[bounds[:-1]]
        updated = moment / mass
        if torch.max(torch.abs(updated - centroids)) < 1e-12:
            centroids = updated
            break
        centroids = updated
    # The density is even: make the codebook exactly symmetric (the kernels rely on it).
    centroids = 0.5 * (centroids - centroids.flip(0))
    return centroids, 0.5 * (centroids[1:] + centroids[:-1])


@functools.lru_cache(maxsize=None)
def codebook_distortion(dim: int, bits: int) -> float:
    """Expected ``||x - x_hat||^2 / ||x||^2`` of the codebook (Theorem 1 of the paper)."""

    centroids, thresholds = lloyd_max_codebook(dim, bits)
    support = min(math.sqrt(dim), _MAX_SUPPORT)
    edges = torch.linspace(-support, support, _GRID_POINTS + 1, dtype=torch.float64)
    grid = 0.5 * (edges[1:] + edges[:-1])
    density = torch.exp(0.5 * (dim - 3) * torch.log1p(-(grid * grid) / dim))
    density = density / density.sum()
    quantized = centroids[torch.searchsorted(thresholds, grid)]
    return float((density * (grid - quantized) ** 2).sum())
