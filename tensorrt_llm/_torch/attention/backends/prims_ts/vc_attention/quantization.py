# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VC-Attention quantization of Q, K and V for the prims_ts context kernel.

VC-Attention (arXiv 2609.15810) adds two changes on the value side of
attention: a token permutation that groups similar value rows into the same
128-token K/V tile, and per-tile value smoothing. Each V tile is stored as E4M3
residuals around its tile mean and the kernel restores the mean inside the
online softmax, so the residual quantization error no longer carries the value
DC component. P is E4M3 through the ExpCast path in both recipes of the Q/K
side. VC-Attention-QK16 keeps Q and K in bf16. VC-Attention-QK8 centres K by
its per-head channel mean, rotates Q and K by the normalised 128-point Hadamard
matrix and quantizes them to E4M3 with one scale per 128-token Q block and per
128-token K tile, as the paper does.

* ``vc_token_permutation`` runs an online k-means over value rows, one
  permutation per batch and head, applied to K and V. Attention is invariant
  to a common key and value permutation.
* ``vc_quantize`` permutes K, subtracts the V tile means, quantizes the
  residuals per channel to E4M3 and packs the kernel operands with the CuTe
  DSL kernels of ``prepare``.
* ``vc_quantize_fp8`` does the same plus the E4M3 Q/K operands and scales.
* ``vc_quantize_repair`` writes V repair rows instead of tile means, for
  either recipe.
* ``VCAttentionPreprocessor`` applies the paper's V-Smooth schedule over the
  denoising steps on top of the two.

The operand contract the kernel consumes is defined in
``flashinfer.attention.prims_ts.vc_attention``.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import torch

from . import (
    VC_HEAD_DIM,
    VCAttentionParams,
    vc_repair_tiles,
)

# V-Smooth schedule. The paper groups and demeans on the first quarter of the
# denoising steps and keeps the permutation afterwards. The refresh cadence
# inside that window and the k-means size are not fixed by the paper. 64
# clusters sort the values into 128-token tiles of similar rows while keeping
# the online k-means a small fraction of the attention call, and 3 iterations
# are enough because every refresh warm-starts from the previous centroids.
VC_SMOOTH_STEP_FRACTION = 0.25
VC_PERM_REFRESH_EVERY = 4
VC_KMEANS_CLUSTERS = 64
VC_KMEANS_ITERS = 3


@dataclass(frozen=True)
class VCAttentionOperands:
    """Everything one VC-Attention forward needs, in kernel layout.

    ``k`` is the input in permuted token order; ``v`` is E4M3 ``[B, S, H, D]``
    in the same order holding the tile residuals ``(V[perm] - mean) / v_scale``.
    Under VC-Attention-QK8 ``q`` and ``k`` are the rotated E4M3 operands and
    ``q_scale`` and ``k_scale`` their per-block scales; under QK16 those
    three are ``None`` and the caller's bf16 ``q`` is used as is. With
    ``repair_tiles`` the token order is the input order, ``mean``, ``mu`` and
    ``perm`` are ``None`` and the K/V rows are the whole 128-token tiles, the
    repair tiles of key copies and E4M3 value residuals, then the zero-padded
    partial last tile.
    """

    k: torch.Tensor
    v: torch.Tensor
    v_scale: (
        torch.Tensor
    )  # [B, H, D] fp32 per-channel E4M3 residual scale (feeds output_scale)
    mean: (
        torch.Tensor | None
    )  # [B, H, num_kv_tiles, D] fp32 tile means (permuted order)
    mu: (
        torch.Tensor | None
    )  # [B, H, ceil(num_kv_tiles / 16), 16, 256] bf16 packed kernel operands
    perm: torch.Tensor | None  # [B, H, S_k] int32/int64 token permutation
    demean: bool = True  # whether the means were subtracted (else zero)
    repair_tiles: int = 0
    seq_len_kv: int | None = None  # K/V tokens before the repair tiles and padding
    q: torch.Tensor | None = None  # [B, S_q, H, D] E4M3 rotated Q (QK8)
    q_scale: torch.Tensor | None = None  # [H, B, ceil(S_q / 128)] fp32
    k_scale: torch.Tensor | None = None  # [H, B, ceil(S_k / 128)] fp32

    @property
    def params(self) -> "VCAttentionParams":
        """The per-run operands ``BatchPrefillTSWrapper.run`` takes as ``vc``."""
        return VCAttentionParams(
            q_scale=self.q_scale,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            tile_means=self.mu,
        )


@functools.lru_cache(maxsize=None)
def _hadamard_matrix_cached(d: int, device: str) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(d)).to(device)


def hadamard_matrix(d: int, device: torch.device) -> torch.Tensor:
    """Normalised Sylvester Hadamard matrix ``[d, d]`` (``d`` a power of two)."""
    if d & (d - 1):
        raise ValueError("head_dim must be a power of two for the Hadamard rotation")
    return _hadamard_matrix_cached(d, str(device))


@torch.no_grad()
def vc_token_permutation(
    v: torch.Tensor,
    *,
    num_clusters: int | None = None,
    iters: int = VC_KMEANS_ITERS,
    generator: torch.Generator | None = None,
    init_centroids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group similar value rows by k-means per batch and head, then argsort the labels.

    Returns the int64 permutation ``[B, H, S]`` and the ``[B*H, k, D]`` centroids.
    ``V[b, perm[b, h], h]`` lists the tokens cluster by cluster, so 128-token K/V
    tiles hold similar values and their tile means remove most of the value
    energy. ``init_centroids`` from a previous call warm-starts the clustering,
    as the paper does when the grouping is refreshed a few denoising steps later.
    """
    b, s, h, d = v.shape
    if num_clusters is None:
        num_clusters = VC_KMEANS_CLUSTERS
    # Seeding draws distinct value rows, so a short sequence caps the clusters.
    num_clusters = min(num_clusters, s)
    n_groups = b * h
    # Work on a few heads at a time so the fp32 copy of V and the distance
    # matrix each stay under 0.75 GB.
    per_group = 4 * s * max(d, num_clusters)
    chunk_h = max(1, min(h, int(0.75e9 // per_group)))
    perm_out = torch.empty((b, h, s), dtype=torch.int64, device=v.device)
    cent_out = torch.empty(
        (n_groups, num_clusters, d), dtype=torch.float32, device=v.device
    )
    warm = init_centroids is not None and tuple(init_centroids.shape) == (
        n_groups,
        num_clusters,
        d,
    )
    for bi in range(b):
        for h0 in range(0, h, chunk_h):
            h1 = min(h, h0 + chunk_h)
            gsz = h1 - h0
            g0 = bi * h + h0
            x = v[bi, :, h0:h1, :].permute(1, 0, 2).float()  # [G, S, D]
            if warm:
                centroids = init_centroids[g0 : g0 + gsz].to(x.dtype)
            else:
                idx = torch.stack(
                    [
                        torch.randperm(s, device=v.device, generator=generator)[
                            :num_clusters
                        ]
                        for _ in range(gsz)
                    ]
                )  # [G, k]
                centroids = torch.gather(
                    x, 1, idx.unsqueeze(-1).expand(-1, -1, d)
                )  # [G, k, D]
            x_sq = (x * x).sum(-1, keepdim=True)  # [G, S, 1]
            labels = None
            for _ in range(iters):
                c_sq = (centroids * centroids).sum(-1).unsqueeze(1)  # [G, 1, k]
                dist = x_sq + c_sq - 2.0 * torch.bmm(x, centroids.transpose(1, 2))
                labels = dist.argmin(dim=-1)  # [G, S]
                del dist
                counts = torch.zeros(
                    gsz, num_clusters, device=v.device, dtype=torch.float32
                )
                counts.scatter_add_(
                    1, labels, torch.ones_like(labels, dtype=torch.float32)
                )
                sums = torch.zeros_like(centroids)
                sums.scatter_add_(1, labels.unsqueeze(-1).expand(-1, -1, d), x)
                nonempty = counts > 0
                centroids = torch.where(
                    nonempty.unsqueeze(-1),
                    sums / counts.clamp_min(1.0).unsqueeze(-1),
                    centroids,
                )
            assert labels is not None
            perm_out[bi, h0:h1] = torch.argsort(labels, dim=-1, stable=True)
            cent_out[g0 : g0 + gsz] = centroids
            del x, x_sq
    return perm_out, cent_out


# ---------------------------------------------------------------------------
# Preprocessing (CuTe DSL kernels in prepare.py)
# ---------------------------------------------------------------------------
def _check_prepare_inputs(x: torch.Tensor) -> None:
    if (
        x.shape[-1] != VC_HEAD_DIM
        or x.dtype not in (torch.bfloat16, torch.float16)
        or not x.is_cuda
    ):
        raise ValueError(
            f"VC-Attention preparation needs CUDA bf16/fp16 inputs with head_dim {VC_HEAD_DIM}, "
            f"got {x.dtype} {tuple(x.shape)} on {x.device}"
        )


@torch.no_grad()
def vc_quantize(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    perm: torch.Tensor | None = None,
    kmeans_iters: int = VC_KMEANS_ITERS,
    generator: torch.Generator | None = None,
    demean: bool = True,
) -> VCAttentionOperands:
    """Turn ``[B, S, H, D]`` bf16/fp16 K/V into VC-Attention-QK16 kernel
    operands with two CuTe DSL kernels (``prepare``).

    Keys are permuted with the values and kept in their dtype. Values are
    permuted, split into 128-token tile means and residuals (``demean=False``
    keeps the means at zero), and the residuals are quantized to E4M3 with one
    scale per (batch, head, channel), returned as ``v_scale`` for the run's
    ``vc`` operands. Pass 1 gathers each permuted tile once, writes the
    permuted K rows, the V tile mean and residual amax; pass 2 writes the E4M3
    residuals and the packed bf16 mean operand. Requires CUDA bf16/fp16 inputs
    with head_dim 128.
    """
    if k.dim() != 4 or k.shape != v.shape or k.dtype != v.dtype:
        raise ValueError("k and v must be [B, S, H, D] with matching shapes and dtype")
    _check_prepare_inputs(k)
    from .prepare import vc_prepare

    if perm is None:
        perm = vc_token_permutation(v, iters=kmeans_iters, generator=generator)[0]
    k_p, v8, v_scale, mean, mu = vc_prepare(
        k.contiguous(),
        v.contiguous(),
        perm,
        demean=demean,
    )
    return VCAttentionOperands(
        k=k_p,
        v=v8,
        v_scale=v_scale,
        mean=mean,
        mu=mu if demean else None,
        perm=perm,
        demean=demean,
    )


@torch.no_grad()
def vc_quantize_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    perm: torch.Tensor | None = None,
    kmeans_iters: int = VC_KMEANS_ITERS,
    generator: torch.Generator | None = None,
    demean: bool = True,
) -> VCAttentionOperands:
    """Turn ``[B, S, H, D]`` bf16/fp16 Q/K/V into VC-Attention-QK8 kernel
    operands with four CuTe DSL kernels (``prepare``).

    Keys are centred by their per-(batch, head) channel mean over tokens
    (softmax-invariant), permuted with the values, rotated with Q by the
    normalised Hadamard matrix and quantized to E4M3 with one scale per
    128-token tile; queries take one E4M3 scale per 128-token block. Values
    are prepared as in ``vc_quantize``. Requires CUDA bf16/fp16 inputs with
    head_dim 128.
    """
    if (
        q.dim() != 4
        or k.shape != v.shape
        or q.shape[0] != k.shape[0]
        or q.shape[2:] != k.shape[2:]
        or q.dtype != k.dtype
        or k.dtype != v.dtype
    ):
        raise ValueError(
            "q, k, v must be [B, S, H, D] with matching batch, heads, head_dim and dtype"
        )
    _check_prepare_inputs(q)
    from .prepare import vc_prepare_fp8

    if perm is None:
        perm = vc_token_permutation(v, iters=kmeans_iters, generator=generator)[0]
    q8, k8, v8, q_scale, k_scale, v_scale, mean, mu = vc_prepare_fp8(
        q.contiguous(), k.contiguous(), v.contiguous(), perm, demean=demean
    )
    return VCAttentionOperands(
        k=k8,
        v=v8,
        v_scale=v_scale,
        mean=mean,
        mu=mu if demean else None,
        perm=perm,
        demean=demean,
        q=q8,
        q_scale=q_scale,
        k_scale=k_scale,
    )


@torch.no_grad()
def vc_quantize_repair(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    budget: float,
    q: torch.Tensor | None = None,
) -> VCAttentionOperands:
    """Turn ``[B, S, H, D]`` K/V into V repair operands, of VC-Attention-QK8
    with ``q``.

    Values are quantized to E4M3 in input order with one scale per batch,
    head and channel. The ``budget`` fraction of tokens with the largest
    residual energy per batch and head gets a repair row, a copy of the key
    (under QK8 re-centred, rotated and quantized with the repair tile's own
    scale) with the E4M3 residual of the value.
    """
    if k.dim() != 4 or k.shape != v.shape or k.dtype != v.dtype:
        raise ValueError("k and v must be [B, S, H, D] with matching shapes and dtype")
    if q is not None and (
        q.dim() != 4
        or q.shape[0] != k.shape[0]
        or q.shape[2:] != k.shape[2:]
        or q.dtype != k.dtype
    ):
        raise ValueError(
            "q must be [B, S, H, D] with the batch, heads, head_dim and dtype of k"
        )
    _check_prepare_inputs(k)
    from .prepare import vc_prepare_repair

    s_k = k.shape[1]
    repair_tiles = vc_repair_tiles(s_k, budget)
    repair_tokens = max(1, round(budget * s_k))
    q8, k_all, v_all, q_scale, k_scale, v_scale = vc_prepare_repair(
        None if q is None else q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        repair_tiles=repair_tiles,
        repair_tokens=repair_tokens,
    )
    return VCAttentionOperands(
        k=k_all,
        v=v_all,
        v_scale=v_scale,
        mean=None,
        mu=None,
        perm=None,
        demean=False,
        repair_tiles=repair_tiles,
        seq_len_kv=s_k,
        q=q8,
        q_scale=q_scale,
        k_scale=k_scale,
    )


class VCAttentionPreprocessor:
    """Caller-owned preparation of bf16 K/V with the paper's V-Smooth schedule.

    Value grouping and demeaning run on the first ``smooth_step_fraction`` of
    the denoising steps. Inside that window the token permutation is
    recomputed every ``perm_refresh_every`` steps (k-means warm-started from
    the previous centroids) and kept afterwards, when the plain per-channel
    E4M3 V runs with zero tile means. ``kmeans_clusters`` (``None`` = 64) and
    ``kmeans_iters`` parametrize the online k-means. Without a denoise step
    the grouping is computed once per K/V geometry and kept. With ``q`` the
    preparation is the VC-Attention-QK8 one (one E4M3 Q scale per 128 tokens).
    """

    def __init__(
        self,
        *,
        smooth_step_fraction: float = VC_SMOOTH_STEP_FRACTION,
        perm_refresh_every: int = VC_PERM_REFRESH_EVERY,
        kmeans_clusters: int | None = None,
        kmeans_iters: int = VC_KMEANS_ITERS,
    ) -> None:
        if not (0.0 <= smooth_step_fraction <= 1.0):
            raise ValueError("smooth_step_fraction must be in [0, 1]")
        if perm_refresh_every < 1:
            raise ValueError("perm_refresh_every must be at least 1")
        if kmeans_clusters is not None and kmeans_clusters < 1:
            raise ValueError("kmeans_clusters must be at least 1")
        if kmeans_iters < 1:
            raise ValueError("kmeans_iters must be at least 1")
        self.smooth_step_fraction = smooth_step_fraction
        self.perm_refresh_every = perm_refresh_every
        self.kmeans_clusters = kmeans_clusters
        self.kmeans_iters = kmeans_iters
        # Per (batch, seq_len_k, num_kv_heads): perm, centroids, step of the last refresh.
        self._groupings: dict[tuple[int, int, int], dict[str, Any]] = {}

    @torch.no_grad()
    @torch.compiler.disable
    def prepare(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        q: torch.Tensor | None = None,
        denoise_step: tuple[int, int] | None = None,
    ) -> VCAttentionOperands:
        """Return the operands of one run for ``[B, S, H, D]`` bf16 K/V, and Q
        under VC-Attention-QK8.

        ``denoise_step`` is the sampler's ``(step_index, num_steps)``. Kept
        out of torch.compile so the step index and the k-means refresh do not
        recompile the caller per step.
        """
        key = (k.shape[0], k.shape[1], k.shape[2])
        entry = self._groupings.get(key)
        smooth = True
        refresh = entry is None
        if denoise_step is not None:
            step_idx, num_steps = denoise_step
            smooth = step_idx < math.ceil(self.smooth_step_fraction * num_steps)
            # Step 0 starts a new generation, so it always regroups.
            refresh = smooth and (
                entry is None
                or step_idx == 0
                or (
                    step_idx % self.perm_refresh_every == 0
                    and entry["step"] != step_idx
                )
            )
        if smooth:
            if refresh:
                perm, centroids = vc_token_permutation(
                    v,
                    num_clusters=self.kmeans_clusters,
                    iters=self.kmeans_iters,
                    init_centroids=None if entry is None else entry["centroids"],
                )
                step = None if denoise_step is None else denoise_step[0]
                if entry is None:
                    entry = {
                        "perm": perm.to(torch.int32),
                        "centroids": centroids,
                        "step": step,
                    }
                    self._groupings[key] = entry
                else:
                    # The buffers keep their addresses, so a CUDA graph captured around one call
                    # reads the grouping that later calls or replays write into them.
                    entry["perm"].copy_(perm.to(torch.int32))
                    entry["centroids"].copy_(centroids)
                    entry["step"] = step
            perm = entry["perm"]
        elif entry is not None:
            # V-Smooth off (paper Section 3.4). The plain kernel runs on the last
            # permutation with zero tile means.
            perm = entry["perm"]
        else:
            perm = torch.arange(k.shape[1], device=k.device, dtype=torch.int32)
            perm = perm.view(1, 1, -1).expand(k.shape[0], k.shape[2], -1).contiguous()
        if q is not None:
            return vc_quantize_fp8(q, k, v, perm=perm, demean=smooth)
        return vc_quantize(k, v, perm=perm, demean=smooth)


__all__ = [
    "VC_KMEANS_CLUSTERS",
    "VC_KMEANS_ITERS",
    "VC_PERM_REFRESH_EVERY",
    "VC_SMOOTH_STEP_FRACTION",
    "VCAttentionOperands",
    "VCAttentionPreprocessor",
    "hadamard_matrix",
    "vc_quantize",
    "vc_quantize_fp8",
    "vc_quantize_repair",
    "vc_token_permutation",
]
