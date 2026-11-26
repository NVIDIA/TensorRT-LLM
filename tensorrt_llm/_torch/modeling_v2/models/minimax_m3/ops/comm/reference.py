# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Torch mirrors of the MiniMax-M3 multicast boundaries."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

_E4M3_MAX = 448.0


def mxfp8_rows(normed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 of BF16 rows as ``trtllm::mxfp8_quantize(normed, True)``: E4M3 [M, H] and 128x4-swizzled UE8M0 bytes.

    One scale per 32 columns, the smallest power of two at or above ``amax / 448`` (biased exponent, round up),
    data = row / scale rounded to nearest E4M3. The scale buffer is padded to 128 rows, padding bytes left as the
    op leaves them (the comparison masks them out).
    """
    m, h = normed.shape
    blocks = normed.float().view(m, h // 32, 32)
    amax = blocks.abs().amax(-1)
    bits = (amax / _E4M3_MAX).view(torch.int32)
    byte = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) > 0).to(torch.int32)).clamp(0, 254)
    scale = torch.exp2(byte.float() - 127.0).unsqueeze(-1)
    data = torch.where(amax.unsqueeze(-1) != 0, blocks / scale, torch.zeros_like(blocks))
    data = data.view(m, h).to(torch.float8_e4m3fn)
    return data, swizzle_sf(byte.to(torch.uint8))


def swizzle_sf(byte: torch.Tensor) -> torch.Tensor:
    """Scatter [rows, H / 32] scale bytes into the 128x4-swizzled buffer the op writes (zero padding)."""
    rows, cols = byte.shape
    padded_rows = (rows + 127) // 128 * 128
    r = torch.arange(rows, device=byte.device).view(-1, 1)
    c = torch.arange(cols, device=byte.device).view(1, -1)
    index = (
        (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * (128 * cols)
    )
    out = torch.zeros(padded_rows * cols, dtype=torch.uint8, device=byte.device)
    out[index.reshape(-1)] = byte.reshape(-1)
    return out


def sf_written(rows: int, cols: int, device) -> torch.Tensor:
    """Mask of the swizzled scale bytes the op writes for ``rows`` rows (the padding rows are left unwritten)."""
    return swizzle_sf(torch.ones(rows, cols, dtype=torch.uint8, device=device)).bool()


def boundary(
    partials: Sequence[torch.Tensor],
    residual: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    quant: bool,
    drafter: bool = False,
) -> tuple[torch.Tensor, ...]:
    """The boundary arithmetic once each rank's BF16 partial is known (shared with the finalize entry).

    ``drafter``: the Eagle3 drafter boundary (opts bit 16), whose stock is a plain BF16 all-reduce, then FlashInfer
    fused_add_rmsnorm: the RMS statistic and the normalization take the unrounded FP32 sum bf16(rank sum) + residual;
    only the stored residual is rounded to BF16.

    The stock rounding points of this target (a BF16 all-reduce, then the fused residual add + RMSNorm): ranks are
    summed in FP32 in rank order and rounded to BF16, the residual added and rounded to BF16 again; the RMS statistic
    and the normalization are done in float64 and rounded to BF16 once.
    """
    total = torch.zeros(partials[0].shape, dtype=torch.float32, device=residual.device)
    for p in partials:
        total = total + p.float()
    pre = total.to(torch.bfloat16).float() + residual.float()
    updated = pre.to(torch.bfloat16)
    u = pre.double() if drafter else updated.double()
    inv = torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + eps)
    normed = (u * inv * norm_w.double()).to(torch.bfloat16)
    if not quant:
        return normed, updated
    return (normed, updated, *mxfp8_rows(normed))


def finalize_partial(
    fc2: torch.Tensor, perm: torch.Tensor, scales: torch.Tensor, shared: Optional[torch.Tensor]
) -> torch.Tensor:
    """One rank's BF16 partial: ``bf16(bf16(sum_k scale_k * fc2[perm_k]) + shared)``, the routed sum in FP32 in k
    order and rounded to BF16, then the shared expert added as stock adds it (one more BF16 rounding).

    The kernel's per-k step is a fused multiply-add; it is reproduced exactly by the float64 product (exact for two
    FP32 operands) plus the FP32 accumulator, rounded back to FP32. Dropped entries (perm -1) are skipped.
    """
    m, topk = perm.shape
    acc = torch.zeros(m, fc2.shape[1], dtype=torch.float32, device=fc2.device)
    sc = scales.reshape(m, topk).float()
    for k in range(topk):
        rows = perm[:, k].long()
        keep = (rows >= 0).unsqueeze(-1)
        gathered = fc2[rows.clamp_min(0)].double()
        step = (gathered * sc[:, k : k + 1].double() + acc.double()).float()
        acc = torch.where(keep, step, acc)
    if shared is not None:
        acc = acc.to(torch.bfloat16).float() + shared.float()
    return acc.to(torch.bfloat16)


class McAllreduceNormReference(OpWrapper):
    """Reference arithmetic; targets supply their native call and certification cells."""

    def reference(
        self,
        per_rank_x: Sequence[torch.Tensor],
        per_rank_x2: Optional[Sequence[torch.Tensor]],
        residual: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        quant: bool,
        drafter: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """The boundary the group is supposed to compute (``drafter``: the opts-16 drafter boundary).

        The documented collective exception: it takes every rank's ``x`` (and ``x2``) in rank order, not the calling
        rank's. Each rank's partial is ``bf16(x + x2)`` (``x`` alone without ``x2``), whatever pushed it.
        """
        partials = []
        for r, x in enumerate(per_rank_x):
            if per_rank_x2 is None:
                partials.append(x)
            else:
                partials.append((x.float() + per_rank_x2[r].float()).to(torch.bfloat16))
        return boundary(partials, residual, norm_w, eps, quant, drafter)


class McFinalizeAllreduceNormReference(OpWrapper):
    """Reference arithmetic; targets supply their native call and certification cells."""

    def reference(
        self,
        per_rank_fc2: Sequence[torch.Tensor],
        per_rank_perm: Sequence[torch.Tensor],
        per_rank_scales: Sequence[torch.Tensor],
        per_rank_shared: Optional[Sequence[torch.Tensor]],
        residual: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        quant: bool,
    ) -> tuple[torch.Tensor, ...]:
        """The boundary the group is supposed to compute, from every rank's MoE outputs in rank order (the
        collective exception, as in ``m3_mc_allreduce_norm.reference``)."""
        partials = [
            finalize_partial(
                fc2,
                per_rank_perm[r],
                per_rank_scales[r],
                None if per_rank_shared is None else per_rank_shared[r],
            )
            for r, fc2 in enumerate(per_rank_fc2)
        ]
        return boundary(partials, residual, norm_w, eps, quant)
