# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the K3.1 Rubin MoE entries share: the router's shapes and its true-FP32 reference, the
``noaux_tc`` top-k and ``moe_sort`` references, and the NVFP4 numerics. Not an entry: the MoE
entries beside it import from it."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import true_fp32_matmul

#: The router: hidden size and routed experts.
HIDDEN = 7168
NUM_EXPERTS = 896
#: Experts per token, and the CuteDSL MoE tile the sort tables are cut for.
TOP_K = 16
TILE = 128
#: NVFP4 block: one E4M3 scale per 16 values along K.
SF_VEC = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
#: The 16 E2M1 code points in code order (bit 3 sign, bits 2:1 exponent, bit 0 mantissa).
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def router_logits_reference(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``hidden_states @ weight.T`` of the BF16 operands in true FP32 (exact products, FP32 sums),
    rounded once."""
    with true_fp32_matmul():
        return hidden_states.float() @ weight.float().t()


def sort_buffer_sizes(num_tokens: int, local_num_experts: int, tile: int = TILE) -> tuple[int, int]:
    """``(max tiles, max permuted rows)`` exactly as ``moe_sort`` allocates them."""
    expanded = num_tokens * TOP_K
    filled = min(local_num_experts, expanded)
    max_tiles = filled + (expanded - filled) // tile
    return max_tiles, max_tiles * tile


def noaux_topk(
    logits: torch.Tensor, bias: torch.Tensor, routed_scaling_factor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``noaux_tc_op``'s top-16 with one expert group, in its own arithmetic.

    Tanh-form sigmoid; selection on ``sigmoid + bias``, ties to the lower expert; weights are the
    unbiased scores of the picks, renormalized in FP64 as ``score * scale / (total + 1e-20)`` and
    rounded once to FP32. Returns ``(ids int32 [M, 16], weights fp32 [M, 16])``.
    """
    scores = 0.5 * torch.tanh(0.5 * logits.float()) + 0.5
    choice = scores + bias.float()
    ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :TOP_K]
    picked = torch.gather(scores, 1, ids)
    total = picked.sum(-1, keepdim=True).double()
    weights = picked.double() * float(routed_scaling_factor) / (total + 1e-20)
    return ids.to(torch.int32), weights.float()


def token_ordered_sort(
    ids: torch.Tensor, local_expert_offset: int, local_num_experts: int, tile: int = TILE
) -> tuple[torch.Tensor, ...]:
    """``moe_sort``'s six tables for one rank, rows of an expert in ascending token order.

    Entries no row reaches stay ``-1``; the kernel leaves them unwritten, so only the first
    ``num_non_exiting_tiles`` tiles and the permuted rows they cover are compared.
    """
    m, k = ids.shape
    device = ids.device
    i32 = dict(dtype=torch.int32, device=device)
    max_tiles, max_permuted = sort_buffer_sizes(m, local_num_experts, tile)
    local = ids.reshape(-1).long() - local_expert_offset
    keep = (local >= 0) & (local < local_num_experts)
    experts = local[keep]
    rows = torch.arange(m * k, device=device)[keep]
    order = torch.argsort(experts * (m * k) + rows)
    experts, rows = experts[order], rows[order]
    counts = torch.bincount(experts, minlength=local_num_experts)
    tiles = (counts + tile - 1) // tile
    base = (torch.cumsum(tiles, 0) - tiles) * tile
    starts = torch.cumsum(counts, 0) - counts
    permuted = base[experts] + torch.arange(experts.numel(), device=device) - starts[experts]

    e2p = torch.full((m * k,), -1, **i32)
    e2p[rows] = permuted.to(torch.int32)
    p2e = torch.full((max_permuted,), -1, **i32)
    p2e[permuted] = rows.to(torch.int32)
    n = int(tiles.sum())
    tile_owner = torch.repeat_interleave(torch.arange(local_num_experts, device=device), tiles)
    nth = torch.arange(n, device=device) - (torch.cumsum(tiles, 0) - tiles)[tile_owner]
    tile_expert = torch.full((max_tiles,), -1, **i32)
    tile_limit = torch.full((max_tiles,), -1, **i32)
    tile_expert[:n] = tile_owner.to(torch.int32)
    tile_limit[:n] = torch.minimum(
        base[tile_owner] + (nth + 1) * tile, base[tile_owner] + counts[tile_owner]
    ).to(torch.int32)
    return (
        tile_expert,
        tile_limit,
        e2p.view(m, k),
        p2e,
        torch.tensor([n * tile], **i32),
        torch.tensor([n], **i32),
    )


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    """FP32 values of packed E2M1 codes ``[..., K/2]`` -> ``[..., K]`` (low nibble first)."""
    table = torch.tensor(_E2M1, dtype=torch.float32, device=codes.device)
    lo = table[(codes & 0xF).long()]
    hi = table[(codes >> 4).long()]
    return torch.stack((lo, hi), dim=-1).reshape(*codes.shape[:-1], codes.shape[-1] * 2)


def dequantize_nvfp4(codes: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    """``[N, K]`` FP32 of codes times their E4M3 block scales (no global scale)."""
    scales = block_scales.view(torch.float8_e4m3fn).float().repeat_interleave(SF_VEC, dim=-1)
    return e2m1_values(codes) * scales


def _e2m1_rne(a: torch.Tensor) -> torch.Tensor:
    """Round to nearest even on the E2M1 grid, saturating at +-6 (step 0.5 / 1 / 2 by binade)."""
    step = torch.where(a.abs() < 2.0, 0.5, torch.where(a.abs() < 4.0, 1.0, 2.0))
    return torch.sign(a) * torch.clamp(torch.round(a.abs() / step) * step, max=E2M1_MAX)


def quantize_nvfp4(x: torch.Tensor, global_scale: float) -> torch.Tensor:
    """NVFP4 quantize-dequantize of FP32 ``x [rows, K]`` under ``global_scale`` (``1 /
    input_scale``), back in ``x``'s units.

    Per 16 values ``sf = e4m3(g * amax / 6)`` and ``code = e2m1(x * g / sf)``; returns
    ``code * sf / g``, the value a block-scaled MMA multiplies once its alpha divides ``g`` out.
    """
    rows, cols = x.shape
    blocks = x.reshape(rows, cols // SF_VEC, SF_VEC)
    sf = (
        (global_scale * blocks.abs().amax(dim=-1, keepdim=True) / E2M1_MAX)
        .clamp(max=E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .float()
    )
    scale = torch.where(sf == 0, torch.zeros_like(sf), global_scale / sf)
    return (_e2m1_rne(blocks * scale) * sf / global_scale).reshape(rows, cols)
