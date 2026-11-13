# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter's block attention on the worker's private context arena (``attention_backend="VANILLA"``):
store the block's K / V after each request's context in its arena slot, then every block row attends, non-causally,
to the slot's context and the whole block (flash-attn ``flash_attn_with_kvcache``)."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


@lru_cache(maxsize=1)
def flash_attention():
    """flash-attn's ``flash_attn_with_kvcache`` (raises RuntimeError where flash-attn is not installed, with the
    message the DFlash speculative runtime's loader gives)."""
    try:
        from flash_attn import flash_attn_with_kvcache
    except ImportError as error:
        raise RuntimeError("DFlash VANILLA attention requires the flash-attn package.") from error
    return flash_attn_with_kvcache


def unavailability_reason() -> Optional[str]:
    """Why this build cannot run the op (flash-attn missing), or None."""
    try:
        flash_attention()
    except RuntimeError as error:
        return str(error)
    return None


class K3DSparkDenseBlockAttn(OpWrapper):
    """For B requests of a T-row block (``q`` [B, T, H, D] bf16; ``k`` / ``v`` [B T, KV, D], this layer's block rows):

    1. store row ``i`` of ``k`` / ``v`` in ``k_cache`` / ``v_cache`` ([S, L, C, KV, D], the arena's slots, layers and
       columns) at slot ``append_batch_indices[i]``, layer ``layer_idx``, column ``append_positions[i]``;
    2. return the attention output [B, T, H, D]: request b's rows attend to the first ``cache_seqlens[b]`` columns of
       slot ``cache_batch_idx[b]`` (its context then its block), every row to every key, scaled by ``D ** -0.5``.

    With ``pack_gqa`` the H / KV query heads that share a KV head are folded into the row dimension for the kernel
    and unfolded after (exact: every row of the block attends to the same keys). The output is [B, T, H, D] either
    way."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{heads}/{kv} heads of 64, B={b}, T={t}, GQA {'packed' if pack else 'unpacked'}: {why}",
            spec=dict(heads=heads, kv=kv, batch=b, block=t, pack_gqa=pack),
        )
        for heads, kv, why in (
            (4, 1, "RadixArk drafter, one TP16 rank"),
            (6, 1, "the drafter entries' layout"),
        )
        for b, t in ((1, 7), (8, 7), (3, 8), (16, 7))
        for pack in (True,)
    ) + (
        Cell(
            why="4/1 heads, B=2, T=7, unpacked: the kernel's own GQA handling",
            spec=dict(heads=4, kv=1, batch=2, block=7, pack_gqa=False),
        ),
    )

    note = """
    The store and the flash-attn call the built-in DFlash drafter makes per layer
    on its VANILLA backend for an unwindowed non-causal layer, with the same
    arguments (causal False, window (-1, -1), keyword call). The drafter packs
    GQA whenever H > KV, as the built-in drafter does. The block's K / V stay in the arena
    after the call (the worker overwrites them with the accepted rows' next
    step). Needs flash-attn (`unavailability_reason`). The reference
    accumulates in fp32; the kernel rounds P to bf16.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer_idx: int,
        append_batch_indices: torch.Tensor,
        append_positions: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_batch_idx: torch.Tensor,
        pack_gqa: bool,
    ) -> torch.Tensor:
        batch, block, heads, head_dim = q.shape
        kv_heads = k.shape[1]
        group = heads // kv_heads
        k_cache[append_batch_indices, layer_idx, append_positions] = k
        v_cache[append_batch_indices, layer_idx, append_positions] = v
        if pack_gqa:
            q_in = (
                q.reshape(batch, block, kv_heads, group, head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(batch, group * block, kv_heads, head_dim)
            )
        else:
            q_in = q
        out = flash_attention()(
            q=q_in,
            k_cache=k_cache[:, layer_idx],
            v_cache=v_cache[:, layer_idx],
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            causal=False,
            window_size=(-1, -1),
        )
        if pack_gqa:
            # Unfold the heads: [B, H / KV, T, KV, D] -> [B, T, KV, H / KV, D], then [B, T, H, D] (head kv * group + g,
            # as q's). The one copy the built-in drafter makes when it flattens the rows for o_proj, made here instead.
            out = (
                out.view(batch, group, block, kv_heads, head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(batch, block, heads, head_dim)
            )
        return out

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer_idx: int,
        append_batch_indices: torch.Tensor,
        append_positions: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_batch_idx: torch.Tensor,
        pack_gqa: bool,
    ) -> torch.Tensor:
        """Plain torch, fp32; ``pack_gqa`` does not change the result."""
        batch, block, heads, head_dim = q.shape
        group = heads // k.shape[1]
        k_cache[append_batch_indices.long(), layer_idx, append_positions.long()] = k
        v_cache[append_batch_indices.long(), layer_idx, append_positions.long()] = v
        out = torch.empty_like(q)
        with true_fp32_matmul():
            for b in range(batch):
                slot, n = int(cache_batch_idx[b]), int(cache_seqlens[b])
                keys = (
                    k_cache[slot, layer_idx, :n].float().transpose(0, 1).repeat_interleave(group, 0)
                )
                vals = (
                    v_cache[slot, layer_idx, :n].float().transpose(0, 1).repeat_interleave(group, 0)
                )
                qb = q[b].float().transpose(0, 1)  # [H, T, D]
                p = torch.softmax(qb @ keys.transpose(1, 2) * head_dim**-0.5, -1)
                out[b] = (p @ vals).transpose(0, 1).to(q.dtype)
        return out

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer_idx: int,
        append_batch_indices: torch.Tensor,
        append_positions: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_batch_idx: torch.Tensor,
        pack_gqa: bool,
    ) -> None:
        assert q.dtype == k.dtype == v.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16, (
            "bf16 only"
        )
        assert k_cache.dim() == 5 and k_cache.shape[3] == k.shape[1], (
            f"the arena must be [slots, layers, columns, KV, D]; got {tuple(k_cache.shape)}"
        )
        assert q.shape[2] % k.shape[1] == 0, "query heads must be a multiple of KV heads"
        assert cache_seqlens.dtype == torch.int32 and cache_batch_idx.dtype == torch.int32, (
            "flash-attn takes int32 lengths and slots"
        )


k3_dspark_dense_block_attn = K3DSparkDenseBlockAttn()
