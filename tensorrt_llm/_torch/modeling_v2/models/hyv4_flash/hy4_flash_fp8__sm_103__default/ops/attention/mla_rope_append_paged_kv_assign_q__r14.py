# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RoPE the new context tokens and append their latent rows to the paged MLA cache."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: quant_mode bit for an fp8 KV cache.
QM_FP8_KV_CACHE = 128


class MlaRopeAppendPagedKvAssignQ(OpWrapper):
    """RoPE new context tokens' `q_pe` (in `q`) and `k_pe` (in `latent_cache`) in place, then append
    `[compressed_kv | rope(k_pe)]` to the paged MLA cache."""

    ARCHS = frozenset({Arch.SM_103})

    _HY4 = dict(
        nope_size=192,
        rope_size=64,
        lora_size=512,
        residual_dim=0,
        tokens_per_block=32,
    )

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="swa context call: 64 heads, a 512-token prefill chunk over a bf16 pool",
            spec=dict(head_num=64, cached=[0], new=[512], quant_mode=0, **_HY4),
        ),
        Cell(
            why="swa context call: a 5120-token prefill chunk over a bf16 pool",
            spec=dict(head_num=64, cached=[0], new=[5120], quant_mode=0, **_HY4),
        ),
        Cell(
            why="sparse context call: 32 heads, a 512-token prefill chunk over a bf16 pool",
            spec=dict(head_num=32, cached=[0], new=[512], quant_mode=0, **_HY4),
        ),
        Cell(
            why="fp8 pool: two sequences, the second starting on a 512-token cached prefix",
            spec=dict(
                head_num=64,
                cached=[0, 512],
                new=[512, 512],
                quant_mode=QM_FP8_KV_CACHE,
                **_HY4,
            ),
        ),
        Cell(
            why="swa at tp4: 16 heads per rank, a 512-token prefill chunk over a bf16 pool",
            spec=dict(head_num=16, cached=[0], new=[512], quant_mode=0, **_HY4),
        ),
        Cell(
            why="sparse at tp4: 8 heads per rank, a 512-token prefill chunk over a bf16 pool",
            spec=dict(head_num=8, cached=[0], new=[512], quant_mode=0, **_HY4),
        ),
        Cell(
            why="swa at the ctx engine's tp2: 32 heads, a 512-token prefill chunk over a bf16 pool",
            spec=dict(head_num=32, cached=[0], new=[512], quant_mode=0, **_HY4),
        ),
        Cell(
            why="sparse at the ctx engine's tp2: 16 heads, a 512-token prefill chunk over a bf16 pool",
            spec=dict(head_num=16, cached=[0], new=[512], quant_mode=0, **_HY4),
        ),
        Cell(
            why="the ctx engine's prefill chunk bound at tp2: 32 heads, 5120 new tokens, bf16 pool",
            spec=dict(head_num=32, cached=[0], new=[5120], quant_mode=0, **_HY4),
        ),
        Cell(
            why="envelope's largest row: 8192 new tokens in one context call, 64 heads, bf16",
            spec=dict(head_num=64, cached=[0], new=[8192], quant_mode=0, **_HY4),
        ),
        Cell(
            why="envelope's largest row at the ctx engine's tp2: 32 heads, 8192 new tokens, bf16",
            spec=dict(head_num=32, cached=[0], new=[8192], quant_mode=0, **_HY4),
        ),
    )

    def __call__(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        num_contexts: int,
        cu_ctx_cached_kv_lens: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        max_input_uncached_seq_len: int,
        cos_sin_cache: torch.Tensor,
        head_num: int,
        nope_size: int,
        rope_size: int,
        lora_size: int,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        kv_scale_orig_quant: Optional[torch.Tensor],
        residual_dim: int,
        layer_idx: int,
        tokens_per_block: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
    ) -> torch.Tensor:
        torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
            q,
            latent_cache,
            num_contexts,
            cu_ctx_cached_kv_lens,
            cu_seq_lens,
            max_input_uncached_seq_len,
            cos_sin_cache,
            head_num,
            nope_size,
            rope_size,
            lora_size,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            kv_scale_orig_quant,
            residual_dim,
            layer_idx,
            tokens_per_block,
            attention_window_size,
            beam_width,
            quant_mode,
        )
        return q

    def reference(
        self,
        latent_row: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_size: int,
        kv_scale_orig_quant: Optional[torch.Tensor] = None,
        store_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> torch.Tensor:
        """The latent row the op should append, given `cos`/`sin` already selected per row.

        GPT-J interleaved pairs. The rotated pair rounds to the input dtype before the scale and store,
        as the kernel does.
        """
        row = latent_row.float()
        lora = row[..., : row.shape[-1] - rope_size]
        pe = row[..., row.shape[-1] - rope_size :]
        pairs = pe.reshape(*pe.shape[:-1], rope_size // 2, 2)
        roped = torch.empty_like(pairs)
        roped[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        roped[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        roped = roped.to(latent_row.dtype).float()
        out = torch.cat([lora, roped.reshape(pe.shape)], dim=-1)
        if kv_scale_orig_quant is not None:
            out = out * kv_scale_orig_quant[0]
        return out.to(store_dtype)

    def is_valid(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        num_contexts: int,
        cu_ctx_cached_kv_lens: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        max_input_uncached_seq_len: int,
        cos_sin_cache: torch.Tensor,
        head_num: int,
        nope_size: int,
        rope_size: int,
        lora_size: int,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        kv_scale_orig_quant: Optional[torch.Tensor],
        residual_dim: int,
        *unused_args: object,
        **unused_kwargs: object,
    ) -> None:
        # The op accepts non-zero residual_dim only for an FP4 pool; this target's pool is bf16 or e4m3.
        assert residual_dim == 0, (
            f"residual_dim must be 0 for a bf16 or fp8 KV pool; got {residual_dim}. "
            "A non-zero value is only legal against an FP4 pool"
        )


mla_rope_append_paged_kv_assign_q = MlaRopeAppendPagedKvAssignQ()
