# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA generation-phase RoPE + latent KV-cache append + FMHA scheduler-buffer fill."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: quant_mode bit for an fp8 KV cache.
QM_FP8_KV_CACHE = 128

#: Interleaved mRoPE: the 32 rope pairs split [12, 10, 10] across the t/h/w position streams; the offsets
#: make the three streams differ.
MROPE_SECTION = (12, 10, 10)
MROPE_OFFSETS = (0, 7, 19)

#: The checkpoint's MLA geometry.
_HY4 = dict(
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    kv_lora_rank=512,
    q_lora_rank=1024,
    v_head_dim=256,
    q_scaling=0.0625,
    attention_window_size=512,
    mrope_section=MROPE_SECTION,
    mrope_offsets=MROPE_OFFSETS,
)

#: RoPE bases: sparse layers 1e7, SWA layers 1e4.
THETA_SPARSE = 10_000_000.0
THETA_SWA = 10_000.0

#: The drafter's call at tep4: the 42-row primary pool, row 25 (first reused sparse layer), the pool's
#: served length 5134 (also its window: a sparse layer has none), and the pool's token budget.
NUM_TEXT_LAYERS = 42
DRAFT_TARGET_LAYER = 25
DRAFT_POOL_LEN = 5134
DRAFT_POOL_TOKENS = 24576

_HY4_DRAFT = {**_HY4, "attention_window_size": DRAFT_POOL_LEN}


class MlaRopeGeneration(OpWrapper):
    """One decode step's MLA preprocessing over the batch's generation sequences.

    Each sequence carries `P = predicted_tokens_per_seq` rows, token-major, row `(g, t)` at position
    `sequence_length[num_contexts + g] - P + t`. Over a bf16 pool: roped `q_pe` into `fused_q`'s rope tail,
    `[compressed_kv | roped k_pe]` appended to the pool, scheduler buffers filled. Returns `fused_q`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="SWA decode: 64 heads, P=1, 32-token pages, bf16 pool, mRoPE at 1e4",
            spec=dict(
                num_heads=64,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SWA,
                **_HY4,
            ),
        ),
        Cell(
            why="SWA decode over 64-token pages: 64 heads, P=1, bf16 pool",
            spec=dict(
                num_heads=64,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=64,
                quant_mode=0,
                rope_theta=THETA_SWA,
                **_HY4,
            ),
        ),
        Cell(
            why="the DFly2 verify step: 64 heads, P=6, bf16 pool, mRoPE at 1e4",
            spec=dict(
                num_heads=64,
                num_seqs=4,
                predicted_tokens_per_seq=6,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SWA,
                **_HY4,
            ),
        ),
        Cell(
            why="sparse-layer decode: 32 heads, P=1, bf16 pool, mRoPE at 1e7",
            spec=dict(
                num_heads=32,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SPARSE,
                **_HY4,
            ),
        ),
        Cell(
            why="sparse-layer verify: 32 heads, P=6, bf16 pool, mRoPE at 1e7",
            spec=dict(
                num_heads=32,
                num_seqs=4,
                predicted_tokens_per_seq=6,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SPARSE,
                **_HY4,
            ),
        ),
        Cell(
            why="a TP4 shard of the SWA layers: 16 heads, P=1, bf16 pool",
            spec=dict(
                num_heads=16,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SWA,
                **_HY4,
            ),
        ),
        Cell(
            why="a TP4 shard of the sparse layers: 8 heads, P=1, bf16 pool, mRoPE at 1e7",
            spec=dict(
                num_heads=8,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SPARSE,
                **_HY4,
            ),
        ),
        Cell(
            why="the ctx engine's SWA decode at attn_tp 2: 32 heads, P=1, 32-token pages, bf16 pool, mRoPE at 1e4",
            spec=dict(
                num_heads=32,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SWA,
                **_HY4,
            ),
        ),
        Cell(
            why="the ctx engine's sparse decode at attn_tp 2: 16 heads, P=1, 32-token pages, bf16 pool, mRoPE at 1e7",
            spec=dict(
                num_heads=16,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SPARSE,
                **_HY4,
            ),
        ),
        Cell(
            why="the envelope's largest row: 64 sequences at P=6, 384 q rows, 64 heads, bf16",
            spec=dict(
                num_heads=64,
                num_seqs=64,
                predicted_tokens_per_seq=6,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SWA,
                **_HY4,
            ),
        ),
        Cell(
            why=(
                "the drafter's block prologue at tep4: 8 q heads, P=6, 32-token pages, bf16 pool, "
                "mRoPE at 1e7, layer_idx 25 the target sparse layer's pool row, over a KV past 1024 positions"
            ),
            spec=dict(
                **_HY4_DRAFT,
                num_heads=8,
                num_seqs=64,
                predicted_tokens_per_seq=6,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SPARSE,
                layer_idx=DRAFT_TARGET_LAYER,
                num_layers=NUM_TEXT_LAYERS,
                num_contexts=0,
                kv_len=DRAFT_POOL_LEN,
                max_tokens=DRAFT_POOL_TOKENS,
            ),
        ),
        Cell(
            why="the same call in a mixed batch: context rows ahead of the generation requests, num_contexts above 0",
            spec=dict(
                **_HY4_DRAFT,
                num_heads=8,
                num_seqs=64,
                predicted_tokens_per_seq=6,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SPARSE,
                layer_idx=DRAFT_TARGET_LAYER,
                num_layers=NUM_TEXT_LAYERS,
                num_contexts=1,
                kv_len=DRAFT_POOL_LEN,
                max_tokens=DRAFT_POOL_TOKENS,
            ),
        ),
    )

    def __call__(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        rotary_cos_sin: Optional[torch.Tensor],
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        quant_q_buffer: Optional[torch.Tensor],
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        num_contexts: int,
        kv_cache_block_offsets: Optional[torch.Tensor],
        host_kv_cache_pool_pointers: Optional[torch.Tensor],
        host_kv_cache_pool_mapping: Optional[torch.Tensor],
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        kv_cache_scale_orig_quant: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        helix_tensor_params: List[Optional[torch.Tensor]],
        predicted_tokens_per_seq: int,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        residual_dim: int,
        tokens_per_block: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
        q_scaling: float,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_append: bool,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        precomputed_cu_seqlens: bool = False,
        precomputed_fmha_scheduler: bool = False,
        kv_only: bool = False,
        kv_done_elsewhere: bool = False,
        quant_scale_qkv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch.ops.trtllm.mla_rope_generation(
            fused_q,
            q_pe,
            latent_cache,
            rotary_cos_sin,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            sequence_length,
            host_past_key_value_lengths,
            host_context_lengths,
            num_contexts,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            kv_scale_orig_quant,
            kv_scale_quant_orig,
            kv_cache_scale_orig_quant,
            out_scale,
            block_ids_per_seq,
            helix_tensor_params,
            predicted_tokens_per_seq,
            layer_idx,
            num_heads,
            num_kv_heads,
            head_size,
            residual_dim,
            tokens_per_block,
            attention_window_size,
            beam_width,
            quant_mode,
            q_scaling,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            rope_append,
            kv_norm_weight,
            kv_norm_eps,
            precomputed_cu_seqlens,
            precomputed_fmha_scheduler,
            kv_only,
            kv_done_elsewhere,
            quant_scale_qkv,
        )
        return fused_q

    def reference(
        self,
        row: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_size: int,
        kv_scale_orig_quant: Optional[torch.Tensor] = None,
        store_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> torch.Tensor:
        """One `[.., C+R]` row the op writes: `[passthrough | GPT-J interleaved rope(pe)]`.

        `cos`/`sin` are already selected per row; the rotation rounds to the activation dtype before the
        scale and store, as the kernel does.
        """
        pe = row[..., row.shape[-1] - rope_size :]
        lora = row[..., : row.shape[-1] - rope_size]
        pairs = pe.float().reshape(*pe.shape[:-1], rope_size // 2, 2)
        roped = torch.empty_like(pairs)
        roped[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        roped[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        out = torch.cat([lora, roped.reshape(pe.shape).to(row.dtype)], dim=-1).float()
        if kv_scale_orig_quant is not None:
            out = out * kv_scale_orig_quant[0]
        return out.to(store_dtype)

    def reference_scheduler_buffers(
        self,
        kv_lens: List[int],
        num_heads: int,
        predicted_tokens_per_seq: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The `(cu_q_seqlens, cu_kv_seqlens)` the call fills; `kv_lens` include this step's `P` tokens."""
        p = predicted_tokens_per_seq
        cu_q = [i * num_heads * p for i in range(len(kv_lens) + 1)]
        cu_kv = [0]
        for length in kv_lens:
            cu_kv.append(cu_kv[-1] + length)
        return (
            torch.tensor(cu_q, dtype=torch.int32),
            torch.tensor(cu_kv, dtype=torch.int32),
        )

    def is_valid(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        rotary_cos_sin: Optional[torch.Tensor],
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        quant_q_buffer: Optional[torch.Tensor],
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        num_contexts: int,
        kv_cache_block_offsets: Optional[torch.Tensor],
        host_kv_cache_pool_pointers: Optional[torch.Tensor],
        host_kv_cache_pool_mapping: Optional[torch.Tensor],
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        kv_cache_scale_orig_quant: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        helix_tensor_params: List[Optional[torch.Tensor]],
        predicted_tokens_per_seq: int,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        residual_dim: int,
        tokens_per_block: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
        q_scaling: float,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_append: bool,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        precomputed_cu_seqlens: bool = False,
        precomputed_fmha_scheduler: bool = False,
        kv_only: bool = False,
        kv_done_elsewhere: bool = False,
        quant_scale_qkv: Optional[torch.Tensor] = None,
    ) -> None:
        """Mirrors `__call__` positionally because `validating` forwards arguments verbatim."""
        # Plain MLA row layout only; non-zero values are the DSA PrimTS / residual layouts.
        assert residual_dim == 0, (
            f"residual_dim must be 0 for the plain MLA row layout; got {residual_dim}. "
            "Only the DSA PrimTS record and residual layouts use a non-zero value"
        )

        # The kernel strides the table 2 * qk_rope_head_dim floats per position (duplicate_data=True)
        # and reads past a shorter table unchecked, later killing the CUDA context.
        if rotary_cos_sin is not None and sequence_length is not None:
            need = 2 * qk_rope_head_dim * int(sequence_length.max())
            assert rotary_cos_sin.numel() >= need, (
                f"rotary_cos_sin holds {rotary_cos_sin.numel()} floats, short of the "
                f"{need} a table of {2 * qk_rope_head_dim} floats per position needs for "
                f"position {int(sequence_length.max()) - 1}: the kernel strides "
                f"{2 * qk_rope_head_dim} floats per position (the duplicated pair layout, "
                "RopeParams(duplicate_data=True)) and reads past the end"
            )

        # The op does not presence-check quant_q_buffer on the fp8 path and launches into a null pointer.
        if quant_mode & QM_FP8_KV_CACHE:
            assert quant_q_buffer is not None, (
                "an fp8 KV cache (quant_mode & 128) requires quant_q_buffer; the op does "
                "not presence-check it and launches into a null pointer, killing the "
                "CUDA context"
            )

        # A weight folds kv_a_layernorm into the kernel; every caller here already normalized.
        assert kv_norm_weight is None, (
            "kv_norm_weight folds kv_a_layernorm into this kernel, which then reads "
            "latent_cache unnormalized; a caller that already normalized would "
            "normalize twice"
        )

        # The write scale `w` and the FMHA's read scale `r` only cancel when `r = 1/w`.
        if kv_scale_orig_quant is not None or kv_scale_quant_orig is not None:
            w = 1.0 if kv_scale_orig_quant is None else float(kv_scale_orig_quant[0])
            r = 1.0 if kv_scale_quant_orig is None else float(kv_scale_quant_orig[0])
            assert w != 0.0 and math.isclose(r, 1.0 / w, rel_tol=1e-6), (
                f"kv_scale_quant_orig ({r}) must be the reciprocal of kv_scale_orig_quant "
                f"({w}): the consuming FMHA undoes the write scale with it, and a "
                "non-reciprocal pair mis-scales the decode with nothing to show for it"
            )


mla_rope_generation = MlaRopeGeneration()
