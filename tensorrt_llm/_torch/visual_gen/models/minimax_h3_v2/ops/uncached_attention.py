# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import torch
from torch.nn import functional as F

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.visual_gen.args import QuantAttentionConfig


class ContextBuffers(NamedTuple):
    host_lengths: torch.Tensor
    device_lengths: torch.Tensor
    request_types: torch.Tensor
    host_totals: torch.Tensor
    workspace: torch.Tensor
    kv_scale: torch.Tensor

    @classmethod
    def for_shape(cls, batch: int, length: int, device: torch.device) -> "ContextBuffers":
        lens = torch.full((batch,), length, dtype=torch.int32, device="cpu")
        return cls(
            lens,
            lens.to(device),
            torch.zeros_like(lens),
            torch.tensor([batch * length, 0], dtype=torch.int32, device="cpu"),
            torch.empty(0, dtype=torch.int8, device=device),
            torch.ones(1, dtype=torch.float32, device=device),
        )


class UncachedAttention(OpWrapper):
    """Full context attention on equal-length H3 sequences with packed or separate Q/K/V."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (
        Cell("Two equal-length packed H3 sequences", dict(batch=2, length=65)),
        Cell("H3 release smoke with joint text, video and audio rows", dict(batch=1, length=5)),
    )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        buffers: ContextBuffers,
        skip_stats: torch.Tensor,
        *,
        length: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_idx: int,
        quant_mode: int,
        prefill_threshold: float | None,
        decode_threshold: float | None,
        recipe: QuantAttentionConfig | None,
    ) -> torch.Tensor:
        host_lens, device_lens, types, totals, workspace, kv_scale = buffers
        batch = host_lens.numel()
        out = q.new_empty((batch * length, num_heads * head_dim))
        FallbackFmha.attention(
            q=q,
            k=k,
            v=v,
            output=out,
            output_sf=None,
            workspace=workspace,
            sequence_length=device_lens,
            host_past_key_value_lengths=host_lens,
            host_total_kv_lens=totals,
            context_lengths=device_lens,
            host_context_lengths=host_lens,
            host_request_types=types,
            max_context_q_len_override=None,
            kv_cache_block_offsets=None,
            host_kv_cache_pool_pointers=None,
            host_kv_cache_pool_mapping=None,
            cache_indirection=None,
            kv_scale_orig_quant=kv_scale,
            kv_scale_quant_orig=kv_scale,
            out_scale=None,
            rotary_inv_freq=None,
            rotary_cos_sin=None,
            latent_cache=None,
            q_pe=None,
            block_ids_per_seq=None,
            attention_sinks=None,
            is_fused_qkv=k is None,
            update_kv_cache=True,
            predicted_tokens_per_seq=1,
            local_layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            tokens_per_block=None,
            max_num_requests=batch,
            max_context_length=length,
            max_seq_len=length,
            attention_window_size=length,
            beam_width=1,
            mask_type=0,
            quant_mode=quant_mode,
            q_scaling=1.0,
            position_embedding_type=0,
            rope_dim=0,
            rope_base=10000.0,
            rope_scale_type=0,
            rope_scale=1.0,
            rope_short_m_scale=1.0,
            rope_long_m_scale=1.0,
            rope_max_positions=1024,
            rope_original_max_positions=1024,
            use_paged_context_fmha=False,
            attention_input_type=0,
            is_mla_enable=False,
            chunked_prefill_buffer_batch_size=1,
            q_lora_rank=None,
            kv_lora_rank=None,
            qk_nope_head_dim=None,
            qk_rope_head_dim=None,
            v_head_dim=head_dim,
            rope_append=None,
            mrope_rotary_cos_sin=None,
            mrope_position_deltas=None,
            helix_position_offsets=None,
            helix_is_inactive_rank=None,
            attention_chunk_size=None,
            softmax_stats_tensor=None,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            is_spec_dec_tree=False,
            spec_decoding_generation_lengths=None,
            spec_decoding_position_offsets=None,
            spec_decoding_packed_mask=None,
            spec_decoding_bl_tree_mask_offset=None,
            spec_decoding_bl_tree_mask=None,
            spec_bl_tree_first_sparse_mask_offset_kv=None,
            sparse_kv_indices=None,
            sparse_kv_offsets=None,
            sparse_attn_indices=None,
            sparse_attn_offsets=None,
            sparse_attn_indices_block_size=0,
            skip_softmax_threshold_scale_factor_prefill=prefill_threshold,
            skip_softmax_threshold_scale_factor_decode=decode_threshold,
            skip_softmax_stat=skip_stats,
            sage_attn_num_elts_per_blk_q=recipe.q_block_size if recipe else 0,
            sage_attn_num_elts_per_blk_k=recipe.k_block_size if recipe else 0,
            sage_attn_num_elts_per_blk_v=recipe.v_block_size if recipe else 0,
            sage_attn_qk_int8=recipe is not None and recipe.qk_dtype == "int8",
            sage_attn_smooth_k=recipe is not None and recipe.smooth_k,
            num_contexts=batch,
            num_ctx_tokens=batch * length,
        )
        return out

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        buffers: ContextBuffers,
        skip_stats: torch.Tensor,
        *,
        length: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_idx: int,
        quant_mode: int,
        prefill_threshold: float | None,
        decode_threshold: float | None,
        recipe: QuantAttentionConfig | None,
    ) -> torch.Tensor:
        """FP32 full-attention reference before optional Sage quantization or softmax skipping."""
        if k is None:
            q, k, v = q.split(
                (num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim), dim=-1
            )
        assert k is not None and v is not None
        q = q.reshape(-1, length, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(-1, length, num_kv_heads, head_dim).transpose(1, 2)
        v = v.reshape(-1, length, num_kv_heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), enable_gqa=num_heads != num_kv_heads
        )
        return out.transpose(1, 2).reshape(-1, num_heads * head_dim).to(q.dtype)

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        buffers: ContextBuffers,
        skip_stats: torch.Tensor,
        *,
        length: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_idx: int,
        quant_mode: int,
        prefill_threshold: float | None,
        decode_threshold: float | None,
        recipe: QuantAttentionConfig | None,
    ) -> None:
        rows = buffers.host_lengths.numel() * length
        assert num_heads % num_kv_heads == 0
        assert (k is None) == (v is None)
        assert q.shape == (rows, (num_heads + (2 * num_kv_heads if k is None else 0)) * head_dim)
        if k is not None and v is not None:
            assert k.shape == v.shape == (rows, num_kv_heads * head_dim)
        assert skip_stats.numel() == 2


uncached_attention = UncachedAttention()
