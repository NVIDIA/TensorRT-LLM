# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash MLA attention (paged latent append + FMHA) via ``FallbackFmha.attention``: the sliding-window
layers' two phases (512-token window, per-head sink), the sparse layers' decode and the DFly2 drafter's block,
both with their selection as ``sparse_attn_indices``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: attention_input_type values for MLA's context and generation phases.
MLA_CONTEXT, MLA_GENERATION = 1, 2

#: quant_mode bit for an fp8 KV cache.
QM_FP8_KV_CACHE = 128
#: The sliding-window MLA layer. `prefill`/`cached`/`new`/`decode`/`rows` are test scenario fields.
_HY4_SWA = dict(
    num_heads=64,
    num_kv_heads=64,
    head_size=576,
    v_head_dim=256,
    sinks=True,
    window=512,
    quant_mode=0,
    tokens_per_block=32,
    max_num_requests=64,
    use_paged_context_fmha=False,
)


#: The DFly2 drafter's block call: 8 q heads (32 under tp4), `predicted_tokens_per_seq` = block width.
_HY4_DFLY_BLOCK = dict(
    block=True,
    num_heads=8,
    num_kv_heads=1,
    head_size=576,
    v_head_dim=512,
    predicted_tokens_per_seq=6,
    tokens_per_block=32,
    quant_mode=0,
)


class ThopAttention(OpWrapper):
    """Append this step's K/V to the paged cache, then attend over it. Writes rows `[:num_tokens]` of
    `output` and returns it."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="Hy4 SWA MLA, one request over a bf16 pool: a 600-token prefill "
            "whose late rows see only the 512-token window, then one decode over "
            "601 keys",
            spec=dict(_HY4_SWA, max_num_requests=1, prefill=600, decode=1),
        ),
        Cell(
            why="64 requests over a bf16 pool: a 64-token prefill each, then one "
            "decode each over 65 keys",
            spec=dict(_HY4_SWA, prefill=64, decode=1),
        ),
        Cell(
            why="one request over a bf16 pool on the paged-context path: 32 new "
            "rows over a 512-token cached prefix, then one decode over 545 keys",
            spec=dict(
                _HY4_SWA,
                max_num_requests=1,
                use_paged_context_fmha=True,
                cached=512,
                new=32,
                decode=1,
            ),
        ),
        Cell(
            why="64 requests over an fp8-e4m3 pool: a 64-token prefill each, then "
            "one decode each over 65 keys",
            spec=dict(_HY4_SWA, quant_mode=QM_FP8_KV_CACHE, prefill=64, decode=1),
        ),
        Cell(
            why="64 requests over a bf16 pool on the paged-context path: 32 new "
            "rows over a 64-token cached prefix each, then one decode each",
            spec=dict(_HY4_SWA, use_paged_context_fmha=True, cached=64, new=32, decode=1),
        ),
        Cell(
            why="the tp4 shard, 16 q heads: 64 requests over a bf16 pool, a "
            "64-token prefill each, then one decode each",
            spec=dict(_HY4_SWA, num_heads=16, num_kv_heads=16, prefill=64, decode=1),
        ),
        Cell(
            why="the tp4 shard, 16 q heads, on the paged-context path: 32 new "
            "rows over a 64-token cached prefix each, then one decode each",
            spec=dict(
                _HY4_SWA,
                num_heads=16,
                num_kv_heads=16,
                use_paged_context_fmha=True,
                cached=64,
                new=32,
                decode=1,
            ),
        ),
        Cell(
            why="the ctx engine's SWA layers at attn_tp 2: 32 q / 32 kv heads, "
            "64 requests over a bf16 pool, a 64-token prefill then a decode "
            "each",
            spec=dict(_HY4_SWA, num_heads=32, num_kv_heads=32, prefill=64, decode=1),
        ),
        Cell(
            why="the envelope's largest context row count: 8192 rows (16 "
            "sequences of 512) in one paged-context call over a bf16 pool, then "
            "one decode each over 513 keys",
            spec=dict(
                _HY4_SWA,
                max_num_requests=16,
                use_paged_context_fmha=True,
                prefill=512,
                decode=1,
                rows=8192,
            ),
        ),
        Cell(
            why="the ctx engine's sparse layers at attn_tp 2: 16 q / 1 kv "
            "head, one decode each over the top-k index selection passed as "
            "sparse_attn_indices",
            spec=dict(
                block=True,
                num_heads=16,
                num_kv_heads=1,
                head_size=576,
                v_head_dim=512,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                max_num_requests=64,
            ),
        ),
        Cell(
            why="the DFly2 draft block at the tp4 shard: 4 requests x 6 rows over "
            "the sparse layers' bf16 pool, with the k29 block selection as "
            "sparse_attn_indices",
            spec=dict(_HY4_DFLY_BLOCK, max_num_requests=4),
        ),
        Cell(
            why="the engine's largest block batch: 64 requests x 6 rows, 384 "
            "generation rows in one call over the sparse layers' bf16 pool, each "
            "row reading its own selection",
            spec=dict(_HY4_DFLY_BLOCK, max_num_requests=64),
        ),
        Cell(
            why="the block at the engine's largest batch and the gate's served "
            "length: 64 requests x 6 rows over the sparse layers' bf16 pool, "
            "each row's selection 2176 wide, the batch's device KV length 5134 "
            "and the entry's bound the writer's count through the block's last "
            "row (2054); the device length is the refused control",
            spec=dict(
                _HY4_DFLY_BLOCK,
                max_num_requests=64,
                num_contexts=0,
                kv_len=5134,
                num_sparse_topk=2176,
            ),
        ),
        Cell(
            why="the same block in a mixed batch: 8 context rows ahead of 56 "
            "generation rows, the engine's largest batch, every sequence's "
            "device KV length 5134 and each generation row's selection 2176 "
            "wide, the generation rows bound by the writer's count through the "
            "block's last row; the device length is the refused control",
            spec=dict(
                _HY4_DFLY_BLOCK,
                max_num_requests=64,
                num_contexts=8,
                kv_len=5134,
                num_sparse_topk=2176,
            ),
        ),
    )

    note = """
    The MLA generation path does not apply the window (`MLA::enqueueGeneration` hard-codes it to -1), so a
    decode over a history longer than the window also reads aged-out ring slots. The drafter's block call
    must pass the writer's count through the block's last row as `sequence_length`, not the device KV length.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        output: torch.Tensor,
        output_sf: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_total_kv_lens: torch.Tensor,
        context_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        host_request_types: torch.Tensor,
        max_context_q_len_override: Optional[int],
        kv_cache_block_offsets: Optional[torch.Tensor],
        host_kv_cache_pool_pointers: Optional[torch.Tensor],
        host_kv_cache_pool_mapping: Optional[torch.Tensor],
        cache_indirection: Optional[torch.Tensor],
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        rotary_inv_freq: Optional[torch.Tensor],
        rotary_cos_sin: Optional[torch.Tensor],
        latent_cache: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        attention_sinks: Optional[torch.Tensor],
        is_fused_qkv: bool,
        update_kv_cache: bool,
        predicted_tokens_per_seq: int,
        local_layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        tokens_per_block: Optional[int],
        max_num_requests: int,
        max_context_length: int,
        max_seq_len: int,
        attention_window_size: int,
        beam_width: int,
        mask_type: int,
        quant_mode: int,
        q_scaling: float,
        position_embedding_type: int,
        rope_dim: int,
        rope_base: float,
        rope_scale_type: int,
        rope_scale: float,
        rope_short_m_scale: float,
        rope_long_m_scale: float,
        rope_max_positions: int,
        rope_original_max_positions: int,
        use_paged_context_fmha: bool,
        attention_input_type: Optional[int],
        is_mla_enable: bool,
        chunked_prefill_buffer_batch_size: Optional[int],
        q_lora_rank: Optional[int],
        kv_lora_rank: Optional[int],
        qk_nope_head_dim: Optional[int],
        qk_rope_head_dim: Optional[int],
        v_head_dim: Optional[int],
        rope_append: Optional[bool],
        mrope_rotary_cos_sin: Optional[torch.Tensor],
        mrope_position_deltas: Optional[torch.Tensor],
        helix_position_offsets: Optional[torch.Tensor],
        helix_is_inactive_rank: Optional[torch.Tensor],
        attention_chunk_size: Optional[int],
        softmax_stats_tensor: Optional[torch.Tensor],
        is_spec_decoding_enabled: bool,
        use_spec_decoding: bool,
        is_spec_dec_tree: bool,
        spec_decoding_generation_lengths: Optional[torch.Tensor],
        spec_decoding_position_offsets: Optional[torch.Tensor],
        spec_decoding_packed_mask: Optional[torch.Tensor],
        spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor],
        spec_decoding_bl_tree_mask: Optional[torch.Tensor],
        spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor],
        sparse_kv_indices: Optional[torch.Tensor],
        sparse_kv_offsets: Optional[torch.Tensor],
        sparse_attn_indices: Optional[torch.Tensor],
        sparse_attn_offsets: Optional[torch.Tensor],
        sparse_attn_indices_block_size: int,
        num_sparse_topk: Optional[int] = None,
        sparse_attn_kv_lens: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor_prefill: Optional[float] = None,
        skip_softmax_threshold_scale_factor_decode: Optional[float] = None,
        skip_softmax_stat: Optional[torch.Tensor] = None,
        cu_q_seqlens: Optional[torch.Tensor] = None,
        cu_kv_seqlens: Optional[torch.Tensor] = None,
        fmha_scheduler_counter: Optional[torch.Tensor] = None,
        mla_bmm1_scale: Optional[torch.Tensor] = None,
        mla_bmm2_scale: Optional[torch.Tensor] = None,
        quant_q_buffer: Optional[torch.Tensor] = None,
        flash_mla_tile_scheduler_metadata: Optional[torch.Tensor] = None,
        flash_mla_num_splits: Optional[torch.Tensor] = None,
        sage_attn_num_elts_per_blk_q: int = 0,
        sage_attn_num_elts_per_blk_k: int = 0,
        sage_attn_num_elts_per_blk_v: int = 0,
        sage_attn_qk_int8: bool = False,
        num_contexts: int = 0,
        num_ctx_tokens: int = 0,
        trtllm_gen_jit_warmup: bool = False,
        aux_kv_cache_pool_ptr: Optional[int] = None,
        is_cross: bool = False,
        cross_kv: Optional[torch.Tensor] = None,
        relative_attention_bias: Optional[torch.Tensor] = None,
        relative_attention_max_distance: int = 0,
        spec_decoding_target_max_draft_tokens: Optional[int] = None,
        quant_scale_qkv: Optional[torch.Tensor] = None,
        dsv4_inv_rope_cos_sin_cache: Optional[torch.Tensor] = None,
        enable_dsv4_epilogue_fusion: bool = False,
        # Defaults match what `attention/backends/fmha/fallback.py` passes on a dense, non-sparse path.
        max_num_sequences: Optional[int] = None,
        force_prepare_spec_dec_tree_mask: bool = False,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        skip_correction_threshold: float = 0.0,
    ) -> torch.Tensor:
        """Writes `output` in place and returns it."""
        if is_mla_enable and q_lora_rank is None:
            raise ValueError("q_lora_rank must be an int for MLA; use 0 when there is no q-LoRA.")
        FallbackFmha.attention(
            q=q,
            k=k,
            v=v,
            output=output,
            output_sf=output_sf,
            workspace=workspace,
            sequence_length=sequence_length,
            host_past_key_value_lengths=host_past_key_value_lengths,
            host_total_kv_lens=host_total_kv_lens,
            context_lengths=context_lengths,
            host_context_lengths=host_context_lengths,
            host_request_types=host_request_types,
            max_context_q_len_override=max_context_q_len_override,
            kv_cache_block_offsets=kv_cache_block_offsets,
            host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
            cache_indirection=cache_indirection,
            kv_scale_orig_quant=kv_scale_orig_quant,
            kv_scale_quant_orig=kv_scale_quant_orig,
            out_scale=out_scale,
            rotary_inv_freq=rotary_inv_freq,
            rotary_cos_sin=rotary_cos_sin,
            latent_cache=latent_cache,
            q_pe=q_pe,
            block_ids_per_seq=block_ids_per_seq,
            attention_sinks=attention_sinks,
            is_fused_qkv=is_fused_qkv,
            update_kv_cache=update_kv_cache,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            local_layer_idx=local_layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            tokens_per_block=tokens_per_block,
            max_num_requests=max_num_requests,
            max_context_length=max_context_length,
            max_seq_len=max_seq_len,
            attention_window_size=attention_window_size,
            beam_width=beam_width,
            mask_type=mask_type,
            quant_mode=quant_mode,
            q_scaling=q_scaling,
            position_embedding_type=position_embedding_type,
            rope_dim=rope_dim,
            rope_base=rope_base,
            rope_scale_type=rope_scale_type,
            rope_scale=rope_scale,
            rope_short_m_scale=rope_short_m_scale,
            rope_long_m_scale=rope_long_m_scale,
            rope_max_positions=rope_max_positions,
            rope_original_max_positions=rope_original_max_positions,
            use_paged_context_fmha=use_paged_context_fmha,
            attention_input_type=attention_input_type,
            is_mla_enable=is_mla_enable,
            chunked_prefill_buffer_batch_size=chunked_prefill_buffer_batch_size,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            rope_append=rope_append,
            mrope_rotary_cos_sin=mrope_rotary_cos_sin,
            mrope_position_deltas=mrope_position_deltas,
            helix_position_offsets=helix_position_offsets,
            helix_is_inactive_rank=helix_is_inactive_rank,
            attention_chunk_size=attention_chunk_size,
            softmax_stats_tensor=softmax_stats_tensor,
            is_spec_decoding_enabled=is_spec_decoding_enabled,
            use_spec_decoding=use_spec_decoding,
            is_spec_dec_tree=is_spec_dec_tree,
            spec_decoding_generation_lengths=spec_decoding_generation_lengths,
            spec_decoding_position_offsets=spec_decoding_position_offsets,
            spec_decoding_packed_mask=spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=spec_decoding_bl_tree_mask,
            spec_bl_tree_first_sparse_mask_offset_kv=spec_bl_tree_first_sparse_mask_offset_kv,
            sparse_kv_indices=sparse_kv_indices,
            sparse_kv_offsets=sparse_kv_offsets,
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=sparse_attn_offsets,
            sparse_attn_indices_block_size=sparse_attn_indices_block_size,
            num_sparse_topk=num_sparse_topk,
            sparse_attn_kv_lens=sparse_attn_kv_lens,
            skip_softmax_threshold_scale_factor_prefill=skip_softmax_threshold_scale_factor_prefill,
            skip_softmax_threshold_scale_factor_decode=skip_softmax_threshold_scale_factor_decode,
            skip_softmax_stat=skip_softmax_stat,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            quant_q_buffer=quant_q_buffer,
            flash_mla_tile_scheduler_metadata=flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=flash_mla_num_splits,
            sage_attn_num_elts_per_blk_q=sage_attn_num_elts_per_blk_q,
            sage_attn_num_elts_per_blk_k=sage_attn_num_elts_per_blk_k,
            sage_attn_num_elts_per_blk_v=sage_attn_num_elts_per_blk_v,
            sage_attn_qk_int8=sage_attn_qk_int8,
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
            trtllm_gen_jit_warmup=trtllm_gen_jit_warmup,
            aux_kv_cache_pool_ptr=aux_kv_cache_pool_ptr,
            is_cross=is_cross,
            cross_kv=cross_kv,
            relative_attention_bias=relative_attention_bias,
            relative_attention_max_distance=relative_attention_max_distance,
            spec_decoding_target_max_draft_tokens=spec_decoding_target_max_draft_tokens,
            quant_scale_qkv=quant_scale_qkv,
            dsv4_inv_rope_cos_sin_cache=dsv4_inv_rope_cos_sin_cache,
            enable_dsv4_epilogue_fusion=enable_dsv4_epilogue_fusion,
            max_num_sequences=max_num_sequences,
            force_prepare_spec_dec_tree_mask=force_prepare_spec_dec_tree_mask,
            kv_norm_weight=kv_norm_weight,
            kv_norm_eps=kv_norm_eps,
            skip_correction_threshold=skip_correction_threshold,
        )
        return output

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: Sequence[int],
        kv_lens: Sequence[int],
        num_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        softmax_scale: float,
        causal: bool = True,
        attention_sinks: Optional[torch.Tensor] = None,
        window: Optional[int] = None,
        e4m3_inputs: bool = False,
        out_scale: float = 1.0,
        e4m3_probabilities: bool = False,
    ) -> torch.Tensor:
        """Masked softmax attention over explicit per-sequence K/V, in fp32.

        `k`/`v` are the gathered histories the call should attend over. Causal is bottom-right aligned.
        `attention_sinks` adds an unscaled per-head logit to the denominator only. `window` keeps the newest
        `window` keys per row. `e4m3_inputs` rounds q/k/v through e4m3; `e4m3_probabilities` rounds
        `exp(score - row max)` through e4m3 at scale 448 over an fp32 row sum, as the fp8 kernels do.
        """
        rep = num_heads // num_kv_heads
        total_q = sum(q_lens)
        out = torch.full(
            (total_q, num_heads * v_head_dim), float("nan"), dtype=torch.float32, device=q.device
        )
        q_start = kv_start = 0
        for q_len, kv_len in zip(q_lens, kv_lens):
            if kv_len == 0:
                # The op leaves these rows undefined.
                q_start += q_len
                continue
            q_seq = q[q_start : q_start + q_len].reshape(q_len, num_heads, qk_head_dim).float()
            k_seq = k[kv_start : kv_start + kv_len].reshape(kv_len, num_kv_heads, qk_head_dim)
            v_seq = v[kv_start : kv_start + kv_len].reshape(kv_len, num_kv_heads, v_head_dim)
            k_seq, v_seq = k_seq.float(), v_seq.float()
            if rep > 1:
                k_seq = k_seq.repeat_interleave(rep, dim=1)
                v_seq = v_seq.repeat_interleave(rep, dim=1)
            if e4m3_inputs:
                q_seq = q_seq.to(torch.float8_e4m3fn).float()
                k_seq = k_seq.to(torch.float8_e4m3fn).float()
                v_seq = v_seq.to(torch.float8_e4m3fn).float()

            scores = torch.einsum("ihd,jhd->hij", q_seq, k_seq) * softmax_scale
            keep = self._mask(q_len, kv_len, causal, window, q.device)
            scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))

            if e4m3_probabilities:
                e4m3_max = torch.finfo(torch.float8_e4m3fn).max
                peak = scores.max(dim=-1, keepdim=True).values
                if attention_sinks is not None:
                    sink = attention_sinks.float().view(num_heads, 1)
                    peak = torch.maximum(peak.squeeze(-1), sink).unsqueeze(-1)
                weights = torch.exp(scores - peak)
                rounded = (weights * e4m3_max).to(torch.float8_e4m3fn).float() / e4m3_max
                den = weights.sum(dim=-1, keepdim=True)
                if attention_sinks is not None:
                    den = den + torch.exp(sink - peak.squeeze(-1)).unsqueeze(-1)
                probs = rounded / den
            elif attention_sinks is None:
                probs = torch.softmax(scores, dim=-1)
            else:
                sink = attention_sinks.float().view(num_heads, 1)
                peak = torch.maximum(scores.max(dim=-1).values, sink)
                weights = torch.exp(scores - peak.unsqueeze(-1))
                probs = weights / (weights.sum(-1) + torch.exp(sink - peak)).unsqueeze(-1)

            out[q_start : q_start + q_len] = (
                torch.einsum("hij,jhd->ihd", probs, v_seq).reshape(q_len, -1) * out_scale
            )
            q_start += q_len
            kv_start += kv_len
        return out.to(q.dtype)

    @staticmethod
    def _mask(q_len: int, kv_len: int, causal: bool, window: Optional[int], device) -> torch.Tensor:
        """`[q_len, kv_len]` keep-mask, bottom-right aligned."""
        if not causal and window is None:
            return torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
        keep = torch.zeros(q_len, kv_len, dtype=torch.bool, device=device)
        offset = kv_len - q_len
        for i in range(q_len):
            high = offset + i + 1 if causal else kv_len
            low = 0 if window is None else max(0, high - window)
            keep[i, low:high] = True
        return keep

    def is_valid(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        attention_sinks: Optional[torch.Tensor] = None,
        **unused_kwargs: object,
    ) -> None:
        """Keyword-only: call sites pass keywords; `**unused_kwargs` absorbs the other arguments."""
        # A non-multiple silently leaves trailing head columns zero.
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        )
        # Read as a raw buffer of exactly num_heads fp32 values; only the dtype is checked by the op.
        if attention_sinks is not None:
            assert attention_sinks.is_contiguous() and attention_sinks.numel() == num_heads, (
                f"attention_sinks must be contiguous with exactly num_heads "
                f"({num_heads}) elements, got shape "
                f"{tuple(attention_sinks.shape)} contiguous="
                f"{attention_sinks.is_contiguous()}"
            )


thop_attention = ThopAttention()
