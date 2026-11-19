# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 DSA sparse MLA attention: the backend declaration and the per-layer attention object.

``K31DSABackend`` is the engine's DSA backend class with the package's metadata class, which the engine builds every
step; a target declares it (``RuntimeHooks.attention_backend_cls``). ``K31MLAAttention`` is the per-layer object the
MLA module calls (``mla.mqa``). It owns the DSA indexer, the identity RoPE table (K3.1 is NoPE) and the attention
call's fixed arguments, and runs one phase of the absorbed sparse MLA: the indexer's Top-K as pool rows, then the
catalog ``thop_attention`` with the arguments the TRT-LLM attention op takes on this path (an FP8 or BF16 latent pool,
separate Q, the latent cache appended in the op, no FlashMLA, no sinks, no paged-context FMHA).
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.fmha.phased import get_spec_decoding_position_offsets
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention
from tensorrt_llm._torch.modeling_v2.catalog.attention.mla_rope_append_paged_kv_assign_q import (
    mla_rope_append_paged_kv_assign_q,
)
from tensorrt_llm._torch.modeling_v2.catalog.attention.thop_attention import thop_attention
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer import (
    K31Indexer,
    transform_local_topk_and_prepare_pool_view,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_metadata import (
    K31DSAtrtllmAttentionMetadata,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_params import k31_dsa_params
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

#: The causal mask type of the attention op.
_MASK_CAUSAL = 1
#: SMs whose MLA kernels take a skip-correction threshold; elsewhere it is 0 (off).
_SKIP_CORRECTION_SMS = (100, 103)


class K31PerfDSAtrtllmAttentionMetadata(K31DSAtrtllmAttentionMetadata):
    """The DSA metadata of the SM 10.3 perf targets (``_k31_perf``)."""

    _k31_perf = True


class K31DSABackend(DSATrtllmAttention):
    """The DSA backend class the targets declare (``RuntimeHooks.attention_backend_cls``): the engine's DSA backend
    with the package's metadata class. The MLA layers attend through ``K31MLAAttention``, not instances of it."""

    Metadata = K31DSAtrtllmAttentionMetadata

    @classmethod
    def support_fused_rope(cls) -> bool:
        return True

    @classmethod
    def support_mla(cls) -> bool:
        return True

    @classmethod
    def support_fp4_kv_cache(cls) -> bool:
        return False


class K31PerfDSABackend(K31DSABackend):
    """The SM 10.3 perf targets' declared DSA backend class: their metadata class."""

    Metadata = K31PerfDSAtrtllmAttentionMetadata


def attention_backend_for(backend_name: str, sparse_params, perf: bool = False) -> Optional[type]:
    """The package's backend class for the engine's backend name and sparse parameters (the perf targets' with
    ``perf``), or None unless they are the TRTLLM DSA ones; ``K31MLAAttention`` checks its arguments with it. The
    targets declare their class in ``RuntimeHooks.attention_backend_cls``."""
    if backend_name.upper() == "TRTLLM" and getattr(sparse_params, "algorithm", None) == "dsa":
        return K31PerfDSABackend if perf else K31DSABackend
    return None


def _skip_correction_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if threshold <= 0.0:
        return 0.0
    sm = get_sm_version()
    if sm in _SKIP_CORRECTION_SMS:
        return threshold
    logger.warning_once(
        f"Skip-correction is supported only on SM100 and SM103; disabling it on SM{sm}.",
        key="skip_correction_unsupported_sm",
    )
    return 0.0


def _write_identity_rope_values(cos_sin: torch.Tensor) -> None:
    """Overwrite an interleaved (cos, sin) table with cos = 1, sin = 0: the MLA rope kernels' rotation becomes a
    copy (K3.1 is NoPE). The write reaches device memory before any other stream reads the table."""
    flat = cos_sin.reshape(-1)
    with torch.no_grad():
        flat[0::2] = 1.0
        flat[1::2] = 0.0
    if cos_sin.is_cuda:
        torch.cuda.synchronize(cos_sin.device)


class K31MLAAttention:
    """One K3.1 MLA layer's DSA sparse attention (``mla.mqa``): extracted glue, not an engine class.

    ``num_heads`` query heads (the padded per-rank heads), one latent KV head of ``kv_lora_rank +
    qk_rope_head_dim`` channels, an FP8 or BF16 latent pool. ``forward`` runs one phase (context or generation rows)
    with the same kernels, in the same order and with the same arguments as the engine's TRT-LLM attention backend on
    this path: the indexer's Top-K and its pool-row remap, then the attention op (its FMHA selection serves every K3.1
    call with the thop op).
    """

    #: The SM 10.3 perf targets' attention: its indexer runs the perf Top-K paths and a generation call's decode
    #: Top-K writes ordered pool rows directly.
    _k31_perf = False
    #: Pool rows a caller already predicted for the next call (set and cleared by the caller around ``forward``):
    #: that call attends over them instead of running the indexer again.
    _k31_predicted_topk: Optional[torch.Tensor] = None

    is_mla_enable = True

    def __init__(
        self,
        backend_name: str,
        layer_idx: int,
        num_heads: int,
        *,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        hidden_size: int,
        predicted_tokens_per_seq: int,
        q_scaling: float,
        pos_embd_params: PositionalEmbeddingParams,
        quant_config: Optional[QuantConfig],
        sparse_params,
        dtype: Optional[torch.dtype],
        aux_stream: Optional[torch.cuda.Stream],
        kv_cache_dtype: str = "auto",
        skip_correction_threshold: float = 0.0,
        perf: bool = False,
        logits_pdl: bool = True,
    ) -> None:
        if attention_backend_for(backend_name, sparse_params) is None:
            raise ValueError(
                "the Kimi K3.1 MLA attends through the TRTLLM backend's DSA sparse attention only"
            )
        if kv_cache_dtype == "fp8_ds_mla":
            raise ValueError(
                "the Kimi K3.1 MLA latent KV cache is the paged FP8 one, not fp8_ds_mla"
            )
        self._k31_perf = perf
        self.layer_idx = layer_idx
        self.local_layer_idx: Optional[int] = None
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.head_dim = kv_lora_rank + qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        # The absorbed attention's value head is the latent; the rope tail is appended to the latent cache.
        self.v_head_dim = kv_lora_rank
        self.rope_append = True
        self.predicted_tokens_per_seq = predicted_tokens_per_seq
        self.q_scaling = q_scaling or 1.0
        self.sparse_params = k31_dsa_params(sparse_params)
        self.skip_correction_threshold = _skip_correction_threshold(skip_correction_threshold)

        # The rope table the MLA rope kernels read: the shape of the configured RoPE, identity values.
        self.position_embedding_type = int(pos_embd_params.type)
        self.rope_params = pos_embd_params.rope
        self.rotary_inv_freq, self.rotary_cos_sin = self.rope_params.create_rope_const_params()
        if self.rotary_cos_sin is None:
            raise RuntimeError("the Kimi K3.1 MLA needs a RoPE table shape (rope dim > 0)")
        _write_identity_rope_values(self.rotary_cos_sin)
        self.skip_softmax_stat = torch.zeros(2, dtype=torch.uint32, device="cuda")
        # Layer-level FP8 KV-cache scales of 1.0: the op always receives valid scale pointers.
        self.kv_cache_scaling_factor = torch.ones(1, dtype=torch.float32, device="cuda")
        self.kv_scale_quant_orig = self.kv_cache_scaling_factor
        self.kv_scale_orig_quant = 1.0 / self.kv_cache_scaling_factor
        self.update_quant_config(quant_config)

        # Cross-layer indexer sharing: only "full" layers own an indexer; "shared" layers reuse the previous full
        # layer's Top-K (``metadata.shared_topk_indices``).
        self.is_full_indexer_layer = getattr(self.sparse_params, "is_full_indexer_layer", True)
        self.indexer = None
        if self.is_full_indexer_layer:
            self.indexer = K31Indexer(
                MLAParams(
                    q_lora_rank=q_lora_rank,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=qk_rope_head_dim,
                    qk_nope_head_dim=qk_nope_head_dim,
                    v_head_dim=kv_lora_rank,
                    rope_append=True,
                    predicted_tokens_per_seq=predicted_tokens_per_seq,
                    hidden_size=hidden_size,
                ),
                self.sparse_params,
                dtype=dtype,
                layer_idx=layer_idx,
                aux_stream=aux_stream,
            )
            self.indexer._k31_perf = perf
            self.indexer._k31_logits_pdl = logits_pdl

    def update_quant_config(self, quant_config: Optional[QuantConfig]) -> None:
        self.quant_config = quant_config or QuantConfig()
        layer_quant_mode = self.quant_config.layer_quant_mode
        self.quant_mode = int(layer_quant_mode)
        self.has_fp8_kv_cache = layer_quant_mode.has_fp8_kv_cache()
        if layer_quant_mode.has_fp4_kv_cache():
            raise ValueError("the Kimi K3.1 MLA latent KV cache is FP8 or BF16, not NVFP4")

    def get_local_layer_idx(self, metadata: K31DSAtrtllmAttentionMetadata) -> int:
        if self.local_layer_idx is not None:
            return self.local_layer_idx
        if metadata.kv_cache_manager is None:
            return self.layer_idx
        self.local_layer_idx = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
        return self.local_layer_idx

    def ensure_rope_table_size(self, required_max_positions: int) -> None:
        """Grow the identity rope table to ``required_max_positions`` (the rope kernels index it up to the longest
        sequence); a regenerated table is rewritten to identity values."""
        rope = self.rope_params
        if required_max_positions <= rope.max_positions:
            return
        floats_per_position = rope.dim * (2 if rope.duplicate_data else 1)
        table_max_positions = self.rotary_cos_sin.numel() // floats_per_position
        rope.max_positions = max(rope.max_positions, table_max_positions, required_max_positions)
        if required_max_positions > table_max_positions:
            self.rotary_inv_freq, self.rotary_cos_sin = rope.create_rope_const_params()
        _write_identity_rope_values(self.rotary_cos_sin)

    def mla_rope_append_paged_kv_assign_q(
        self, q: torch.Tensor, latent_cache: torch.Tensor, metadata: K31DSAtrtllmAttentionMetadata
    ) -> None:
        """The context rows' latent append to the paged pool and the RoPE of ``q``'s rope columns (identity), through
        the catalog ``mla_rope_append_paged_kv_assign_q`` with the arguments of the engine's DSA backend (an FP8 or
        BF16 pool: no NVFP4 KV scale, beam width 1)."""
        kv_cache_manager = metadata.kv_cache_manager
        assert kv_cache_manager is not None
        mla_rope_append_paged_kv_assign_q(
            q,
            latent_cache,
            metadata.num_contexts,
            metadata.ctx_cached_token_indptr,
            metadata.ctx_kv_indptr,
            metadata.max_ctx_seq_len,
            self.rotary_cos_sin,
            self.num_heads,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.kv_lora_rank,
            metadata.kv_cache_block_offsets,
            kv_cache_manager.kv_cache_pool_pointers,
            kv_cache_manager.kv_cache_pool_mapping,
            None,
            kv_cache_manager.mla_kv_cache_residual_dim,
            self.get_local_layer_idx(metadata),
            kv_cache_manager.tokens_per_block,
            kv_cache_manager.max_seq_len,
            1,
            self.quant_mode,
        )

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        metadata: K31DSAtrtllmAttentionMetadata,
        is_generation: bool,
        sparse_backend_args,
    ) -> torch.Tensor:
        """This call's attended pool rows: the DSA indexer's Top-K (or the shared layer's, or the caller's
        prediction) remapped from request-local positions to pool rows."""
        if self._k31_predicted_topk is not None:
            return self._k31_predicted_topk
        phase_start = metadata.num_ctx_tokens if is_generation else 0
        phase_end = metadata.num_tokens if is_generation else metadata.num_ctx_tokens
        shared_topk_indices = metadata.shared_topk_indices
        indexer = self.indexer
        if indexer is None:
            topk_indices = shared_topk_indices[phase_start:phase_end]
        else:
            indexer._gen_rows_paged = False
            indexer._gen_pool_remap = None
            if self._k31_perf and is_generation and shared_topk_indices is None:
                metadata._ensure_pool_view_cached()
                scale, layer_offset = metadata.kv_cache_manager.get_primary_pool_page_index_params(
                    self.get_local_layer_idx(metadata)
                )
                tpb = metadata._cached_tokens_per_block
                indexer._gen_pool_remap = (
                    metadata._cached_block_table_gen,
                    tpb,
                    scale * tpb,
                    layer_offset,
                )
            try:
                topk_indices = indexer.forward_from_projected(
                    metadata,
                    q,
                    sparse_backend_args.indexer_intermediates,
                    is_generation=is_generation,
                )
            finally:
                indexer._gen_pool_remap = None
            if shared_topk_indices is not None:
                shared_topk_indices[
                    phase_start : phase_start + topk_indices.shape[0],
                    : topk_indices.shape[1],
                ].copy_(topk_indices)
            if indexer._gen_rows_paged:
                # The decode Top-K already wrote ordered pool rows.
                return topk_indices
        topk_indices_global, _ = transform_local_topk_and_prepare_pool_view(
            topk_indices, metadata, self.get_local_layer_idx(metadata), is_generation
        )
        return topk_indices_global

    def forward(
        self,
        q: torch.Tensor,
        metadata: K31DSAtrtllmAttentionMetadata,
        *,
        attention_input_type: AttentionInputType,
        latent_cache: torch.Tensor,
        q_pe: torch.Tensor,
        sparse_backend_args=None,
        out_scale: Optional[torch.Tensor] = None,
        cu_q_seqlens: Optional[torch.Tensor] = None,
        cu_kv_seqlens: Optional[torch.Tensor] = None,
        fmha_scheduler_counter: Optional[torch.Tensor] = None,
        mla_bmm1_scale: Optional[torch.Tensor] = None,
        mla_bmm2_scale: Optional[torch.Tensor] = None,
        quant_q_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The latent attention output ``[rows, num_heads * kv_lora_rank]`` of ``q`` (``[rows, num_heads *
        (kv_lora_rank + qk_rope_head_dim)]``, the absorbed query) for one phase's rows.

        Context rows: the op rotates ``q_pe``, appends ``latent_cache`` to the pool and attends. Generation rows:
        ``mla_rope_generation`` already appended the latent and filled the scheduler buffers, the BMM scales and the
        FP8 query (``cu_q_seqlens`` .. ``quant_q_buffer``)."""
        if out_scale is not None:
            raise ValueError("the Kimi K3.1 MLA attention output is not quantized")
        if metadata.enable_flash_mla:
            raise ValueError("the Kimi K3.1 MLA does not run FlashMLA")
        is_generation = attention_input_type == AttentionInputType.generation_only
        if not is_generation and attention_input_type != AttentionInputType.context_only:
            raise ValueError("the MLA attention runs context-only or generation-only calls")
        # The MLA context attends separate Q / KV, not through the paged-context FMHA.
        metadata.use_paged_context_fmha = False
        output = q.new_empty((q.size(0), self.num_heads * self.kv_lora_rank))
        if cu_q_seqlens is None:
            cu_q_seqlens = metadata.cu_q_seqlens
        if cu_kv_seqlens is None:
            cu_kv_seqlens = metadata.cu_kv_seqlens

        sparse_attn_indices = self.sparse_attn_predict(
            q, metadata, is_generation, sparse_backend_args
        )

        local_layer_idx = self.get_local_layer_idx(metadata)
        if local_layer_idx == 0 and (
            metadata.spec_bl_tree_first_sparse_mask_offset_kv is not None
            and metadata._seq_lens_cuda is not None
        ):
            metadata.update_blackwell_first_sparse_mask_offset()
        self.ensure_rope_table_size(metadata.max_seq_len)
        self.local_layer_idx = local_layer_idx
        if metadata.spec_decoding_bl_tree_mask is not None and local_layer_idx == 0:
            metadata.spec_decoding_bl_tree_mask.zero_()

        attention_window_size = None
        if metadata.kv_cache_manager is not None:
            attention_window_size = metadata.kv_cache_manager.max_attention_window_vec[
                local_layer_idx
            ]
        if attention_window_size is None:
            attention_window_size = metadata.max_seq_len
        if metadata.max_context_q_len_override is not None:
            raise ValueError("the Kimi K3.1 MLA has no encoder CUDA graphs")

        rope = self.rope_params
        if metadata.is_cuda_graph:
            metadata.workspace_reclaimable = False
        thop_attention(
            q=q,
            k=None,
            v=None,
            output=output,
            output_sf=None,
            workspace=metadata.effective_workspace,
            sequence_length=metadata.kv_lens_cuda_runtime,
            host_past_key_value_lengths=metadata.kv_lens_runtime,
            host_total_kv_lens=metadata.host_total_kv_lens,
            context_lengths=metadata.prompt_lens_cuda_runtime,
            host_context_lengths=metadata.prompt_lens_cpu_runtime,
            host_request_types=metadata.host_request_types_runtime,
            max_context_q_len_override=None,
            kv_cache_block_offsets=metadata.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=metadata.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=metadata.host_kv_cache_pool_mapping,
            cache_indirection=metadata.cache_indirection,
            kv_scale_orig_quant=self.kv_scale_orig_quant,
            kv_scale_quant_orig=self.kv_scale_quant_orig,
            out_scale=None,
            rotary_inv_freq=self.rotary_inv_freq,
            rotary_cos_sin=self.rotary_cos_sin,
            latent_cache=latent_cache,
            q_pe=q_pe,
            block_ids_per_seq=metadata.block_ids_per_seq,
            attention_sinks=None,
            is_fused_qkv=False,
            update_kv_cache=True,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
            local_layer_idx=local_layer_idx,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            tokens_per_block=metadata.tokens_per_block,
            max_num_requests=metadata.max_num_requests,
            max_context_length=metadata.max_context_length,
            max_seq_len=metadata.max_seq_len,
            attention_window_size=attention_window_size,
            beam_width=metadata.effective_beam_width,
            mask_type=_MASK_CAUSAL,
            quant_mode=self.quant_mode,
            q_scaling=self.q_scaling,
            position_embedding_type=self.position_embedding_type,
            rope_dim=rope.dim,
            rope_base=rope.theta,
            rope_scale_type=int(rope.scale_type),
            rope_scale=rope.scale,
            rope_short_m_scale=rope.short_m_scale,
            rope_long_m_scale=rope.long_m_scale,
            rope_max_positions=rope.max_positions,
            rope_original_max_positions=rope.original_max_positions,
            use_paged_context_fmha=metadata.use_paged_context_fmha,
            attention_input_type=int(attention_input_type),
            is_mla_enable=True,
            chunked_prefill_buffer_batch_size=1,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            rope_append=self.rope_append,
            mrope_rotary_cos_sin=None,
            mrope_position_deltas=None,
            helix_position_offsets=metadata.helix_position_offsets,
            helix_is_inactive_rank=metadata.helix_is_inactive_rank,
            attention_chunk_size=None,
            softmax_stats_tensor=None,
            is_spec_decoding_enabled=metadata.is_spec_decoding_enabled,
            use_spec_decoding=metadata.use_spec_decoding,
            is_spec_dec_tree=metadata.is_spec_dec_tree,
            spec_decoding_generation_lengths=metadata.spec_decoding_generation_lengths,
            spec_decoding_position_offsets=get_spec_decoding_position_offsets(metadata),
            spec_decoding_packed_mask=metadata.spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=metadata.spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=metadata.spec_decoding_bl_tree_mask,
            spec_bl_tree_first_sparse_mask_offset_kv=metadata.spec_bl_tree_first_sparse_mask_offset_kv,
            sparse_kv_indices=None,
            sparse_kv_offsets=None,
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=None,
            sparse_attn_indices_block_size=self.sparse_params.indices_block_size,
            num_sparse_topk=metadata.num_sparse_topk,
            sparse_attn_kv_lens=None,
            skip_softmax_threshold_scale_factor_prefill=0.0,
            skip_softmax_threshold_scale_factor_decode=0.0,
            skip_softmax_stat=self.skip_softmax_stat,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            quant_q_buffer=quant_q_buffer,
            flash_mla_tile_scheduler_metadata=metadata.flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=metadata.flash_mla_num_splits,
            num_contexts=metadata.num_contexts,
            num_ctx_tokens=metadata.num_ctx_tokens,
            trtllm_gen_jit_warmup=metadata.trtllm_gen_jit_warmup,
            aux_kv_cache_pool_ptr=None,
            is_cross=metadata.is_cross,
            cross_kv=None,
            relative_attention_bias=None,
            relative_attention_max_distance=0,
            spec_decoding_target_max_draft_tokens=metadata.max_total_draft_tokens,
            quant_scale_qkv=None,
            dsv4_inv_rope_cos_sin_cache=None,
            enable_dsv4_epilogue_fusion=False,
            max_num_sequences=metadata.max_num_sequences,
            force_prepare_spec_dec_tree_mask=metadata.force_prepare_spec_dec_tree_mask,
            kv_norm_weight=None,
            kv_norm_eps=1e-6,
            skip_correction_threshold=self.skip_correction_threshold,
        )
        return output
