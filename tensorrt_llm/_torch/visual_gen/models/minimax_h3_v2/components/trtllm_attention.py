# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Uncached full and block-sparse TRTLLM attention for H3."""

import math
import os

import torch
from torch.nn import functional as F

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.attention.backends.sparse.params import (
    BlockSparseForwardInputs,
    SparseBackendForwardArgs,
    SparseParams,
)
from tensorrt_llm._torch.attention.backends.sparse.timestep_phase import graph_phase_for_timestep
from tensorrt_llm._torch.modeling_v2.catalog.attention.thop_attention import thop_attention
from tensorrt_llm._torch.visual_gen.attention.layout import AttentionTensorLayout
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig

from .trtllm_attention_prims import H3PrimTS


@torch.compile
def _pack(q, k, v):
    return torch.cat((q.flatten(2), k.flatten(2), v.flatten(2)), dim=-1).flatten(0, 1)


@torch.compile
def _compact(q, k, v):
    return q.contiguous(), k.contiguous(), v.contiguous()


class H3TrtllmAttention:
    """Full self-attention on NHD activations with one host prefix length per batch."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        attention_metadata_state: dict,
        layer_idx: int = 0,
        num_kv_heads: int | None = None,
        quant_config: QuantConfig | None = None,
        dtype: torch.dtype | None = None,
        attention_config: AttentionConfig | None = None,
        sparse_params: SparseParams | None = None,
    ):
        if attention_metadata_state is None:
            raise ValueError("TRTLLM attention requires model-scoped attention metadata state.")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.quant_mode = int(quant_config.layer_quant_mode) if quant_config is not None else 0
        self.recipe = attention_config.quant_attention_config if attention_config else None
        self.sparse = sparse_params
        self.sol = sparse_params is not None and sparse_params.algorithm == "sol_attn"
        self.skip_softmax = sparse_params is not None and sparse_params.algorithm == "skip_softmax"
        self.state = attention_metadata_state
        self.prims = H3PrimTS(self.recipe, attention_metadata_state)
        self.skip_stats = None
        self.print_stats = os.getenv("TRTLLM_PRINT_SKIP_SOFTMAX_STAT") == "1"

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return AttentionTensorLayout.NHD

    def support_fused_qkv(self) -> bool:
        return self.recipe is None and not self.sol

    @torch.compiler.disable
    def _skip_params(self, timestep):
        return self.sparse.scheduler.get_runtime_params(timestep=timestep)

    @torch.compiler.disable
    def _sol_routes(self, q, k, v, timestep):
        from tensorrt_llm._torch.visual_gen.attention.backends.sparse.sol import predictor

        if self.layer_idx in self.sparse.dense_layers:
            return None
        phase = graph_phase_for_timestep(
            timestep, disabled_until_timestep=self.sparse.disabled_until_timestep
        )
        if phase == 0:
            return None
        pred = predictor.predict(
            q,
            k,
            v,
            tau=self.sparse.tau,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            thresh_type=self.sparse.thresh_type,
        )
        return BlockSparseForwardInputs(
            q_block_size=predictor.BLOCK_SIZE,
            kv_block_size=predictor.BLOCK_SIZE,
            exact_block_bits=pred.exact_block_bits,
            k_summary=pred.k_summary,
            v_summary=pred.v_summary,
        )

    @torch.compiler.disable
    def _thop(self, q, k, v, batch, length, sparse):
        if not self.prims.fallback_enabled:
            raise RuntimeError("Uncached H3 context attention requires the fallback FMHA library.")
        cache = self.state.setdefault("h3_thop", {})
        key = (q.device, batch, length)
        if key not in cache:
            lens = torch.full((batch,), length, dtype=torch.int32, device="cpu")
            cache[key] = (
                lens,
                lens.to(q.device),
                torch.zeros_like(lens),
                torch.tensor([batch * length, 0], dtype=torch.int32, device="cpu"),
                torch.empty(0, dtype=torch.int8, device=q.device),
                torch.ones(1, dtype=torch.float32, device=q.device),
            )
        host_lens, device_lens, types, totals, workspace, kv_scale = cache[key]
        if self.skip_stats is None or self.skip_stats.device != q.device:
            self.skip_stats = q.new_zeros((2,), dtype=torch.uint32)
        if self.print_stats:
            self.skip_stats.zero_()
        out = q.new_empty((batch * length, self.num_heads * self.head_dim))
        thop_attention(
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
            local_layer_idx=self.layer_idx,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            tokens_per_block=None,
            max_num_requests=batch,
            max_context_length=length,
            max_seq_len=length,
            attention_window_size=length,
            beam_width=1,
            mask_type=0,
            quant_mode=self.quant_mode,
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
            v_head_dim=self.head_dim,
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
            skip_softmax_threshold_scale_factor_prefill=sparse.threshold_scale_factor_prefill
            if sparse
            else None,
            skip_softmax_threshold_scale_factor_decode=sparse.threshold_scale_factor_decode
            if sparse
            else None,
            skip_softmax_stat=self.skip_stats,
            sage_attn_num_elts_per_blk_q=self.recipe.q_block_size if self.recipe else 0,
            sage_attn_num_elts_per_blk_k=self.recipe.k_block_size if self.recipe else 0,
            sage_attn_num_elts_per_blk_v=self.recipe.v_block_size if self.recipe else 0,
            sage_attn_qk_int8=self.recipe is not None and self.recipe.qk_dtype == "int8",
            sage_attn_smooth_k=self.recipe is not None and self.recipe.smooth_k,
            num_contexts=batch,
            num_ctx_tokens=batch * length,
        )
        if self.print_stats:
            total, skipped = self.skip_stats.tolist()
            if total:
                print(
                    f"SKIP_SOFTMAX_STAT: layer{self.layer_idx}: {skipped}/{total} = {skipped / total:.2f}%"
                )
        return out.view(batch, length, -1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        *,
        batch_size: int,
        seq_len: int,
        seq_len_kv: int | None = None,
        valid_seq_len: int | None = None,
        discard_padded_queries: bool = False,
        key_padding_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        sparse_backend_args: SparseBackendForwardArgs | None = None,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        if seq_len_kv not in (None, seq_len) or attention_mask != PredefinedAttentionMask.FULL:
            raise ValueError("H3 TRTLLM attention requires full self-attention.")
        if discard_padded_queries and (valid_seq_len is not None or key_padding_mask is not None):
            if (
                self.recipe is not None
                or (self.sparse is not None and not self.sol)
                or sparse_backend_args is not None
            ):
                raise ValueError(
                    "TRTLLM padding requires dense self-attention with discarded padded queries."
                )
            if type(valid_seq_len) is not int or not 0 <= valid_seq_len <= seq_len:
                raise ValueError("valid_seq_len must be a host integer in [0, sequence length].")
            if valid_seq_len < seq_len:
                if valid_seq_len == 0:
                    return q.new_zeros((batch_size, seq_len, self.num_heads * self.head_dim))
                out = self.forward(
                    q[:, :valid_seq_len].contiguous(),
                    k[:, :valid_seq_len].contiguous() if k is not None else None,
                    v[:, :valid_seq_len].contiguous() if v is not None else None,
                    batch_size=batch_size,
                    seq_len=valid_seq_len,
                    timestep=timestep,
                )
                return F.pad(out, (0, 0, 0, seq_len - valid_seq_len))
        sparse = self._skip_params(timestep) if self.skip_softmax else None
        routes = sparse_backend_args.block_sparse_inputs if sparse_backend_args else None
        if self.sol or routes is not None or self.recipe is not None:
            if k is None or v is None:
                raise ValueError("Quantized and sparse H3 attention require separate Q, K and V.")
            q, k, v = _compact(q, k, v)
            if self.sol:
                routes = self._sol_routes(q, k, v, timestep)
            if routes is not None or self.recipe is not None:
                out = self.prims(
                    q,
                    k,
                    v,
                    routes,
                    skip_softmax=self.skip_softmax,
                    skip_threshold_active=bool(
                        sparse
                        and (
                            sparse.threshold_scale_factor_prefill
                            or sparse.threshold_scale_factor_decode
                        )
                    ),
                )
                if out is not None:
                    return out.flatten(2)
                if routes is not None:
                    raise RuntimeError(
                        "H3 block-sparse attention requires PrimTS block-sparse FMHA."
                    )
            if self.recipe:
                return self._thop(
                    q.flatten(0, 1).flatten(1),
                    k.flatten(0, 1).flatten(1),
                    v.flatten(0, 1).flatten(1),
                    batch_size,
                    seq_len,
                    sparse,
                )
        qkv = q.reshape(batch_size * seq_len, -1) if k is None else _pack(q, k, v)
        return self._thop(qkv, None, None, batch_size, seq_len, sparse)
