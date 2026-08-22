# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vanilla correctness backend for DeepSeek-V4 sparse MLA."""

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    merge_attention_forward_args,
)
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttention
from tensorrt_llm.models.modeling_utils import QuantConfig

from .metadata import DeepseekV4TrtllmAttentionMetadata
from .params import DeepseekV4AttentionType, DeepSeekV4Params

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


class DeepseekV4VanillaAttention(VanillaAttention):
    """PyTorch golden for DeepSeek-V4 dual-pool selected attention.

    Selection and compression remain owned by DeepSeek-V4. The Vanilla backend
    consumes caller-provided ratio-4 compressed indices, reads the SWA and
    compressed pools using their native local index spaces, and reuses
    :meth:`VanillaAttention._selected_mla_attention` for the attention math.
    """

    Metadata = DeepseekV4TrtllmAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        q_scaling: Optional[float] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        mla_params: Optional[MLAParams] = None,
        skip_create_weights_in_init: bool = False,
        attention_chunk_size: Optional[int] = None,
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        sparse_params: Optional[DeepSeekV4Params] = None,
        dtype: Optional[torch.dtype] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> None:
        del skip_create_weights_in_init, attention_chunk_size, dtype, aux_stream
        if sparse_attention_config is None:
            sparse_attention_config = sparse_params
        if sparse_attention_config is None:
            raise ValueError(
                "sparse_attention_config or sparse_params is required for "
                "DeepseekV4VanillaAttention"
            )
        if sparse_params is None:
            sparse_params = sparse_attention_config.to_sparse_params()
        if mla_params is None:
            raise ValueError("DeepSeek-V4 attention requires MLA parameters")

        mla_params = replace(
            mla_params,
            v_head_dim=head_dim,
            rope_append=False,
        )
        super().__init__(
            layer_idx,
            num_heads,
            head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            sparse_params=sparse_params,
            **kwargs,
        )

        compress_ratios = sparse_attention_config.compress_ratios
        if layer_idx >= len(compress_ratios):
            raise ValueError(
                f"DeepSeek-V4 layer {layer_idx} has no compression ratio in "
                f"a {len(compress_ratios)}-layer configuration"
            )
        self.sparse_attention_config = sparse_attention_config
        self.compress_ratio = compress_ratios[layer_idx]
        self.window_size = sparse_attention_config.window_size
        if self.compress_ratio not in (1, 4, 128):
            raise ValueError(
                "DeepSeek-V4 Vanilla attention supports compression ratios "
                f"1, 4, and 128, got {self.compress_ratio}"
            )

    def mla_rope_generation(
        self,
        q: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        latent_cache: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        quant_q_buffer: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor] = None,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        precomputed_cu_seqlens: bool = False,
        precomputed_fmha_scheduler: bool = False,
        kv_only: bool = False,
        kv_done_elsewhere: bool = False,
        quant_scale_qkv: Optional[torch.Tensor] = None,
    ) -> None:
        """No-op counterpart of the fused TRTLLM generation preparation.

        Vanilla does not fuse RoPE or require FMHA scheduler buffers. The MLA
        module applies RoPE before invoking this method.
        """
        del (
            q,
            q_pe,
            latent_cache,
            metadata,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            out_scale,
            kv_norm_weight,
            kv_norm_eps,
            precomputed_cu_seqlens,
            precomputed_fmha_scheduler,
            kv_only,
            kv_done_elsewhere,
            quant_scale_qkv,
        )

    @staticmethod
    def _phase_sequence_range(
        metadata: DeepseekV4TrtllmAttentionMetadata,
        attention_input_type: AttentionInputType,
    ) -> tuple[int, int]:
        if attention_input_type == AttentionInputType.context_only:
            return 0, metadata.num_contexts
        if attention_input_type == AttentionInputType.generation_only:
            return metadata.num_contexts, metadata.num_seqs
        raise ValueError(
            "DeepSeek-V4 Vanilla attention requires a context-only or generation-only input"
        )

    @staticmethod
    def _page_locations(
        block_table: torch.Tensor,
        local_indices: torch.Tensor,
        tokens_per_block: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if local_indices.ndim != 1:
            raise ValueError(
                f"DeepSeek-V4 local cache indices must be 1D, got {tuple(local_indices.shape)}"
            )

        block_indices = torch.div(local_indices, tokens_per_block, rounding_mode="floor")
        page_indices = block_table.index_select(0, block_indices).to(torch.long)
        offsets = torch.remainder(local_indices, tokens_per_block)
        return page_indices, offsets

    def _gather_paged_rows(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        local_indices: torch.Tensor,
        tokens_per_block: int,
    ) -> torch.Tensor:
        if local_indices.numel() == 0:
            return cache.new_empty((0, self.head_dim))
        page_indices, offsets = self._page_locations(
            block_table,
            local_indices.to(device=block_table.device, dtype=torch.long),
            tokens_per_block,
        )
        return cache[page_indices, offsets, : self.head_dim]

    def _store_swa_rows(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        positions: torch.Tensor,
        rows: torch.Tensor,
        tokens_per_block: int,
    ) -> None:
        if positions.numel() == 0:
            return
        page_indices, offsets = self._page_locations(
            block_table,
            positions.to(device=block_table.device, dtype=torch.long),
            tokens_per_block,
        )
        cache[page_indices, offsets, : self.head_dim] = rows.to(cache.dtype)

    def _validate_inputs(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        topk_indices: Optional[torch.Tensor],
        num_phase_tokens: int,
    ) -> None:
        expected_q_dim = self.num_heads * self.head_dim
        if q.ndim != 2 or q.shape != (num_phase_tokens, expected_q_dim):
            raise ValueError(
                "DeepSeek-V4 query must have shape "
                f"[{num_phase_tokens}, {expected_q_dim}], got {tuple(q.shape)}"
            )
        if latent_cache.ndim != 2 or latent_cache.shape != (num_phase_tokens, self.head_dim):
            raise ValueError(
                "DeepSeek-V4 latent cache must have shape "
                f"[{num_phase_tokens}, {self.head_dim}], got "
                f"{tuple(latent_cache.shape)}"
            )

        if self.compress_ratio == 4:
            if topk_indices is None:
                raise ValueError(
                    "DeepSeek-V4 ratio-4 Vanilla attention requires caller-provided "
                    "compressed top-k indices"
                )
            if topk_indices.ndim != 2 or topk_indices.shape[0] != num_phase_tokens:
                raise ValueError(
                    "DeepSeek-V4 compressed top-k indices must have shape "
                    f"[{num_phase_tokens}, top_k], got {tuple(topk_indices.shape)}"
                )
            if topk_indices.dtype != torch.int32:
                raise ValueError(
                    "DeepSeek-V4 compressed top-k indices must have dtype int32, "
                    f"got {topk_indices.dtype}"
                )
            if torch.any(topk_indices < -1):
                raise ValueError("DeepSeek-V4 compressed top-k indices may only use -1 as padding")

    def _forward_sparse(
        self,
        q: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> torch.Tensor:
        seq_start, seq_end = self._phase_sequence_range(metadata, forward_args.attention_input_type)
        phase_seq_lens = metadata.seq_lens.tolist()[seq_start:seq_end]
        phase_past_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq[seq_start:seq_end]
        num_phase_tokens = sum(phase_seq_lens)

        latent_cache = forward_args.latent_cache
        if latent_cache is None:
            raise ValueError("DeepSeek-V4 Vanilla attention requires latent_cache")
        sparse_backend_args = forward_args.sparse_backend_args
        topk_indices = sparse_backend_args.topk_indices if sparse_backend_args is not None else None
        self._validate_inputs(q, latent_cache, topk_indices, num_phase_tokens)
        if self.compress_ratio == 4:
            assert topk_indices is not None
            available_compressed = torch.tensor(
                [
                    (int(past) + token_idx + 1) // self.compress_ratio
                    for q_len, past in zip(phase_seq_lens, phase_past_tokens, strict=True)
                    for token_idx in range(q_len)
                ],
                dtype=topk_indices.dtype,
                device=topk_indices.device,
            )
            if torch.any(topk_indices >= available_compressed.unsqueeze(1)):
                raise ValueError(
                    "DeepSeek-V4 compressed top-k index selects an "
                    "unavailable or future compressed entry"
                )

        cache_manager = metadata.kv_cache_manager
        if cache_manager is None:
            raise ValueError("DeepSeek-V4 Vanilla attention requires a KV cache manager")
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_kv_cache():
            raise NotImplementedError(
                "DeepSeek-V4 Vanilla attention does not support an FP8 KV cache"
            )

        swa_cache = cache_manager.get_buffers(self.layer_idx, DeepseekV4AttentionType.SWA)
        local_layer_idx = cache_manager.layer_offsets[self.layer_idx]
        swa_block_tables = metadata.sliding_block_tables[
            local_layer_idx, DeepseekV4AttentionType.SWA.value
        ]
        tokens_per_block = cache_manager.tokens_per_block

        compressed_cache = None
        compressed_block_tables = None
        compressed_tokens_per_block = 0
        if self.compress_ratio > 1:
            compressed_cache = cache_manager.get_buffers(
                self.layer_idx, DeepseekV4AttentionType.COMPRESS
            )
            compressed_block_tables = metadata.compress_block_tables[self.compress_ratio]
            compressed_tokens_per_block = cache_manager.compressed_block_sizes[self.layer_idx]

        attention_sink = forward_args.attention_sinks
        if attention_sink is None:
            attention_sink = getattr(self, "attn_sink", None)
            if isinstance(attention_sink, torch.nn.Parameter):
                attention_sink = attention_sink.data

        q = q.view(num_phase_tokens, self.num_heads, self.head_dim)
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        scale = 1.0 / (
            math.sqrt(qk_head_dim) * (self.q_scaling if self.q_scaling is not None else 1.0)
        )

        outputs = []
        token_offset = 0
        for phase_idx, (q_len, past) in enumerate(
            zip(phase_seq_lens, phase_past_tokens, strict=True)
        ):
            seq_idx = seq_start + phase_idx
            past = int(past)
            block_table_swa = swa_block_tables[seq_idx]
            latent_seq = latent_cache[token_offset : token_offset + q_len].to(q.dtype)

            past_window_start = max(0, past - self.window_size + 1)
            past_positions = torch.arange(
                past_window_start,
                past,
                device=block_table_swa.device,
                dtype=torch.long,
            )
            past_window = self._gather_paged_rows(
                swa_cache,
                block_table_swa,
                past_positions,
                tokens_per_block,
            ).to(q.dtype)

            per_token_outputs = []
            for token_idx in range(q_len):
                current_position = past + token_idx
                swa_start = max(0, current_position - self.window_size + 1)
                selected_parts = []
                if swa_start < past:
                    selected_parts.append(past_window[swa_start - past_window_start :])
                current_start = max(0, swa_start - past)
                selected_parts.append(latent_seq[current_start : token_idx + 1])

                if self.compress_ratio > 1:
                    if self.compress_ratio == 4:
                        assert topk_indices is not None
                        compressed_row = topk_indices[token_offset + token_idx]
                        compressed_indices = compressed_row[compressed_row >= 0].to(
                            device=block_table_swa.device, dtype=torch.long
                        )
                    else:
                        num_compressed = (current_position + 1) // self.compress_ratio
                        compressed_indices = torch.arange(
                            num_compressed,
                            device=block_table_swa.device,
                            dtype=torch.long,
                        )

                    if compressed_indices.numel() > 0:
                        assert compressed_cache is not None
                        assert compressed_block_tables is not None
                        selected_parts.append(
                            self._gather_paged_rows(
                                compressed_cache,
                                compressed_block_tables[seq_idx],
                                compressed_indices,
                                compressed_tokens_per_block,
                            ).to(q.dtype)
                        )

                selected_latent = torch.cat(selected_parts, dim=0)
                per_token_outputs.append(
                    self._selected_mla_attention(
                        q[token_offset + token_idx],
                        selected_latent,
                        value_dim=self.v_head_dim,
                        scale=scale,
                        attention_sink=attention_sink,
                    )
                )

            outputs.append(
                torch.stack(per_token_outputs).reshape(q_len, self.num_heads * self.v_head_dim)
            )

            total_length = past + q_len
            first_stored_position = max(past, total_length - self.window_size)
            stored_positions = torch.arange(
                first_stored_position,
                total_length,
                device=block_table_swa.device,
                dtype=torch.long,
            )
            stored_rows = latent_seq[first_stored_position - past :]
            self._store_swa_rows(
                swa_cache,
                block_table_swa,
                stored_positions,
                stored_rows,
                tokens_per_block,
            )
            token_offset += q_len

        result = torch.cat(outputs, dim=0)
        if forward_args.output is not None:
            if forward_args.output.shape != result.shape:
                raise ValueError(
                    "DeepSeek-V4 output buffer must have shape "
                    f"{tuple(result.shape)}, got {tuple(forward_args.output.shape)}"
                )
            forward_args.output.copy_(result)
            return forward_args.output
        return result

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ) -> torch.Tensor:
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        if k is not None or v is not None:
            raise ValueError(
                "DeepSeek-V4 Vanilla attention expects absorbed queries and "
                "latent cache, not explicit K/V tensors"
            )
        if metadata.multi_item_part_lens is not None:
            raise ValueError("DeepSeek-V4 Vanilla attention does not support multi-item scoring")
        return self._forward_sparse(q, metadata, forward_args)
