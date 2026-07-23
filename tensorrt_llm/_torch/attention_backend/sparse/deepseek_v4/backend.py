# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    merge_attention_forward_args,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm.models.modeling_utils import QuantConfig

from .cache_manager import get_token_bytes
from .compressor import Compressor
from .indexer import DeepseekV4Indexer
from .kernels import deepseek_v4_local_to_global_indices
from .metadata import DeepseekV4TrtllmAttentionMetadata
from .params import DeepseekV4AttentionType, DeepSeekV4Params

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


class DeepseekV4TrtllmAttention(TrtllmAttention):
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
    ):
        if sparse_attention_config is None:
            sparse_attention_config = sparse_params
        assert sparse_attention_config is not None, (
            "sparse_attention_config is required for DeepseekV4TrtllmAttention and cannot be None"
        )
        if sparse_params is None:
            sparse_params = sparse_attention_config.to_sparse_params()
        assert sparse_params is not None, (
            "sparse_params is required for DeepseekV4TrtllmAttention and cannot be None"
        )
        assert mla_params is not None, "DeepSeek-V4 attention requires MLA parameters"
        mla_params = replace(
            mla_params,
            v_head_dim=head_dim,
            rope_append=False,
        )
        TrtllmAttention.__init__(
            self,
            layer_idx,
            num_heads,
            head_dim,
            sparse_params=sparse_params,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            skip_create_weights_in_init=skip_create_weights_in_init,
            attention_chunk_size=attention_chunk_size,
            **kwargs,
        )

        self.sparse_attention_config = sparse_attention_config
        self.compress_ratio = sparse_attention_config.compress_ratios[layer_idx]

        if self.compress_ratio == 4:
            self.indexer = DeepseekV4Indexer(
                quant_config,
                pos_embd_params,
                mla_params,
                skip_create_weights_in_init,
                sparse_attention_config,
                dtype,
                self.compress_ratio,
                layer_idx,
                aux_stream,
            )

        if self.compress_ratio > 1:
            rms_norm_eps = 1e-6
            has_fp8_kv_cache = False
            if quant_config is not None:
                has_fp8_kv_cache = quant_config.layer_quant_mode.has_fp8_kv_cache()
            kv_cache_dtype = "fp8_pertensor" if has_fp8_kv_cache else "default"
            self.compressor = Compressor(
                mla_params,
                layer_idx,
                self.compress_ratio,
                rms_norm_eps,
                skip_create_weights_in_init,
                pos_embd_params,
                kv_cache_dtype=kv_cache_dtype,
                dtype=dtype,
                rotate_activation=False,
            )

    def _prepare_sparse_forward_args(
        self,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> None:
        attention_input_type = forward_args.attention_input_type
        if attention_input_type == AttentionInputType.context_only:
            start_idx = 0
            end_idx = metadata.num_ctx_tokens
        elif attention_input_type == AttentionInputType.generation_only:
            start_idx = metadata.num_ctx_tokens
            end_idx = metadata.num_tokens
        else:
            start_idx = 0
            end_idx = metadata.num_tokens

        sparse_args = forward_args.sparse_runtime_params
        sparse_args.sparse_mla_topk_lens = metadata.sparse_mla_topk_lens[self.compress_ratio][
            start_idx:end_idx
        ]
        if self.compress_ratio > 1:
            sparse_args.compressed_kv_cache_pool_ptr = metadata.sparse_mla_base_ptrs[
                self.compress_ratio
            ]
        else:
            sparse_args.compressed_kv_cache_pool_ptr = None

        metadata.num_sparse_topk = (
            self.sparse_attention_config.window_size
            + metadata.max_compressed_indices[self.compress_ratio]
        )

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ):
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        attn_sink = getattr(self, "attn_sink", None)
        if attn_sink is not None:
            if forward_args.attention_sinks is None:
                forward_args = replace(forward_args, attention_sinks=attn_sink.data)

        self._prepare_sparse_forward_args(metadata, forward_args)
        return super().forward(q, k, v, metadata, forward_args=forward_args)

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Convert local indices (SWA + compressed) to global pool indices."""
        layer_idx = self.layer_idx
        kv_cache_manager = metadata.kv_cache_manager
        attention_input_type = forward_args.attention_input_type

        swa_pool_base_ptr = metadata.sparse_mla_base_ptrs[1]

        # Get cached buffer pointers
        swa_buffer_ptr = metadata.swa_buffer_ptrs[layer_idx]

        # Token stride
        index_head_dim = self.sparse_attention_config.index_head_dim
        has_fp8_kv_cache = False
        if self.quant_config is not None:
            has_fp8_kv_cache = self.quant_config.layer_quant_mode.has_fp8_kv_cache()
        token_stride = get_token_bytes(
            self.head_dim,
            index_head_dim,
            self.compress_ratio,
            DeepseekV4AttentionType.SWA,
            has_fp8_kv_cache,
        )

        # Select token range based on phase
        if attention_input_type == AttentionInputType.context_only:
            start_idx = 0
            end_idx = metadata.num_ctx_tokens
        elif attention_input_type == AttentionInputType.generation_only:
            start_idx = metadata.num_ctx_tokens
            end_idx = metadata.num_tokens
        else:
            start_idx = 0
            end_idx = metadata.num_tokens

        # Use global req_id directly
        req_id = metadata.req_idx_per_token[start_idx:end_idx]
        swa_local_indices = metadata.swa_local_indices_cuda[start_idx:end_idx]
        local_layer_idx = kv_cache_manager.layer_offsets[layer_idx]
        block_table_swa = metadata.sliding_block_tables[
            local_layer_idx, DeepseekV4AttentionType.SWA.value
        ]

        if self.compress_ratio > 1:
            compressed_buffer_ptr = metadata.compressed_buffer_ptrs[layer_idx]
            compress_pool_base_ptr = metadata.sparse_mla_base_ptrs[self.compress_ratio]
            block_table_compressed = metadata.compress_block_tables[self.compress_ratio]
            if self.compress_ratio == 4:
                sparse_backend_args = forward_args.sparse_backend_args
                assert sparse_backend_args is not None, (
                    "sparse_backend_args is required when compress_ratio=4"
                )
                topk_indices = sparse_backend_args.topk_indices
                assert topk_indices is not None, "topk_indices is required when compress_ratio=4"
                compressed_local_indices = topk_indices
            else:
                compressed_local_indices = metadata.compressed_local_indices_cuda[start_idx:end_idx]
        else:
            compressed_buffer_ptr = 0
            compress_pool_base_ptr = 0
            block_table_compressed = None
            compressed_local_indices = None

        global_indices = deepseek_v4_local_to_global_indices(
            req_id=req_id,
            block_table_swa=block_table_swa,
            swa_local_indices=swa_local_indices,
            swa_pool_base_ptr=swa_pool_base_ptr,
            swa_buffer_ptr=swa_buffer_ptr,
            tokens_per_block=kv_cache_manager.tokens_per_block,
            token_stride=token_stride,
            block_table_compressed=block_table_compressed,
            compressed_local_indices=compressed_local_indices,
            compress_pool_base_ptr=compress_pool_base_ptr,
            compressed_buffer_ptr=compressed_buffer_ptr,
            compress_ratio=self.compress_ratio,
            num_compressed_indices=metadata.max_compressed_indices[self.compress_ratio],
        )

        return global_indices, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None
