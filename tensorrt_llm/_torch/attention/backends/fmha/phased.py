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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, cast

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_sm_version

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


@dataclass(slots=True)
class FmhaParams:
    """Phase inputs with exactly one of packed QKV or a separate query tensor.

    MLA uses ``query_input`` with ``is_fused_qkv=False``.
    """

    attn: "TrtllmAttention"
    meta: "TrtllmAttentionMetadata"
    fwd: AttentionForwardArgs
    workspace: torch.Tensor
    qkv_input: Optional[torch.Tensor] = None
    query_input: Optional[torch.Tensor] = None
    key_input: Optional[torch.Tensor] = None
    value_input: Optional[torch.Tensor] = None
    output: Optional[torch.Tensor] = None
    sequence_lengths: Optional[torch.Tensor] = None
    context_lengths: Optional[torch.Tensor] = None
    input_seq_length: int = 0
    max_past_kv_length: int = 0
    max_attention_window_size: int = 0
    cyclic_attention_window_size: int = 0
    num_tokens: int = 0
    seq_offset: int = 0
    # First query token of this phase on the axis of the q handed to the
    # library, which covers the whole batch only where both phases share one
    # tensor. The phase tensors above are already sliced by it; a library that
    # indexes a separate per-token input, such as a sparse block table, needs
    # it to take the matching slice.
    token_offset: int = 0
    tokens_per_block: int = 64
    kv_factor: int = 0
    total_num_blocks: int = 0
    # Number of sequence rows in the active phase.
    batch_size: int = 0
    # Number of logical requests in the active phase.
    num_requests: int = 0
    use_spec_decoding: bool = False
    spec_decoding_generation_lengths: Optional[torch.Tensor] = None
    spec_decoding_position_offsets: Optional[torch.Tensor] = None
    spec_decoding_packed_mask: Optional[torch.Tensor] = None
    spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor] = None
    spec_decoding_bl_tree_mask: Optional[torch.Tensor] = None
    spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor] = None
    is_cross: bool = False


class PhasedFmha(Fmha):
    """FMHA helper for paged-KV libraries that split work by request phase."""

    REQUIRES_PAGED_KV = True
    # Required by backends that construct tensor views over the KV pool.
    NEEDS_BLOCK_EXTENT = True

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self.kv_factor = 1 if attn.is_mla_enable else 2
        self.generation_out_head_size = attn.out_head_size(is_gen_only=True)
        self.context_out_head_size = attn.out_head_size(is_gen_only=False)
        self._v1_total_num_blocks_cache: Optional[tuple[object, int, int]] = None

    def _get_total_num_blocks(
        self,
        meta: "TrtllmAttentionMetadata",
    ) -> int:
        kv_cache_manager = meta.kv_cache_manager
        if kv_cache_manager is None:
            return 0

        local_layer_idx = self.attn.local_layer_idx
        if local_layer_idx is None:
            local_layer_idx = self.attn.get_local_layer_idx(meta)

        # Views can address a different page size and pool root than their
        # owning manager, so its physical pool bound cannot be reused.
        view_bound = getattr(kv_cache_manager, "get_attention_op_num_blocks", None)
        if view_bound is not None:
            return int(view_bound(local_layer_idx))

        if isinstance(kv_cache_manager, KVCacheManagerV2):
            return int(kv_cache_manager.impl.get_page_index_upper_bound(local_layer_idx, Role.KEY))
        if not isinstance(kv_cache_manager, KVCacheManager):
            raise TypeError(f"Unsupported KV cache manager: {type(kv_cache_manager).__name__}.")

        # KVCacheManager V1 compatibility path.
        cached_v1_extent = self._v1_total_num_blocks_cache
        if (
            cached_v1_extent is not None
            and cached_v1_extent[0] is kv_cache_manager
            and cached_v1_extent[1] == local_layer_idx
        ):
            return cached_v1_extent[2]

        pool_mapping = meta.host_kv_cache_pool_mapping
        if pool_mapping is None or pool_mapping.ndim != 2 or pool_mapping.shape[1] < 2:
            raise RuntimeError("KV-cache pool mapping must have shape [num_layers, >=2].")
        if not 0 <= local_layer_idx < pool_mapping.shape[0]:
            raise RuntimeError(
                f"Local layer index {local_layer_idx} is outside the KV-cache pool mapping."
            )

        pool_index = int(pool_mapping[local_layer_idx, 0])
        layer_index_in_pool = int(pool_mapping[local_layer_idx, 1])
        layers_in_pool = int((pool_mapping[:, 0] == pool_index).sum())
        if not 0 <= layer_index_in_pool < layers_in_pool:
            raise RuntimeError(
                f"Layer index {layer_index_in_pool} is outside KV-cache pool {pool_index} "
                f"with {layers_in_pool} layers."
            )

        blocks_in_pool = int(kv_cache_manager.impl.get_primary_pool_data(local_layer_idx).shape[0])
        total_num_blocks = (blocks_in_pool * layers_in_pool - layer_index_in_pool) * self.kv_factor
        self._v1_total_num_blocks_cache = (
            kv_cache_manager,
            local_layer_idx,
            total_num_blocks,
        )
        return total_num_blocks

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        pass

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        attn = self.attn
        output = forward_args.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires output.")
        if self.REQUIRES_PAGED_KV and metadata.kv_cache_block_offsets is None:
            raise RuntimeError(f"{type(self).__name__} requires paged KV cache.")

        workspace = cast(torch.Tensor, metadata.effective_workspace)

        num_tokens = q.size(0)
        attention_input_type = forward_args.attention_input_type
        is_gen_only = attention_input_type == AttentionInputType.generation_only

        num_contexts = metadata.num_contexts
        num_ctx_tokens = metadata.num_ctx_tokens
        num_generations = metadata.num_generations
        num_gen_tokens = num_tokens if is_gen_only else num_tokens - num_ctx_tokens
        if num_gen_tokens < 0:
            raise RuntimeError(
                f"Invalid FMHA token counts: num_tokens={num_tokens}, "
                f"num_ctx_tokens={num_ctx_tokens}, attention_input_type={attention_input_type}."
            )

        self.prepare_workspace(
            q,
            k,
            v,
            metadata,
            forward_args,
            workspace,
        )

        out_head_size = self.generation_out_head_size if is_gen_only else self.context_out_head_size
        if output.dtype == torch.uint8:
            # NVFP4 stores two output values in each byte.
            out_head_size //= 2
        out_tensor = output.view(num_tokens, attn.num_heads, out_head_size)

        attention_window_size = forward_args.attention_window_size
        cache_indirection = metadata.cache_indirection
        max_attention_window_size = (
            attention_window_size
            if metadata.beam_width == 1
            else (
                cache_indirection.size(2)
                if cache_indirection is not None
                else attention_window_size
            )
        )
        tokens_per_block = (
            metadata.tokens_per_block if metadata.tokens_per_block is not None else 64
        )

        is_fused_qkv = forward_args.is_fused_qkv
        params = FmhaParams(
            attn=attn,
            meta=metadata,
            fwd=forward_args,
            workspace=workspace,
            max_attention_window_size=max_attention_window_size,
            cyclic_attention_window_size=attention_window_size,
            tokens_per_block=tokens_per_block,
            kv_factor=self.kv_factor,
            total_num_blocks=(
                self._get_total_num_blocks(metadata) if self.NEEDS_BLOCK_EXTENT else 0
            ),
            is_cross=metadata.is_cross,
        )

        sequence_length = metadata.kv_lens_cuda_runtime
        host_past_key_value_lengths = metadata.kv_lens_runtime

        if num_contexts > 0 and attention_input_type != AttentionInputType.generation_only:
            seq_offset = 0
            token_offset = 0
            num_seqs = num_contexts

            context_lengths = metadata.prompt_lens_cuda_runtime
            host_context_lengths = metadata.prompt_lens_cpu_runtime
            max_context_q_len = int(host_context_lengths[seq_offset : seq_offset + num_seqs].max())
            max_past_kv_len = int(
                host_past_key_value_lengths[seq_offset : seq_offset + num_seqs].max()
            )

            phase_input = q[token_offset : token_offset + num_ctx_tokens]
            params.qkv_input = phase_input if is_fused_qkv else None
            params.query_input = None if is_fused_qkv else phase_input
            params.key_input = (
                k[token_offset : token_offset + num_ctx_tokens] if k is not None else None
            )
            params.value_input = (
                v[token_offset : token_offset + num_ctx_tokens] if v is not None else None
            )
            params.output = out_tensor[token_offset : token_offset + num_ctx_tokens]
            params.sequence_lengths = sequence_length[seq_offset:]
            params.context_lengths = context_lengths[seq_offset:]
            params.max_past_kv_length = max_past_kv_len
            params.num_tokens = num_ctx_tokens
            params.seq_offset = seq_offset
            params.token_offset = token_offset
            params.input_seq_length = max_context_q_len
            params.batch_size = num_seqs
            params.num_requests = num_seqs
            if attn.is_mla_enable:
                self.run_mla_context(params)
            else:
                self.run_context(params)

        if num_generations > 0 and attention_input_type != AttentionInputType.context_only:
            seq_offset = num_contexts
            token_offset = 0 if is_gen_only else num_ctx_tokens
            num_seqs = num_generations

            max_past_kv_len = int(
                host_past_key_value_lengths[seq_offset : seq_offset + num_seqs].max()
            )
            input_seq_length = num_gen_tokens // num_seqs if num_seqs > 0 else 1

            params.use_spec_decoding = (
                metadata.is_spec_decoding_enabled and metadata.use_spec_decoding
            )
            if params.use_spec_decoding:
                params.spec_decoding_generation_lengths = metadata.spec_decoding_generation_lengths
                offsets = metadata.spec_decoding_position_offsets
                if offsets is not None and offsets.dim() == 1:
                    if not metadata.is_sm_version_trtllm_gen_kernel(sm=get_sm_version()):
                        # Hopper masks and offsets use the current compact query width,
                        # which can be smaller than the persistent buffer's capacity.
                        query_len = metadata.spec_decoding_query_len
                        if query_len <= 0:
                            raise ValueError(
                                "1-D speculative position offsets require a positive query length."
                            )
                        offsets = offsets[: metadata.max_num_requests * query_len].view(
                            metadata.max_num_requests, query_len
                        )
                    else:
                        offsets = offsets.view(metadata.max_num_requests, -1)
                params.spec_decoding_position_offsets = offsets
                params.spec_decoding_packed_mask = metadata.spec_decoding_packed_mask
                params.spec_decoding_bl_tree_mask_offset = (
                    metadata.spec_decoding_bl_tree_mask_offset
                )
                params.spec_decoding_bl_tree_mask = metadata.spec_decoding_bl_tree_mask
                params.spec_bl_tree_first_sparse_mask_offset_kv = (
                    metadata.spec_bl_tree_first_sparse_mask_offset_kv
                )

            phase_input = q[token_offset : token_offset + num_gen_tokens]
            params.qkv_input = phase_input if is_fused_qkv else None
            params.query_input = None if is_fused_qkv else phase_input
            params.key_input = (
                k[token_offset : token_offset + num_gen_tokens] if k is not None else None
            )
            params.value_input = (
                v[token_offset : token_offset + num_gen_tokens] if v is not None else None
            )
            params.output = out_tensor[token_offset : token_offset + num_gen_tokens]
            params.sequence_lengths = sequence_length[seq_offset:]
            params.context_lengths = metadata.prompt_lens_cuda_runtime[
                seq_offset : seq_offset + num_seqs
            ]
            params.max_past_kv_length = max_past_kv_len
            params.num_tokens = num_gen_tokens
            params.seq_offset = seq_offset
            params.token_offset = token_offset
            params.input_seq_length = input_seq_length
            params.batch_size = num_seqs
            params.num_requests = num_seqs // metadata.beam_width
            if attn.is_mla_enable:
                self.run_mla_generation(params)
            else:
                # The custom mask covers only the context portion of a mixed batch.
                if (
                    not metadata.is_cross
                    and params.fwd.attention_mask == CustomAttentionMask.CUSTOM
                ):
                    params.fwd.attention_mask = PredefinedAttentionMask.CAUSAL
                    params.fwd.attention_mask_data = None
                self.run_generation(params)

    def run_context(self, params: FmhaParams) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support context attention.")

    def run_generation(self, params: FmhaParams) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support generation attention.")

    def run_mla_context(self, params: FmhaParams) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support MLA context attention.")

    def run_mla_generation(self, params: FmhaParams) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support MLA generation attention."
        )
