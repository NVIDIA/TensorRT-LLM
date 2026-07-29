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

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


@dataclass(slots=True)
class FmhaParams:
    attn: "TrtllmAttention"
    meta: "TrtllmAttentionMetadata"
    fwd: AttentionForwardArgs
    workspace: torch.Tensor
    attention_input: Optional[torch.Tensor] = None
    qkv_input: Optional[torch.Tensor] = None
    context_buf: Optional[torch.Tensor] = None
    sequence_lengths: Optional[torch.Tensor] = None
    context_lengths: Optional[torch.Tensor] = None
    input_seq_length: int = 0
    max_past_kv_length: int = 0
    max_attention_window_size: int = 0
    cyclic_attention_window_size: int = 0
    num_tokens: int = 0
    seq_offset: int = 0
    tokens_per_block: int = 64
    fp8_context_fmha: bool = False
    kv_factor: int = 0
    total_num_blocks: int = 0
    # Context-only fields
    batch_size: int = 0
    # Generation-only fields
    num_requests: int = 0
    spec_decoding_generation_lengths: Optional[torch.Tensor] = None
    spec_decoding_position_offsets: Optional[torch.Tensor] = None
    is_cross: bool = False


class PhasedFmha(Fmha):
    """FMHA helper for paged-KV libraries that split work by request phase."""

    REQUIRES_PAGED_KV = True

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self.kv_factor = 1 if attn.is_mla_enable else 2
        kv_lora_rank = attn.kv_lora_rank or 0
        self.generation_out_head_size = (
            kv_lora_rank if attn.is_mla_enable and kv_lora_rank else attn.head_dim
        )
        self.context_out_head_size = (
            attn.v_head_dim if attn.is_mla_enable and attn.v_head_dim else attn.head_dim
        )

    def _get_total_num_blocks(
        self,
        meta: "TrtllmAttentionMetadata",
    ) -> int:
        kv_cache_manager = meta.kv_cache_manager
        if kv_cache_manager is None:
            return 0

        blocks_in_primary_pool = getattr(kv_cache_manager, "blocks_in_primary_pool", None)
        if blocks_in_primary_pool is None:
            blocks_per_window = getattr(kv_cache_manager, "blocks_per_window", None)
            if blocks_per_window:
                blocks_in_primary_pool = max(
                    int(primary) for primary, _ in blocks_per_window.values()
                )
        if blocks_in_primary_pool is None:
            return 0
        return int(blocks_in_primary_pool) * kv_cache_manager.num_local_layers * self.kv_factor

    def get_fp8_context_fmha(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        is_gen_only: bool,
    ) -> bool:
        return False

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

        fp8_context_fmha = self.get_fp8_context_fmha(q, output, metadata, forward_args, is_gen_only)
        self.prepare_workspace(
            q,
            k,
            v,
            metadata,
            forward_args,
            workspace,
        )

        out_head_size = self.generation_out_head_size if is_gen_only else self.context_out_head_size
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

        params = FmhaParams(
            attn=attn,
            meta=metadata,
            fwd=forward_args,
            workspace=workspace,
            max_attention_window_size=max_attention_window_size,
            cyclic_attention_window_size=attention_window_size,
            tokens_per_block=tokens_per_block,
            fp8_context_fmha=fp8_context_fmha,
            kv_factor=self.kv_factor,
            total_num_blocks=self._get_total_num_blocks(metadata),
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

            params.attention_input = q[token_offset : token_offset + num_ctx_tokens]
            params.qkv_input = params.attention_input
            params.context_buf = out_tensor[token_offset : token_offset + num_ctx_tokens]
            params.sequence_lengths = sequence_length[seq_offset:]
            params.context_lengths = context_lengths[seq_offset:]
            params.max_past_kv_length = max_past_kv_len
            params.num_tokens = num_ctx_tokens
            params.seq_offset = seq_offset
            params.input_seq_length = max_context_q_len
            params.batch_size = num_seqs
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

            predicted_tokens_per_seq = attn.predicted_tokens_per_seq
            spec_gen_lengths = None
            spec_pos_offsets = None
            if metadata.is_spec_decoding_enabled and predicted_tokens_per_seq > 1:
                spec_gen_lengths = metadata.spec_decoding_generation_lengths
                position_offsets_for_cpp = metadata.spec_decoding_position_offsets_for_cpp
                if position_offsets_for_cpp is not None and position_offsets_for_cpp.dim() == 1:
                    position_offsets_for_cpp = position_offsets_for_cpp.view(
                        metadata.max_num_requests, -1
                    )
                spec_pos_offsets = position_offsets_for_cpp

            params.attention_input = q[token_offset : token_offset + num_gen_tokens]
            params.qkv_input = params.attention_input
            params.context_buf = out_tensor[token_offset : token_offset + num_gen_tokens]
            params.sequence_lengths = sequence_length[seq_offset:]
            params.max_past_kv_length = max_past_kv_len
            params.num_tokens = num_gen_tokens
            params.seq_offset = seq_offset
            params.input_seq_length = input_seq_length
            params.num_requests = num_seqs // metadata.beam_width
            params.spec_decoding_generation_lengths = spec_gen_lengths
            params.spec_decoding_position_offsets = spec_pos_offsets
            if attn.is_mla_enable:
                self.run_mla_generation(params)
            else:
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
