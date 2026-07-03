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

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, cast

import torch

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
    PredefinedAttentionMask,
)
from tensorrt_llm.bindings.internal import thop

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


@lru_cache(maxsize=128)
def get_trtllm_gen_context_workspace_size(
    dtype: torch.dtype,
    max_num_seq: int,
    max_num_tokens: int,
    num_heads: int,
    head_size: int,
    rotary_embedding_dim: int,
    fp8_context_fmha: bool,
) -> int:
    """Return the fused context-preprocessing workspace size in bytes."""
    if max_num_tokens == 0:
        return 0
    layout = thop.get_trtllm_gen_context_workspace_layout(
        dtype,
        max_num_seq,
        max_num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        True,
        fp8_context_fmha,
    )
    return int(layout["total_size"])


@dataclass(slots=True)
class FmhaParams:
    attn: "TrtllmAttention"
    meta: "TrtllmAttentionMetadata"
    fwd: AttentionForwardArgs
    workspace: torch.Tensor
    attention_input: Optional[torch.Tensor] = None
    qkv_input: Optional[torch.Tensor] = None
    key_input: Optional[torch.Tensor] = None
    value_input: Optional[torch.Tensor] = None
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
        self._followup_fmhas: tuple[PhasedFmha, ...] = ()
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

        get_page_index_upper_bound = getattr(
            getattr(kv_cache_manager, "impl", None),
            "get_page_index_upper_bound",
            None,
        )
        if get_page_index_upper_bound is not None:
            # KVCacheManagerV2 exposes an already-flattened page-index bound,
            # unlike the legacy logical block count.
            return int(kv_cache_manager.blocks_in_primary_pool)

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

    def set_followup_fmhas(self, followup_fmhas: tuple["PhasedFmha", ...]) -> None:
        """Set later phased libraries that may provide an unsupported phase."""
        self._followup_fmhas = followup_fmhas

    @staticmethod
    def _phase_forward_args(
        forward_args: AttentionForwardArgs,
        input_type: AttentionInputType,
        is_cross: bool,
    ) -> AttentionForwardArgs:
        updates: dict[str, object] = {"attention_input_type": input_type}
        if (
            input_type == AttentionInputType.generation_only
            and not is_cross
            and forward_args.attention_mask == CustomAttentionMask.CUSTOM
        ):
            # Custom multimodal masks describe context tokens only. Generation
            # uses the regular causal decode semantics.
            updates.update(
                attention_mask=PredefinedAttentionMask.CAUSAL,
                attention_mask_data=None,
            )
        return replace(forward_args, **updates)

    def is_context_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        return False

    def is_generation_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        return False

    def is_mla_context_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        return False

    def is_mla_generation_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        return False

    def _select_phase_fmha(
        self,
        phase: str,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> Optional["PhasedFmha"]:
        is_mla = self.attn.is_mla_enable
        support_method_name = f"is_{'mla_' if is_mla else ''}{phase}_supported"
        for fmha in (self, *self._followup_fmhas):
            support_method = getattr(fmha, support_method_name)
            if support_method(q, k, v, metadata, forward_args):
                return fmha
        return None

    def _select_phase_fmhas(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> tuple[
        Optional["PhasedFmha"],
        Optional["PhasedFmha"],
        AttentionForwardArgs,
        AttentionForwardArgs,
    ]:
        context_args = self._phase_forward_args(
            forward_args,
            AttentionInputType.context_only,
            metadata.is_cross,
        )
        generation_args = self._phase_forward_args(
            forward_args,
            AttentionInputType.generation_only,
            metadata.is_cross,
        )
        has_context = (
            metadata.num_contexts > 0
            and forward_args.attention_input_type != AttentionInputType.generation_only
        )
        has_generation = (
            metadata.num_generations > 0
            and forward_args.attention_input_type != AttentionInputType.context_only
        )
        context_fmha = (
            self._select_phase_fmha("context", q, k, v, metadata, context_args)
            if has_context
            else None
        )
        generation_fmha = (
            self._select_phase_fmha("generation", q, k, v, metadata, generation_args)
            if has_generation
            else None
        )
        return context_fmha, generation_fmha, context_args, generation_args

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        context_fmha, generation_fmha, _, _ = self._select_phase_fmhas(
            q,
            k,
            v,
            metadata,
            forward_args,
        )
        return self._selected_phases_support_request(
            context_fmha,
            generation_fmha,
            metadata,
            forward_args,
        )

    @staticmethod
    def _selected_phases_support_request(
        context_fmha: Optional["PhasedFmha"],
        generation_fmha: Optional["PhasedFmha"],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        has_context = (
            metadata.num_contexts > 0
            and forward_args.attention_input_type != AttentionInputType.generation_only
        )
        has_generation = (
            metadata.num_generations > 0
            and forward_args.attention_input_type != AttentionInputType.context_only
        )
        return (
            (has_context or has_generation)
            and (not has_context or context_fmha is not None)
            and (not has_generation or generation_fmha is not None)
        )

    def try_forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        """Run a supported phased request after selecting each provider once."""
        phase_selection = self._select_phase_fmhas(q, k, v, metadata, forward_args)
        if not self._selected_phases_support_request(
            phase_selection[0],
            phase_selection[1],
            metadata,
            forward_args,
        ):
            return False
        self._forward_with_selected_phases(
            q,
            k,
            v,
            metadata,
            forward_args,
            *phase_selection,
        )
        return True

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
        phase_selection = self._select_phase_fmhas(q, k, v, metadata, forward_args)
        if not self._selected_phases_support_request(
            phase_selection[0],
            phase_selection[1],
            metadata,
            forward_args,
        ):
            raise RuntimeError(f"{type(self).__name__} does not support this phased request.")
        self._forward_with_selected_phases(
            q,
            k,
            v,
            metadata,
            forward_args,
            *phase_selection,
        )

    def _forward_with_selected_phases(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        context_fmha: Optional["PhasedFmha"],
        generation_fmha: Optional["PhasedFmha"],
        context_args: AttentionForwardArgs,
        generation_args: AttentionForwardArgs,
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

        fp8_context_fmha = (
            context_fmha.get_fp8_context_fmha(q, output, metadata, context_args, is_gen_only)
            if context_fmha is not None
            else False
        )
        prepared_fmhas: set[PhasedFmha] = set()
        for fmha, phase_args in (
            (context_fmha, context_args),
            (generation_fmha, generation_args),
        ):
            if fmha is not None and fmha not in prepared_fmhas:
                fmha.prepare_workspace(q, k, v, metadata, phase_args, workspace)
                prepared_fmhas.add(fmha)

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
            params.key_input = (
                k[token_offset : token_offset + num_ctx_tokens] if k is not None else None
            )
            params.value_input = (
                v[token_offset : token_offset + num_ctx_tokens] if v is not None else None
            )
            params.context_buf = out_tensor[token_offset : token_offset + num_ctx_tokens]
            params.sequence_lengths = sequence_length[seq_offset:]
            params.context_lengths = context_lengths[seq_offset:]
            params.max_past_kv_length = max_past_kv_len
            params.num_tokens = num_ctx_tokens
            params.seq_offset = seq_offset
            params.input_seq_length = max_context_q_len
            params.batch_size = num_seqs
            if attn.is_mla_enable:
                if context_fmha is None:
                    raise RuntimeError("No phased FMHA library supports MLA context attention.")
                context_fmha.run_mla_context(replace(params, fwd=context_args))
            else:
                if context_fmha is None:
                    raise RuntimeError("No phased FMHA library supports context attention.")
                context_fmha.run_context(replace(params, fwd=context_args))

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
            params.key_input = (
                k[token_offset : token_offset + num_gen_tokens] if k is not None else None
            )
            params.value_input = (
                v[token_offset : token_offset + num_gen_tokens] if v is not None else None
            )
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
                if generation_fmha is None:
                    raise RuntimeError("No phased FMHA library supports MLA generation attention.")
                generation_fmha.run_mla_generation(replace(params, fwd=generation_args))
            else:
                if generation_fmha is None:
                    raise RuntimeError("No phased FMHA library supports generation attention.")
                generation_fmha.run_generation(replace(params, fwd=generation_args))

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
