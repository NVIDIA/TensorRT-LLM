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
"""The XQA FMHA library of the native attention op."""

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm.bindings.internal import thop

from . import fallback
from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class XqaFmha(PhasedFmha):
    """XQA generation kernels."""

    SERVES_CONTEXT = False
    SERVES_GENERATION = True
    SERVES_MHA = True
    SERVES_MLA = True
    REQUIRES_PAGED_KV = False
    # The native op consumes pool pointers directly; sparse managers may have
    # no Role.KEY buffer from which to query a dense block extent.
    NEEDS_BLOCK_EXTENT = False
    supports_skip_correction = True
    supports_workspace_reclamation = True

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        # has_perf_gain() by (generation batch, history in XQA multi-block units, spec decoding,
        # output dtype).
        self._perf_gain: dict[tuple[int, int, bool, torch.dtype], bool] = {}

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        return fallback.is_native_op_available(
            attn, serves_mha=cls.SERVES_MHA, serves_mla=cls.SERVES_MLA
        )

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        def is_supported(params: FmhaParams, is_gen: bool) -> bool:
            # Judge support at the longest KV history the selection cache key admits, so the
            # choice holds for every request sharing the key; has_perf_gain() is keyed separately.
            if is_gen and metadata.max_seq_len is not None:
                if metadata.beam_width == 1:
                    params.max_past_kv_length = metadata.max_seq_len
                elif not self.kv_exceeds_beam_search_window(metadata, forward_args):
                    params.max_past_kv_length = min(
                        metadata.max_seq_len, params.cyclic_attention_window_size - 1
                    )
            return thop.xqa_is_supported(params, is_gen)

        return fallback.is_native_call_supported(
            self,
            q,
            k,
            v,
            metadata,
            forward_args,
            phase=phase,
            serves_context=self.SERVES_CONTEXT,
            serves_generation=self.SERVES_GENERATION,
            is_supported=is_supported,
        ) and self.has_perf_gain(q, k, v, metadata, forward_args)

    def has_perf_gain(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        """Whether XQA is expected to outrun MMHA for this call's generation batch and history.

        The native op weighs the generation batch against the KV history in multi-block units,
        so the answer is memoized on those, on speculative decoding and on the output dtype.
        It does not check support; ``is_supported`` does.
        """
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        max_kv_len = int(
            metadata.kv_lens_runtime[num_contexts : num_contexts + num_generations].max()
        )
        key = (
            num_generations,
            max_kv_len // thop.xqa_min_history_tokens_per_block,
            bool(metadata.is_spec_decoding_enabled and metadata.use_spec_decoding),
            forward_args.output.dtype,
        )
        gain = self._perf_gain.get(key)
        if gain is None:
            is_gen_only = forward_args.attention_input_type == AttentionInputType.generation_only
            num_gen_tokens = q.shape[0] if is_gen_only else q.shape[0] - metadata.num_ctx_tokens
            params = self.generation_params(
                q, k, v, metadata, forward_args, metadata.effective_workspace, 0, num_gen_tokens
            )
            gain = thop.xqa_has_perf_gain(params)
            if gain is None:
                return False
            self._perf_gain[key] = gain
        return gain

    @staticmethod
    def kv_exceeds_beam_search_window(
        metadata: "TrtllmAttentionMetadata", forward_args: AttentionForwardArgs
    ) -> bool:
        """Whether beam search generation holds KV history past the cyclic window, where XQA declines it."""
        if metadata.beam_width <= 1 or metadata.num_generations <= 0:
            return False
        num_contexts = metadata.num_contexts
        max_kv_len = int(
            metadata.kv_lens_runtime[num_contexts : num_contexts + metadata.num_generations].max()
        )
        return max_kv_len + 1 > forward_args.attention_window_size

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        fallback.warn_unsupported_sage_smoothing(forward_args)

    @torch.compiler.disable
    def run_generation(self, params: FmhaParams) -> None:
        thop.xqa_run_generation(params)

    run_mla_generation = run_generation


__all__ = ["XqaFmha"]
