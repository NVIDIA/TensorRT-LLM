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

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)

from .interface import Fmha
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


def generation_rows_uniform(q: torch.Tensor, metadata: "TrtllmAttentionMetadata") -> bool:
    """Whether every generation row of a mixed batch carries the same number of query tokens.

    A phased generation library slices the generation tokens into ``num_generations`` equal
    rows. The executor pads linear speculative drafts to ``max_total_draft_tokens``, but a
    request without drafts (speculation disabled for it, or no overlap scheduler) is packed as a
    single-token row, and such a batch can still divide evenly. Rows hold between 1 and
    ``1 + max_total_draft_tokens`` tokens, so an average of either bound proves uniformity
    without touching a tensor; any other average is checked on the host sequence lengths.
    """
    num_contexts = metadata.num_contexts
    num_generations = metadata.num_generations
    num_gen_tokens = q.shape[0] - metadata.num_ctx_tokens
    if num_generations <= 0 or num_gen_tokens <= 0 or num_gen_tokens % num_generations:
        return False
    q_len = num_gen_tokens // num_generations
    if q_len == 1 or q_len == 1 + (getattr(metadata, "max_total_draft_tokens", None) or 0):
        return True
    seq_lens = metadata.seq_lens
    if seq_lens is None or seq_lens.device.type != "cpu":
        return False
    gen_lens = seq_lens[num_contexts : num_contexts + num_generations]
    return gen_lens.numel() == num_generations and bool((gen_lens == q_len).all())


class CombinedFmha(PhasedFmha):
    """Combine context and generation implementations for mixed batches.

    The context implementation is either a phased library or a whole-batch library that
    honours a context-only call on the mixed tensors (``Fmha.serves_context_only_input``,
    the thop op). With ``whole_batch_context``, the context implementation supports the whole
    mixed batch and serves it alone unless the generation library takes this step's
    generation rows: they must be uniform and
    ``PhasedFmha.serves_mixed_generation_rows`` must accept them.
    """

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self._context_impl: Optional[Fmha] = None
        self._generation_impl: Optional[PhasedFmha] = None
        self._whole_batch_context = False

    def set_fmha_impls(
        self,
        context_impl: Fmha,
        generation_impl: PhasedFmha,
        *,
        whole_batch_context: bool = False,
    ) -> None:
        if not isinstance(context_impl, PhasedFmha) and not context_impl.serves_context_only_input:
            raise TypeError(
                f"{type(context_impl).__name__} can serve the context phase of a mixed batch "
                "only if it is phased or honours a context-only call."
            )
        self._context_impl = context_impl
        self._generation_impl = generation_impl
        self._whole_batch_context = whole_batch_context
        self.supports_workspace_reclamation = (
            context_impl.supports_workspace_reclamation
            and generation_impl.supports_workspace_reclamation
        )

    def _get_context_impl(self) -> Fmha:
        if self._context_impl is None:
            raise RuntimeError("CombinedFmha context implementation is not configured.")
        return self._context_impl

    def _get_generation_impl(self) -> PhasedFmha:
        if self._generation_impl is None:
            raise RuntimeError("CombinedFmha generation implementation is not configured.")
        return self._generation_impl

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        context_impl = self._get_context_impl()
        if (self._whole_batch_context or not isinstance(context_impl, PhasedFmha)) and not (
            forward_args.attention_input_type == AttentionInputType.mixed
            and metadata.num_contexts > 0
            and metadata.num_generations > 0
            and generation_rows_uniform(q, metadata)
            and self._get_generation_impl().serves_mixed_generation_rows(metadata)
        ):
            context_impl.forward(q, k, v, metadata, forward_args)
            return
        if isinstance(context_impl, PhasedFmha):
            self._forward_phases(q, k, v, metadata, forward_args)
            return
        # The whole-batch library serves the context requests on the mixed tensors and
        # leaves the generation rows, which the phased library serves next.
        forward_args.attention_input_type = AttentionInputType.context_only
        try:
            context_impl.forward(q, k, v, metadata, forward_args)
        finally:
            forward_args.attention_input_type = AttentionInputType.mixed
        self._forward_phases(q, k, v, metadata, forward_args, run_context_phase=False)

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        context_impl = self._get_context_impl()
        generation_impl = self._get_generation_impl()
        if isinstance(context_impl, PhasedFmha):
            context_impl.prepare_workspace(
                q,
                k,
                v,
                metadata,
                forward_args,
                workspace,
            )
        generation_impl.prepare_workspace(
            q,
            k,
            v,
            metadata,
            forward_args,
            workspace,
        )

    def run_context(self, params: FmhaParams) -> None:
        context_impl = self._get_context_impl()
        if not isinstance(context_impl, PhasedFmha):
            raise RuntimeError(
                f"{type(context_impl).__name__} serves the context phase through forward()."
            )
        context_impl.run_context(params)

    def run_generation(self, params: FmhaParams) -> None:
        self._get_generation_impl().run_generation(params)
