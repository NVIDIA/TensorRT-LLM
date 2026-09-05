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
    AttentionForwardArgs, AttentionInputType)

from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class CombinedFmha(PhasedFmha):
    """Combine context and generation implementations for mixed batches."""

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self._context_impl: Optional[PhasedFmha] = None
        self._generation_impl: Optional[PhasedFmha] = None

    def set_fmha_impls(
        self,
        context_impl: PhasedFmha,
        generation_impl: PhasedFmha,
    ) -> None:
        self._context_impl = context_impl
        self._generation_impl = generation_impl

    def _get_context_impl(self) -> PhasedFmha:
        if self._context_impl is None:
            raise RuntimeError("CombinedFmha context implementation is not configured.")
        return self._context_impl

    def _get_generation_impl(self) -> PhasedFmha:
        if self._generation_impl is None:
            raise RuntimeError("CombinedFmha generation implementation is not configured.")
        return self._generation_impl

    @staticmethod
    def _resolve_fp8_context_fmha(
        impl: PhasedFmha,
        params: FmhaParams,
        metadata: "TrtllmAttentionMetadata",
    ) -> None:
        """Re-answer the phase's FP8 question through the impl that will run it.

        ``PhasedFmha.forward`` resolves ``fp8_context_fmha`` once, against ``self``
        -- which here is this router rather than either delegate, so it answers with
        the base implementation's blanket ``False``. The flag is not advisory: the
        trtllm-gen generation path keys the kernel it selects off it (it decides
        whether Q is handed over as FP8), so a delegate that receives a value some
        other implementation computed silently runs a different kernel than it does
        when it owns the whole forward.
        """
        fwd = params.fwd
        params.fp8_context_fmha = impl.get_fp8_context_fmha(
            params.qkv_or_q,
            params.output,
            metadata,
            fwd,
            fwd.attention_input_type == AttentionInputType.generation_only,
        )

    def prepare_workspace(
        self,
        params: FmhaParams,
        metadata: "TrtllmAttentionMetadata",
    ) -> None:
        # Both phases carve from the same workspace, so each impl must get a chance
        # to grow it before either runs -- each under its own FP8 answer, since that
        # is an input to how much scratch the phase needs.
        for impl in (self._get_context_impl(), self._get_generation_impl()):
            self._resolve_fp8_context_fmha(impl, params, metadata)
            impl.prepare_workspace(params, metadata)

    def run_context(self, params: FmhaParams) -> None:
        impl = self._get_context_impl()
        self._resolve_fp8_context_fmha(impl, params, params.meta)
        impl.run_context(params)

    def run_generation(self, params: FmhaParams) -> None:
        impl = self._get_generation_impl()
        self._resolve_fp8_context_fmha(impl, params, params.meta)
        impl.run_generation(params)
