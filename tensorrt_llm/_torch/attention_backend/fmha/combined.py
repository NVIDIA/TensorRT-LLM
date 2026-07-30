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

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs

from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
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
        self._get_context_impl().run_context(params)

    def run_generation(self, params: FmhaParams) -> None:
        self._get_generation_impl().run_generation(params)
