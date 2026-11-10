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
"""The masked multi-head attention library of the native attention op."""

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
from tensorrt_llm.bindings.internal import thop

from . import fallback
from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class MmhaFmha(PhasedFmha):
    """Masked multi-head attention: generation for what XQA and trtllm-gen decline."""

    SERVES_CONTEXT = False
    SERVES_GENERATION = True
    SERVES_MHA = True
    SERVES_MLA = False
    REQUIRES_PAGED_KV = False
    # The native op consumes pool pointers directly; sparse managers may have
    # no Role.KEY buffer from which to query a dense block extent.
    NEEDS_BLOCK_EXTENT = False
    supports_skip_correction = True
    supports_workspace_reclamation = True

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
            is_supported=thop.mmha_is_supported,
        )

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
        thop.mmha_run_generation(params)


__all__ = ["MmhaFmha"]
