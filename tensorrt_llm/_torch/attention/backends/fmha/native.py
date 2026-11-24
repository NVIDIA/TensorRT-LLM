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
"""The base of the native attention op's FMHA libraries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._utils import is_sm_120f

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.attention_layer import AttentionLayer
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata


class NativeFmha(PhasedFmha):
    """A library of the native attention op.

    It serves the attention variants and phases whose ``run_*`` methods it overrides, and
    ``op_is_supported``, the op's ``thop.<library>_is_supported``, checks each phase call.
    """

    REQUIRES_PAGED_KV = False
    # The native op consumes pool pointers directly; sparse managers may have
    # no Role.KEY buffer from which to query a dense block extent.
    NEEDS_BLOCK_EXTENT = False
    supports_skip_correction = True
    supports_workspace_reclamation = True

    op_is_supported: ClassVar[Callable[[FmhaParams, bool], bool]]

    @classmethod
    def _is_available(cls, attn: "AttentionLayer") -> bool:
        is_mla = attn.is_mla_enable
        if not (cls.serves(is_mla, is_gen=False) or cls.serves(is_mla, is_gen=True)):
            return False
        if getattr(attn.sparse_params, "algorithm", None) in ("deepseek_v4", "dsa"):
            if getattr(attn, "kv_cache_dtype", None) == "fp8_ds_mla" or is_sm_120f():
                return False
        return True

    def is_phase_supported(self, params: FmhaParams, is_gen: bool) -> bool:
        """Whether the library serves one phase call; the op checks its input requirements."""
        return type(self).op_is_supported(params, is_gen)

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
        """Builds the parameters of each phase the call checks and asks ``is_phase_supported``."""
        if forward_args.output is None:
            raise RuntimeError(f"{type(self).__name__} requires output.")
        input_type = forward_args.attention_input_type
        is_gen_only = input_type == AttentionInputType.generation_only
        has_context = metadata.num_contexts > 0 and not is_gen_only
        has_generation = (
            metadata.num_generations > 0 and input_type != AttentionInputType.context_only
        )
        check_context = has_context and phase != FmhaPhase.GENERATION
        check_generation = has_generation and phase != FmhaPhase.CONTEXT
        if not (check_context or check_generation):
            return False
        is_mla = self.attn.is_mla_enable
        if (check_context and not self.serves(is_mla, is_gen=False)) or (
            check_generation and not self.serves(is_mla, is_gen=True)
        ):
            return False
        workspace = metadata.effective_workspace
        num_tokens = q.shape[0]
        num_gen_tokens = num_tokens if is_gen_only else num_tokens - metadata.num_ctx_tokens
        phase_num_gen_tokens = num_gen_tokens if has_generation else 0
        if check_context:
            num_contexts = metadata.num_contexts
            params = self.context_params(
                q,
                k,
                v,
                metadata,
                forward_args,
                workspace,
                0,
                max_q_len=int(metadata.prompt_lens_cpu_runtime[:num_contexts].max()),
                max_kv_len=int(metadata.kv_lens_runtime[:num_contexts].max()),
                num_gen_tokens=phase_num_gen_tokens,
            )
            if not self.is_phase_supported(params, False):
                return False
        if check_generation:
            params = self.generation_params(
                q, k, v, metadata, forward_args, workspace, 0, num_gen_tokens
            )
            if not self.is_phase_supported(params, True):
                return False
        return True


__all__ = ["NativeFmha"]
