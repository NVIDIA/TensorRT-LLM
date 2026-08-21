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
"""Inkling's backend selection key and its per-forward backend inputs.

Inkling is not sparse attention (see the package docstring); it reuses the
``sparse/`` seams because ``sparse/registry.py`` is where model-specific backends
and cache managers are selected, and ``sparse_backend_args`` is the registered
slot for a model-specific forward input. Consumers holding no ``ModelConfig``
reach the backend only through the registry, so the config must be populated for
them to resolve anything but the family default.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from tensorrt_llm.llmapi.llm_args import BaseSparseAttentionConfig

from ..params import SparseBackendForwardArgs, SparseParams


class InklingSparseParams(SparseParams):
    """Selection key for Inkling's backend and cache manager in
    ``sparse/registry.py``; the kernels' per-layer scalars live on the backend.
    """

    algorithm = "inkling"


class InklingSparseAttentionConfig(BaseSparseAttentionConfig):
    """Architecture-derived config that carries :class:`InklingSparseParams`.

    Subclasses the interface but is deliberately not a member of the user-facing
    ``SparseAttentionConfig`` union: there is no knob to expose, and the field is
    injected by ``ModelConfig.from_pretrained`` keyed on architecture.
    """

    algorithm: Literal["inkling"] = "inkling"

    def to_sparse_params(self, pretrained_config=None, layer_idx=None) -> InklingSparseParams:
        return InklingSparseParams()


@dataclass(kw_only=True, slots=True)
class InklingBackendForwardArgs(SparseBackendForwardArgs):
    """Per-forward inputs ``InklingAttention`` hands to its backend, riding
    ``AttentionForwardArgs.sparse_backend_args``.

    ``rel_logits`` is a ``[num_query_tokens, local_heads, rel_extent]`` fp32
    additive bias projected from the hidden states -- content-dependent, so the
    shared ``relative_attention_bias`` (T5's position-indexed table) cannot carry
    it. ``allow_mixed`` certifies the short-conv state pool is active, which is
    what makes a mixed context+generation batch legal.
    """

    rel_logits: torch.Tensor
    allow_mixed: bool = False


def inkling_forward_args(
    rel_logits: torch.Tensor,
    *,
    allow_mixed: bool,
    output: Optional[torch.Tensor] = None,
):
    """Build the ``AttentionForwardArgs`` carrying Inkling's backend inputs."""
    from ...interface import AttentionForwardArgs

    return AttentionForwardArgs(
        output=output,
        sparse_backend_args=InklingBackendForwardArgs(
            rel_logits=rel_logits, allow_mixed=allow_mixed
        ),
    )
