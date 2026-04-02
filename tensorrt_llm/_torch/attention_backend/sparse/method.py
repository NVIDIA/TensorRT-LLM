# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Generic sparse attention method protocol for MLA integration.

Sparse attention algorithms (DSA, RocketKV, etc.) implement this protocol
to plug into MLAAttention without the attention module needing algorithm-specific
code. The MLA instance is passed as the first argument so implementations can
access projection layers, absorption methods, and attention backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from ...modules.attention import MLA

from ..interface import AttentionMetadata


@runtime_checkable
class SparseAttentionMethod(Protocol):
    """Protocol for sparse attention methods that plug into MLA.

    Implementations receive the MLA instance (``mla``) to access shared
    building blocks (projections, absorption paths, attention backends)
    without owning them.

    The workflow is:
    1. ``pre_attn_process()`` — token-parallel preprocessing (projections,
       k-cache scatter) on ALL tokens before ctx/gen split.
    2. ``dispatch_context()`` / ``dispatch_generation()`` — phase-specific
       attention dispatch. Sparse index prediction happens inside the
       trtllm backend (via ``sparse_attn_predict``), not in MLA.
    """

    def pre_attn_process(
        self,
        mla: MLA,
        attn_metadata: AttentionMetadata,
        hidden_states: torch.Tensor,
        qr: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Pre-attention processing for sparse methods.

        Called once on ALL tokens before the ctx/gen split.  Performs
        token-parallel compute (e.g. indexer projections, k-cache scatter)
        and returns intermediates that will be sliced per ctx/gen and
        passed through to the backend's ``sparse_attn_predict()``.

        Returns:
            Dict of intermediate tensors (e.g. q_fp8, k_fp8, k_scale,
            weights) keyed by name.
        """
        ...

    def dispatch_context(
        self,
        mla: MLA,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        **kwargs,
    ) -> None:
        """Dispatch context-phase attention for this sparse method.

        Sparse index prediction is handled inside the trtllm backend
        (via ``sparse_attn_predict``), not here.  Intermediates from
        ``pre_attn_process`` are passed through via kwargs.
        """
        ...

    def dispatch_generation(
        self,
        mla: MLA,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor],
        **kwargs,
    ) -> None:
        """Dispatch generation-phase attention for this sparse method.

        Sparse index prediction is handled inside the trtllm backend
        (via ``sparse_attn_predict``), not here.  Intermediates from
        ``pre_attn_process`` are passed through via kwargs.
        """
        ...
