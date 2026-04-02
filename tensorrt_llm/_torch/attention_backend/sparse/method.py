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

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

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
    """

    def predict_sparse_indices(
        self,
        mla: MLA,
        attn_metadata: AttentionMetadata,
        hidden_states: torch.Tensor,
        qr: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Predict sparse attention indices for the current batch.

        Called from MLA.forward_impl() to determine which KV blocks each
        token should attend to.  Delegates to the trtllm attention backend's
        predict_sparse_indices() so all trtllm-based sparse methods share
        the same prediction interface.

        Args:
            mla: The MLA module instance.
            attn_metadata: Attention metadata for the current batch.
            hidden_states: Pre-attention hidden states.
            qr: Compressed query before q_b_proj (needed by DSA indexer).
            position_ids: Token position IDs.

        Returns:
            Optional topk index tensor ``[num_tokens, topk]``, or None
            when sparse routing is not applicable.
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
        topk_indices: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> None:
        """Dispatch context-phase attention for this sparse method.

        Called from MLA.forward_impl() when topk_indices is not None.
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
        topk_indices: Optional[torch.Tensor],
    ) -> None:
        """Dispatch generation-phase attention for this sparse method.

        Called from MLA.forward_impl() when topk_indices is not None.
        """
        ...
