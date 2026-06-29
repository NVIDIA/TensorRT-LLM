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

"""Abstract base for block-sparse paged-KV FMHA backends.

Sparse-attention algorithms typically split work into two phases:

  1. A predictor pass that produces a per-(query, KV head) list of
     "selected KV blocks". See
     :class:`tensorrt_llm._torch.attention_backend.fmha.indexer_proxy.IndexerProxyFmha`
     for the FMHA library family that implements that phase.
  2. A *block-sparse* main attention pass that consumes the selected
     block indices and runs the actual attention on a paged KV cache,
     skipping unselected blocks.

:class:`BlockSparseFmha` is the abstract base for the FMHA libraries
that implement phase (2). Like :class:`IndexerProxyFmha`, they live in
the same :data:`FMHA_LIBS` registry as standard main-attention FMHA
backends (FlashInfer trtllm-gen, fallback) so the same
``TLLM_FMHA_LIBS`` env var selects them; they opt out of the standard
:meth:`TrtllmAttention.forward` dispatch loop by returning ``False``
from :meth:`is_supported` because their input contract
(``kv_block_indexes``, sparse-attention metadata) does not fit the
standard :class:`AttentionForwardArgs` signature, and they are invoked
directly by sparse-attention attention backends that have access to
those extra inputs.

See
:class:`tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa.MsaSparseGqaFmha`
for the canonical concrete implementation (MSA's ``fmha_sm100``
sparse GQA kernel).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata


class BlockSparseFmha(Fmha):
    """Abstract FMHA backend that consumes ``kv_block_indexes``.

    Block-sparse backends accept a per-query list of selected KV block
    indices (produced by a sparse predictor; see
    :class:`IndexerProxyFmha`) and run paged GQA attention restricted
    to those blocks. They are invoked directly by sparse-attention
    backends that own the predictor output, not by the standard
    :meth:`TrtllmAttention.forward` dispatch loop. The standard
    dispatch loop is opted out of via :meth:`is_supported` returning
    ``False``.

    Concrete subclasses must implement :meth:`forward_block_sparse`,
    which has a stable, dedicated signature carrying the
    sparse-attention metadata that does not fit the standard
    :class:`AttentionForwardArgs`. Sparse-attention backends locate
    concrete subclasses via :func:`get_enabled_fmha_lib_classes`
    filtered to subclasses of :class:`BlockSparseFmha` and call
    :meth:`forward_block_sparse` directly.
    """

    @abstractmethod
    def forward_block_sparse(
        self,
        q: torch.Tensor,
        k_paged: torch.Tensor,
        v_paged: torch.Tensor,
        kv_block_indexes: torch.Tensor,
        *,
        qo_lens_cpu: torch.Tensor,
        kv_lens_cpu: torch.Tensor,
        qo_offset_cpu: Optional[torch.Tensor],
        kv_indices: torch.Tensor,
        sm_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        """Run block-sparse paged GQA attention.

        Parameters
        ----------
        q : torch.Tensor
            Shape ``[total_q, num_qo_heads, head_dim]`` (bf16/fp16).
        k_paged : torch.Tensor
            Paged K cache in HND layout
            ``[num_pages, num_kv_heads, page_size, head_dim]``.
        v_paged : torch.Tensor
            Paged V cache, same shape as ``k_paged``.
        kv_block_indexes : torch.Tensor
            Shape ``[total_q, num_kv_heads, topk]``, dtype int32,
            ascending per row with ``-1`` padding at the tail. Encodes
            the per-query subset of KV blocks selected by the
            preceding sparse predictor.
        qo_lens_cpu, kv_lens_cpu : torch.Tensor
            Shape ``[batch]``, dtype int32, on CPU. Per-request Q/O
            and KV lengths.
        qo_offset_cpu : torch.Tensor, optional
            Shape ``[batch]``, dtype int32, on CPU. Per-request causal
            offset (i.e. prefix length). Ignored when ``causal=False``.
        kv_indices : torch.Tensor
            Shape ``[sum_pages_across_batch]``, dtype int32, on the
            cache device. Flattened paged-KV page table.
        sm_scale : float
            Softmax scale.
        causal : bool
            Whether to apply a causal mask.

        Returns
        -------
        torch.Tensor
            Shape ``[total_q, num_qo_heads, head_dim]``, dtype
            bfloat16. The attention output over the selected KV blocks.
        """
        ...

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
    ) -> bool:
        # Block-sparse backends consume sparse-attention metadata
        # (kv_block_indexes, paged HND KV, per-batch lens) that does
        # not fit AttentionForwardArgs. They are invoked directly by
        # sparse-attention attention backends; returning False keeps
        # us out of the standard TrtllmAttention.forward dispatch loop.
        return False

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is a block-sparse FMHA backend; it is "
            "invoked via forward_block_sparse() by sparse-attention "
            "backends, not by the standard FMHA dispatch path. Locate it "
            "via get_enabled_fmha_lib_classes() filtered to subclasses of "
            "BlockSparseFmha."
        )


__all__ = ["BlockSparseFmha"]
