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

"""Abstract base for indexer-style FMHA backends.

Sparse-attention predictors (the MiniMax-M3 indexer, future Top-k
selectors) need a fast 'score every KV block against an MQA query'
pass over a paged KV cache. The score tensor is then fed into a top-k
selector to produce the sparse block indices consumed by the main
attention.

That proxy attention is structurally a regular FMHA call -- it takes
``Q/K/V`` over a paged KV cache, runs causal attention, etc. -- but
its output is a ``max_score`` tensor rather than an attention output.
The :class:`IndexerProxyFmha` base class lets multiple proxy
implementations (MSA's ``fmha_sm100``, a future Triton path, etc.)
live in the same :data:`FMHA_LIBS` registry as main-attention FMHA
backends. They opt out of the main-attention dispatch loop by
returning ``False`` from :meth:`is_supported` and instead expose a
custom :meth:`forward_proxy` entry point that callers (sparse
indexers) invoke directly after looking the class up in the registry.

See :class:`tensorrt_llm._torch.attention_backend.fmha.msa_proxy_mqa.MsaProxyMqaFmha`
for the canonical concrete implementation.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata


class IndexerProxyFmha(Fmha):
    """Abstract FMHA backend that produces per-block max scores.

    Indexer-style backends are owned by sparse-attention indexers, not
    by :class:`TrtllmAttention`. They are constructed without an
    ``attn`` argument and never dispatched by the standard
    ``TrtllmAttention.forward`` loop -- :meth:`is_supported` always
    returns ``False``, so the loop skips past them when iterating
    :data:`FMHA_LIBS`.

    Concrete subclasses must implement :meth:`forward_proxy`, which
    has a custom (and stable) signature tailored to the proxy-MQA use
    case. Indexers locate concrete subclasses via the standard
    :func:`get_enabled_fmha_lib_classes` helper, filter by
    ``issubclass(IndexerProxyFmha)`` and ``is_available()``, and call
    ``forward_proxy`` directly. The Fmha registry remains the single
    source of truth for which proxy implementations are reachable on
    this build.

    Future expansion: this base may grow companion methods for other
    indexer compute primitives (e.g. block-score reductions). The
    no-op :meth:`is_supported` keeps the implementation outside the
    main-attention dispatch path regardless of how many additional
    methods are added.
    """

    @abstractmethod
    def forward_proxy(
        self,
        idx_q: torch.Tensor,
        idx_k_paged: torch.Tensor,
        *,
        qo_lens_cpu: torch.Tensor,
        kv_lens_cpu: torch.Tensor,
        qo_offset_cpu: Optional[torch.Tensor],
        kv_indices: torch.Tensor,
        sm_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        """Compute a per-(qo_head, kv_tile) max-score tensor.

        Parameters
        ----------
        idx_q : torch.Tensor
            Shape ``[total_q, num_qo_heads, head_dim]`` (bf16/fp16).
        idx_k_paged : torch.Tensor
            Paged KV in HND layout
            ``[num_pages, num_kv_heads, page_size, head_dim]``. For the
            canonical MQA proxy, ``num_kv_heads == 1`` and ``idx_k`` is
            broadcast across every QO head during scoring.
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
            Softmax scale applied to the QK scores prior to the
            per-block max reduction.
        causal : bool
            Whether to apply a causal mask. Prefill batches typically
            use ``True``; pure-decode batches use ``False``.

        Returns
        -------
        torch.Tensor
            Shape ``[num_qo_heads, max_k_tiles, total_q]``, dtype
            float32. Out-of-range tile slots are padded with ``-inf``
            so a subsequent top-k selector can ignore them.
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
        # Indexer-style backends never participate in the standard
        # TrtllmAttention.forward dispatch loop. Returning False keeps
        # us out of `for fmha in self.fmha_libs: if fmha.is_supported`
        # so we don't have to spuriously claim/refuse main attention
        # work.
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
            f"{type(self).__name__} is an indexer-style proxy FMHA backend; "
            "it produces per-block max scores via forward_proxy() and is "
            "not driven by the standard FMHA dispatch path. Sparse-attention "
            "indexers should locate it via get_enabled_fmha_lib_classes() "
            "filtered to subclasses of IndexerProxyFmha."
        )


__all__ = ["IndexerProxyFmha"]
