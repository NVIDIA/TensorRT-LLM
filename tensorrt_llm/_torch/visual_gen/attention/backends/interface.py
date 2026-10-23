# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Visual Generation Attention Backend Interface

Defines shared types, enums, and the abstract base class for attention backends.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch


@torch.compiler.disable
def is_fused_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Whether Q, K, and V are gap-free slices from one fused-QKV buffer.

    At TRTLLM/VisualGen, Attention.get_qkv() at QKVMode.FUSE_QKV produces this type of buffer,
    where one qkv_proj output is split along the last dim.
    """
    if q.dim() != 4 or k.shape != v.shape or q.shape[3] != k.shape[3]:
        return False
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return False
    storage_ptr = q.untyped_storage().data_ptr()
    if any(x.untyped_storage().data_ptr() != storage_ptr for x in (k, v)):
        return False

    b, s, h_q, d = q.shape
    h_kv = k.shape[2]
    q_dim, kv_dim = h_q * d, h_kv * d
    total = q_dim + 2 * kv_dim

    # Heads packed within each slice, and one shared row pitch across all three.
    for x, h in ((q, h_q), (k, h_kv), (v, h_kv)):
        if x.shape[:3] != (b, s, h):
            return False
        if x.stride() != (s * total, total, d, 1):
            return False

    base = q.storage_offset()
    return k.storage_offset() == base + q_dim and v.storage_offset() == base + q_dim + kv_dim


@torch.compiler.disable
def as_fused_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Zero-copy [B, S, q_dim + 2 * kv_dim] view of the parent QKV buffer."""
    b, s, h_q, d = q.shape
    total = (h_q + 2 * k.shape[2]) * d
    return torch.as_strided(q, (b, s, total), (s * total, total, 1), q.storage_offset())


class AttentionTensorLayout(str, Enum):
    """
    Tensor layout for attention backend input/output.

    Backends declare their preferred layout so the attention module
    can reshape tensors optimally before calling the backend.
    """

    NHD = "NHD"  # [B, S, H, D] - batch, seq, heads, dim
    HND = "HND"  # [B, H, S, D] - batch, heads, seq, dim


class AttentionBackend(ABC):
    """Contract for all visual-gen attention backends.

    Every backend must implement ``forward`` and declare a ``preferred_layout``.
    Backends pick the kwargs they need from the caller and ignore the rest
    via ``**kwargs``.
    """

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Attention over ``q``, ``k``, ``v`` in the backend's ``preferred_layout``.

        Keyword contract shared by every backend (unknown keywords are ignored):

        * ``batch_size``, ``seq_len``, ``seq_len_kv``: the batch and the query and
          key row counts, padding included, as the Attention module derives them from
          the tensor shapes.
        * ``attention_mask``, ``key_padding_mask``: the mask, if the backend takes one.
        * ``kv``: a ``ForwardPlan`` from a ``RolloutKVCacheInterface``; only backends
          whose ``support_kv_cache()`` is true accept it. ``k``/``v`` are then this
          call's new tokens, staged into the cache and attended together with what the
          cache holds before them, full within a causal block and causal across causal
          blocks. ``kv.num_tokens`` counts the real tokens of all ranks; rows past it
          are padding added so the sequence splits evenly across ranks, neither
          written to the cache nor attended, and they come back zero.

        Under CUDA graphs a captured forward over a cache depends on ``kv.graph_key``
        besides the tensor shapes: it holds the host values baked into the capture, so
        a graph runner must key on it. The cache's own state (table, lengths, slot ids)
        lives in device tensors rewritten in place by ``commit``, so commits need no
        recapture and are not part of the key.
        """

    @property
    @abstractmethod
    def preferred_layout(self) -> AttentionTensorLayout: ...

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support LSE output. "
            "Override forward_with_lse() or check support_lse() before calling."
        )

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @classmethod
    def support_lse(cls) -> bool:
        """Whether the backend supports returning the softmax log-sum-exp (LSE) of the attention weights."""
        return False

    def support_kv_cache(self) -> bool:
        """Whether ``forward`` accepts a ``ForwardPlan`` over a ``RolloutKVCacheInterface``
        as ``kv``. A backend without support would silently drop the keyword through
        ``**kwargs``, so callers must check before passing one."""
        return False

    def support_kv_cache_page_size(
        self, tokens_per_page: int, dtype: Optional[torch.dtype] = None
    ) -> bool:
        """Whether ``forward`` can attend over a ``BlockCausalKVCacheManager`` with
        ``tokens_per_page``-token pages holding ``dtype`` K/V (default: the backend's
        dtype) on the current GPU. A caller sizing a cache asks this before building it;
        ``forward`` refuses an unsupported cache either way."""
        return self.support_kv_cache()
