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

import torch


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
          key sequence lengths. ``seq_len`` is the number of query tokens this call
          computes, as in the engine's attention metadata, not a cached total. The
          Attention module derives it from ``q.shape[1]``; with a ``kv_cache`` the
          caller passes it and it may be smaller than ``q.shape[1]``: rows past it
          are padding added so the sequence splits evenly across ranks, and they
          are neither written to the cache nor attended, and come back zero.
        * ``attention_mask``, ``key_padding_mask``: the mask, if the backend takes one.
        * ``kv_cache``: a ``CausalKVCacheManager``; only backends whose
          ``support_kv_cache()`` is true accept it. ``k``/``v`` are then this call's
          new tokens, staged into the cache and attended together with what the
          cache holds before them.
        * ``causal_block_size``: with ``kv_cache``, cuts the new tokens into causal
          blocks, full attention within a block and causal across blocks.

        Under CUDA graphs a captured forward belongs to (cache, geometry, ``seq_len``,
        ``causal_block_size``) besides the tensor shapes: those are host values baked
        into the capture, so a graph runner must key on them. The cache's own state
        (table, lengths, slot ids) lives in device tensors rewritten in place by
        ``commit``, so commits need no recapture and ``staging_offset`` must not be
        part of a key.
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
        """Whether ``forward`` accepts a ``CausalKVCacheManager`` as ``kv_cache``. A backend
        without support would silently drop the keyword through ``**kwargs``, so callers
        must check before passing one."""
        return False
