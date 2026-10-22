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
    ) -> torch.Tensor: ...

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

    def accepts_fp8_qkv(self) -> bool:
        """Whether ``forward_fp8`` serves per-tensor e4m3 Q/K/V with the softmax scale in Q/K."""
        return False

    def fp8_v_amax(self, qkv: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor | None:
        """|V| maximum that sets the V quantization scale of packed QKV [T, 3*H*D].

        ``None`` lets the quantizer use the local maximum; sequence-parallel backends return the
        maximum over the group so every rank quantizes V with one scale.
        """
        return None

    def forward_fp8(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        v_descale: torch.Tensor,
        batch_size: int,
        seq_len: int,
        out_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention on e4m3 Q/K/V [B, S, H, D] quantized with the softmax scale folded into Q/K.

        Q and K carry ``x * sqrt(softmax_scale * log2(e))`` and V carries ``x / v_descale``.
        Returns [B, S, H, D] in BF16, or e4m3 quantized with ``out_scale`` (the dequant scale of
        the consumer's input) when given. Only called when ``accepts_fp8_qkv()`` is True.
        """
        raise NotImplementedError(f"{type(self).__name__} does not accept FP8 Q/K/V.")

    @classmethod
    def support_lse(cls) -> bool:
        """Whether the backend supports returning the softmax log-sum-exp (LSE) of the attention weights."""
        return False
