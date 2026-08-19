# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Type

import torch

from ...attention_backend.interface import AttentionMetadata


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

    ``attn_metadata`` is a required argument on every forward, mirroring the LLM
    backend. Backends that do not consume metadata (VANILLA, FA4, CuTe DSL) accept
    and ignore it; only the TRTLLM backend reads it. See
    ``visual_gen/attention_backend/metadata.py`` for the site contract.
    """

    #: Concrete metadata type this backend consumes; callers construct from
    #: it, as the LLM engine does.
    Metadata: Type[AttentionMetadata] = AttentionMetadata

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        *,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor: ...

    @property
    @abstractmethod
    def preferred_layout(self) -> AttentionTensorLayout: ...

    @property
    def requires_metadata(self) -> bool:
        """Queries if the configured attention module requires a prepared metadata to run.

        For size-aligned diffusion models, many attention backends can run without a metadata.
        """
        return True

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        *,
        attn_metadata: AttentionMetadata,
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
