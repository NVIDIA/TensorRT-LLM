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
"""Abstract base classes for encoder layers and encoder-decoder layers."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

from ..attention_backend import AttentionMetadata


class EncoderLayer(nn.Module, ABC):
    """Abstract base class for encoder layers (self-attention only, non-causal)."""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_ids: Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> torch.Tensor: ...


class EncoderDecoderLayer(nn.Module, ABC):
    """Abstract base class for decoder layers with cross-attention.

    Order: self-attention → cross-attention → MLP.
    """

    @abstractmethod
    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attn_metadata: Optional[AttentionMetadata] = None,
        skip_cross_kv_projection: bool = False,
        **kwargs,
    ) -> torch.Tensor: ...
