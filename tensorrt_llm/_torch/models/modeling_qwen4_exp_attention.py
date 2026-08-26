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
"""QSA full-attention module for Qwen3.8-Flash-Next."""

from __future__ import annotations

from typing import Optional

import torch

from ..attention_backend.interface import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_qwen3 import Qwen3Attention


class Qwen4ExpAttention(Qwen3Attention):
    """Qwen4-Exp full-attention (QSA) module.

    Reuses the Qwen3 QK-norm attention stack for projection, partial RoPE,
    output gating, and output projection. The registered QSA sparse hook owns
    the compressed index cache and exact paged sparse-GQA path.
    """

    def __init__(self, model_config: ModelConfig, layer_idx: int, *, reduce_output: bool = False):
        super().__init__(
            model_config,
            layer_idx=layer_idx,
            fuse_qk_norm_rope=True,
            attn_output_gate=True,
            use_gemma_rms_norm=True,
            reduce_output=reduce_output,
        )
        # Enable the fused split-gate + Gemma qk-norm + RoPE kernel (matches
        # Qwen3Next's full-attention path).
        self._fuse_qk_norm_rope_gate = True

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Forward the pre-projection inputs required by the QSA indexer."""
        kwargs["qsa_index_hidden_states"] = hidden_states
        kwargs["qsa_position_ids"] = position_ids
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

    @property
    def is_qsa(self) -> bool:
        """Whether this layer carries a QSA compressed sparse indexer."""
        return getattr(self, "indexer", None) is not None
