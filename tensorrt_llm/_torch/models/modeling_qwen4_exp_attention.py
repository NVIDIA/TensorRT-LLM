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

from typing import Optional

import torch

from ..attention_backend.interface import AttentionMetadata
from ..attention_backend.sparse.qsa.indexer import QSAIndexer
from ..attention_backend.sparse.qsa.params import QSASparseParams
from ..model_config import ModelConfig
from .modeling_qwen3 import Qwen3Attention


class Qwen4ExpAttention(Qwen3Attention):
    """Qwen4-Exp full-attention (QSA) module.

    Reuses the Qwen3 QK-norm attention stack for projection, partial RoPE,
    output gating, and output projection. This module owns the QSA indexer
    (a checkpoint-defined submodule); the registered QSA sparse hook drives
    the compressed index cache and exact paged sparse-GQA path.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        *,
        reduce_output: bool = False,
    ) -> None:
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

        # The indexer carries checkpoint weights, so it has to be a submodule
        # of this nn.Module rather than of the (plain-object) attention
        # backend.  Building it here keeps `skip_create_weights_in_init`
        # sourced from the model config instead of an extra Attention field.
        # Layers configured without sparse attention stay dense and simply
        # never get an indexer (see `is_qsa`).
        if self.sparse_attn_hooks is not None:
            params = self.sparse_params
            if not isinstance(params, QSASparseParams):
                raise TypeError(
                    f"Qwen4ExpAttention requires QSASparseParams, got {type(params).__name__}"
                )
            self.indexer = QSAIndexer(
                self,
                params,
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Forward the pre-projection inputs required by the QSA indexer."""
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            qsa_index_hidden_states=hidden_states,
            qsa_position_ids=position_ids,
            **kwargs,
        )

    @property
    def is_qsa(self) -> bool:
        """Whether this layer carries a QSA compressed sparse indexer."""
        return getattr(self, "indexer", None) is not None
