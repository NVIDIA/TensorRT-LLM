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
"""Native TRT-LLM Gemma4 audio tower (Conformer).

Replaces ``AutoModel.from_config(config.audio_config)`` in
``modeling_gemma4mm.py``. **Greenfield**: this is the first native Conformer
implementation in the fork (Phi4MM also uses HF reuse via ``importlib``
at ``modeling_phi4mm.py:106``). Conformer = mel-subsample conv + macaron
FFN + relative-position attention + depthwise conv branch -- not standard
ViT/transformer layout.

Architecture references (HF transformers 5.5.3, verified 2026-05-14 spike):
- ``Gemma4AudioRelPositionalEncoding`` @ ``transformers/models/gemma4/modeling_gemma4.py:178``
- ``Gemma4AudioAttention`` @ ``:209``
- ``Gemma4AudioLayer`` @ ``:485``

Migration plan: ``GEMMA4_MM_TOWER_MIGRATION.md`` §3 Phase 2.
Gated behind vision Tier 3 parity per acceptance gate matrix (§1.6.10).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..model_config import ModelConfig


class Gemma4AudioModel(nn.Module):
    """Gemma4 audio tower (skeleton; bodies filled by subsequent commits).

    Input contract (preserved from HF ``Gemma4AudioModel.forward``, see
    ``modeling_gemma4mm.py:800`` call site):
        audio_features: (B, mel_T, 128) -- mel-spectrogram frames
        attention_mask: (B, mel_T) bool -- valid frame mask

    Output: object with ``last_hidden_state`` ``(B, T_audio, H_audio)`` and
    optional ``attention_mask`` (bool, post-subsample) used by caller to drop
    padding frames before projection.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config

    @torch.inference_mode()
    def forward(
        self,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        raise NotImplementedError(
            "Gemma4AudioModel.forward is a skeleton; implementation pending "
            "(greenfield Conformer; see GEMMA4_MM_TOWER_MIGRATION.md §3 Phase 2)."
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        raise NotImplementedError(
            "Gemma4AudioModel.load_weights is a skeleton; implementation "
            "pending (HF↔TRT-LLM key remap for Conformer layers)."
        )
