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

"""Eager Nemotron audio encoder helper for the AutoDeploy custom model.

This module intentionally mirrors the native Nemotron audio tower path used by
`_encode_audio_data(...)` in `tensorrt_llm._torch.models.modeling_nemotron_nano`.
The audio tower remains eager while AutoDeploy exports only the text model.
"""

from typing import Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.models.modeling_parakeet import ProjectedParakeet


class NemotronAudioEncoder(nn.Module):
    """Vanilla-PyTorch wrapper around `ProjectedParakeet`.

    The forward returns padded projected embeddings plus the valid per-clip
    output lengths. Callers can flatten the valid rows with
    `flatten_valid_outputs(...)` to reproduce native Nemotron
    `_encode_audio_data(...)` semantics exactly.
    """

    def __init__(
        self,
        sound_config: PretrainedConfig,
        llm_hidden_size: int,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        parakeet = ProjectedParakeet(
            sound_config=sound_config,
            llm_hidden_size=llm_hidden_size,
            dtype=dtype,
        )
        self.encoder = parakeet.encoder
        self.projection = parakeet.projection

    def forward(
        self,
        input_audio_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(
            input_features=input_audio_features,
            attention_mask=feature_attention_mask,
        )
        sound_embeds = self.projection(outputs.last_hidden_state)
        valid_input_lens = feature_attention_mask.sum(dim=1)
        valid_output_lens = self.encoder._get_subsampling_output_length(valid_input_lens).to(
            torch.int32
        )
        return sound_embeds, valid_output_lens

    @staticmethod
    def flatten_valid_outputs(
        sound_embeds: torch.Tensor,
        valid_output_lens: torch.Tensor,
    ) -> torch.Tensor:
        chunks = []
        for i in range(sound_embeds.shape[0]):
            valid_len = int(valid_output_lens[i].item())
            chunks.append(sound_embeds[i, :valid_len])
        if not chunks:
            return torch.empty(
                0,
                sound_embeds.shape[-1],
                device=sound_embeds.device,
                dtype=sound_embeds.dtype,
            )
        return torch.cat(chunks, dim=0)
