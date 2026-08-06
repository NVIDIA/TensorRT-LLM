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

from unittest.mock import patch

import torch
from torch import nn

from tensorrt_llm._torch.attention_backend.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.mla import MLA
from tensorrt_llm.functional import PositionEmbeddingType


class _FakeAttention(nn.Module):
    def support_fused_rope(self) -> bool:
        return True

    def update_quant_config(self, _quant_config: object) -> None:
        pass


def _make_mla(config: ModelConfig) -> MLA:
    position_embedding = PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=RopeParams(dim=2, max_positions=8),
    )
    return MLA(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        predicted_tokens_per_seq=1,
        max_position_embeddings=8,
        bias=False,
        pos_embd_params=position_embedding,
        layer_idx=0,
        dtype=torch.bfloat16,
        config=config,
        o_lora_rank=2,
    )


def test_duplicate_layer_ids_preserve_all_mla_registrations() -> None:
    target_config = ModelConfig(skip_create_weights_in_init=True)
    draft_config = ModelConfig(skip_create_weights_in_init=True)
    next_config = ModelConfig(skip_create_weights_in_init=True)
    draft_config.extra_attrs = target_config.extra_attrs
    next_config.extra_attrs = target_config.extra_attrs

    with patch(
        "tensorrt_llm._torch.modules.mla.create_attention",
        side_effect=lambda *args, **kwargs: _FakeAttention(),
    ):
        target_mla = _make_mla(target_config)
        draft_mla = _make_mla(draft_config)
        next_mla = _make_mla(next_config)

    assert target_mla.layer_idx == draft_mla.layer_idx == next_mla.layer_idx == 0
    assert target_mla.layer_idx_str == "0"
    assert draft_mla.layer_idx_str == "0_0"
    assert next_mla.layer_idx_str == "0_1"
    registry = target_config.extra_attrs["mla_layers"]
    assert registry["0"]() is target_mla
    assert registry["0_0"]() is draft_mla
    assert registry["0_1"]() is next_mla
