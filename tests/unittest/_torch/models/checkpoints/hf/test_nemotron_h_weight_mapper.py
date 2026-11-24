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

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.hf.nemotron_h_weight_mapper import (
    NemotronHHfWeightMapper,
)

pytestmark = pytest.mark.cpu_only


def _mapper() -> NemotronHHfWeightMapper:
    pretrained = SimpleNamespace(
        num_hidden_layers=2,
        mamba_head_dim=64,
        mamba_num_heads=8,
        n_groups=8,
        ssm_state_size=128,
    )
    mapping = SimpleNamespace(enable_attention_dp=False, tp_size=1, tp_rank=0)
    mapper = NemotronHHfWeightMapper()
    mapper._config = SimpleNamespace(
        pretrained_config=pretrained, mapping=mapping, moe_backend="TRTLLM"
    )
    return mapper


# `embeddings` is the name in NVIDIA's released checkpoints; transformers 5.x
# save_pretrained reverses its own load-time rename and writes `embedding`.
# Keys that merely contain the word, or are already mapped, must not change.
@pytest.mark.parametrize(
    ("source_key", "expected_key"),
    [
        ("backbone.embeddings.weight", "model.embed_tokens.weight"),
        ("backbone.embedding.weight", "model.embed_tokens.weight"),
        ("backbone.position_embeddings.weight", "model.position_embeddings.weight"),
        ("model.embed_tokens.weight", "model.embed_tokens.weight"),
    ],
)
def test_embedding_key_mapping(source_key: str, expected_key: str) -> None:
    table = torch.randn(16, 4)
    remapped = _mapper().preprocess_weights({source_key: table})
    assert list(remapped) == [expected_key]
    assert remapped[expected_key] is table
