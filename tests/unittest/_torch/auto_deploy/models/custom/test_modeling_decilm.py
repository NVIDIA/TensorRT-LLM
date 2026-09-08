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

import types

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_decilm import (
    DeciLMCausalLMOutput,
    DeciLMForCausalLM,
)


def _make_block_config(no_op_attn, n_heads_in_group, no_op_ffn, ffn_mult):
    attn = types.SimpleNamespace(no_op=no_op_attn, n_heads_in_group=n_heads_in_group)
    ffn = types.SimpleNamespace(no_op=no_op_ffn, ffn_mult=ffn_mult)
    return types.SimpleNamespace(attention=attn, ffn=ffn)


def _make_config():
    config = PretrainedConfig()
    config.hidden_size = 64
    config.num_attention_heads = 4
    config.num_hidden_layers = 3
    config.vocab_size = 128
    config.pad_token_id = 0
    config.rms_norm_eps = 1e-6
    config.hidden_act = "silu"
    config.attention_bias = False
    config.mlp_bias = False
    config.initializer_range = 0.02
    config.max_position_embeddings = 32
    config.rope_theta = 10000.0
    config.rope_scaling = None
    config.tie_word_embeddings = True
    config.block_configs = [
        _make_block_config(no_op_attn=False, n_heads_in_group=1, no_op_ffn=False, ffn_mult=1.0),
        _make_block_config(no_op_attn=True, n_heads_in_group=1, no_op_ffn=False, ffn_mult=0.5),
        _make_block_config(no_op_attn=False, n_heads_in_group=2, no_op_ffn=False, ffn_mult=1.5),
    ]
    return config


class TestDeciLMForCausalLM:
    def test_construction(self):
        cfg = _make_config()
        model = DeciLMForCausalLM(cfg)

        assert model.vocab_size == cfg.vocab_size
        assert model.lm_head.in_features == cfg.hidden_size
        assert model.lm_head.out_features == cfg.vocab_size
        assert model.model.embed_tokens.num_embeddings == cfg.vocab_size
        assert model.model.embed_tokens.embedding_dim == cfg.hidden_size
        assert len(model.model.layers) == cfg.num_hidden_layers

        layer0 = model.model.layers[0]
        assert layer0.has_attention is True
        assert layer0.has_ffn is True
        assert hasattr(layer0, "self_attn")
        assert hasattr(layer0, "mlp")

        layer1 = model.model.layers[1]
        assert layer1.has_attention is False
        assert layer1.has_ffn is True
        assert not hasattr(layer1, "self_attn")
        assert hasattr(layer1, "mlp")

        layer2 = model.model.layers[2]
        assert layer2.has_attention is True
        assert layer2.has_ffn is True
        assert layer2.self_attn.num_key_value_heads == cfg.num_attention_heads // 2

    def test_forward(self):
        cfg = _make_config()
        device = torch.device("cuda")
        model = DeciLMForCausalLM(cfg).to(device).eval()

        B, S = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=device)
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            output = model(input_ids=input_ids, position_ids=position_ids)

        assert isinstance(output, DeciLMCausalLMOutput)
        assert output.logits is not None
        assert output.logits.shape == (B, S, cfg.vocab_size)
        assert output.logits.dtype == torch.float32
        assert torch.isfinite(output.logits).all()
