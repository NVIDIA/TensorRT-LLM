# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import tempfile

import torch

from tensorrt_llm._torch.configs.deepseek_v4 import DeepseekV4Config
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4ForCausalLM
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config


def test_deepseek_v4_config_loading():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = os.path.join(tmp_dir, "config.json")
        config_data = {
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "q_lora_rank": 64,
            "o_lora_rank": 64,
            "head_dim": 64,
            "index_head_dim": 32,
            "index_n_heads": 4,
            "index_topk": 16,
            "num_hash_layers": 1,
            "hc_sinkhorn_iters": 5,
            "hc_mult": 2,
            "compress_ratios": [0, 4],
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 128,
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = load_pretrained_config(tmp_dir)
        assert isinstance(config, DeepseekV4Config)
        assert config.hidden_size == 256


def test_deepseek_v4_forward_pass():
    config = DeepseekV4Config(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        q_lora_rank=64,
        o_lora_rank=64,
        head_dim=64,
        qk_rope_head_dim=32,
        index_head_dim=32,
        index_n_heads=4,
        index_topk=16,
        num_hash_layers=1,
        hc_sinkhorn_iters=5,
        hc_mult=2,
        compress_ratios=[0, 4],
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
    )

    # Wrap in ModelConfig to match the new init signature
    model_config = ModelConfig(pretrained_config=config)

    model = DeepseekV4ForCausalLM(model_config)

    # Dummy input
    input_ids = torch.randint(0, 1000, (2, 8))  # batch=2, seqlen=8

    # Forward pass
    logits = model(input_ids)

    # ParallelHead.get_logits extracts the last token, so shape is [batch, vocab_size]
    assert logits.shape == (2, config.vocab_size)


def test_deepseek_v4_decoding_loop():
    config = DeepseekV4Config(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        q_lora_rank=64,
        o_lora_rank=64,
        head_dim=64,
        qk_rope_head_dim=32,
        index_head_dim=32,
        index_n_heads=4,
        index_topk=16,
        num_hash_layers=1,
        hc_sinkhorn_iters=5,
        hc_mult=2,
        compress_ratios=[0, 4],
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        o_groups=8,
    )
    model_config = ModelConfig(pretrained_config=config)
    model = DeepseekV4ForCausalLM(model_config)

    input_ids = torch.randint(0, 1000, (1, 8))
    logits = model(input_ids, start_pos=0)

    next_token = logits.argmax(dim=-1).unsqueeze(-1)

    for step in range(5):
        current_pos = 8 + step
        logits = model(next_token, start_pos=current_pos)
        next_token = logits.argmax(dim=-1).unsqueeze(-1)
        print(f"Step {step} generated token, logits shape: {logits.shape}")
