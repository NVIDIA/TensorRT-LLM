# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""YaRN rotary frequencies built from a GPT-OSS config match Hugging Face."""

from types import SimpleNamespace

import pytest
import torch
from transformers import GptOssConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from tensorrt_llm._torch.attention.backends import interface
from tensorrt_llm._torch.attention.backends.interface import RopeParams

pytestmark = pytest.mark.cpu_only


def _gpt_oss_config(**rope_overrides):
    # openai/gpt-oss-120b config.json rope settings ("truncate": false).
    rope = {
        "rope_type": "yarn",
        "rope_theta": 150000.0,
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 4096,
        "truncate": False,
    }
    rope.update(rope_overrides)
    return GptOssConfig(
        max_position_embeddings=64,
        head_dim=64,
        hidden_size=2880,
        num_attention_heads=64,
        num_key_value_heads=8,
        rope_parameters=rope,
    )


@pytest.fixture
def host_rope_tables(monkeypatch):
    """create_rope_const_params puts its tables on CUDA; keep them on the host."""
    host_torch = SimpleNamespace(
        tensor=lambda data, dtype=None, device=None: torch.as_tensor(data, dtype=dtype),
        float32=torch.float32,
    )
    monkeypatch.setattr(interface, "torch", host_torch)


@pytest.mark.parametrize("truncate", [False, True])
def test_yarn_inv_freq_from_config_matches_hf(host_rope_tables, truncate):
    config = _gpt_oss_config(truncate=truncate)
    inv_freq, _ = RopeParams.from_config(config).create_rope_const_params()
    expected, _ = ROPE_INIT_FUNCTIONS["yarn"](config, "cpu")
    torch.testing.assert_close(inv_freq, expected, rtol=1e-5, atol=0)
