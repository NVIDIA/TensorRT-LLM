# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""YaRN rotary frequencies built from a GPT-OSS config match Hugging Face."""

from types import SimpleNamespace

import pytest
import torch
from transformers import GptOssConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.attention.backends import interface
from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.models import modeling_gpt_oss
from tensorrt_llm._torch.models.modeling_gpt_oss import AttentionBlock, GptOssForCausalLM

pytestmark = pytest.mark.cpu_only


def _gpt_oss_config(drop=(), **rope_overrides):
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
    for key in drop:
        del rope[key]
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


def test_yarn_inv_freq_without_truncate_key_defaults_to_true(host_rope_tables):
    # Configs that do not carry the key (every non-GPT-OSS YaRN model) keep the
    # floored/ceiled bounds.
    missing, _ = RopeParams.from_config(
        _gpt_oss_config(drop=("truncate",))
    ).create_rope_const_params()
    explicit, _ = RopeParams.from_config(_gpt_oss_config(truncate=True)).create_rope_const_params()
    not_truncated, _ = RopeParams.from_config(
        _gpt_oss_config(truncate=False)
    ).create_rope_const_params()
    torch.testing.assert_close(missing, explicit, rtol=0, atol=0)
    assert not torch.allclose(missing, not_truncated)


@pytest.fixture
def stub_gpu_init(monkeypatch):
    """Skip the GPU-side construction of GPT-OSS; the config logic still runs."""
    captured = {}

    def attention_init(self, *, pos_embd_params, num_attention_heads, **kwargs):
        torch.nn.Module.__init__(self)
        captured["rope"] = pos_embd_params.rope
        self.attn_backend = "TRTLLM"
        self.tp_size = 1

    monkeypatch.setattr(Attention, "__init__", attention_init)
    monkeypatch.setattr(modeling_gpt_oss, "Transformer", lambda model_config: None)
    monkeypatch.setattr(
        modeling_gpt_oss.SpecDecOneEngineForCausalLM, "__init__", lambda self, *a, **k: None
    )
    return captured


def _pretrained_config(rope_scaling):
    return SimpleNamespace(
        head_dim=64,
        hidden_size=2880,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        rope_theta=150000.0,
        rope_scaling=rope_scaling,
        sliding_window=128,
        rms_norm_eps=1e-5,
        torch_dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    "rope_extra, expected",
    [({"truncate": False}, False), ({"truncate": True}, True), ({}, True)],
)
def test_gpt_oss_attention_block_forwards_truncate(stub_gpu_init, rope_extra, expected):
    rope_scaling = {
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 4096,
        **rope_extra,
    }
    model_config = SimpleNamespace(pretrained_config=_pretrained_config(rope_scaling))
    AttentionBlock(model_config)
    assert stub_gpu_init["rope"].truncate is expected


def test_gpt_oss_original_format_config_disables_truncate(stub_gpu_init):
    # openai/gpt-oss checkpoints ship the original (non-HF) config keys.
    pretrained = _pretrained_config(None)
    pretrained.num_experts = 32
    pretrained.experts_per_token = 4
    pretrained.rope_scaling_factor = 32.0
    pretrained.rope_ntk_beta = 32.0
    pretrained.rope_ntk_alpha = 1.0
    pretrained.initial_context_length = 4096
    model_config = SimpleNamespace(
        pretrained_config=pretrained,
        quant_config=SimpleNamespace(exclude_modules=None),
        mapping=SimpleNamespace(pp_size=1),
    )
    GptOssForCausalLM(model_config)
    assert pretrained.rope_scaling["truncate"] is False
    AttentionBlock(model_config)
    assert stub_gpu_init["rope"].truncate is False
