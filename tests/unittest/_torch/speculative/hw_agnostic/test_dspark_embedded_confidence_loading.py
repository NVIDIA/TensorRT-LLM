# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedded-wrapper validation through the real shared parameter walk."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_dspark, modeling_dspark_v41
from tensorrt_llm._torch.models.modeling_speculative import DSparkConfidenceHead

pytestmark = pytest.mark.cpu_only


def _wrapper(monkeypatch, v41):
    module = modeling_dspark_v41 if v41 else modeling_dspark
    cls = module.DSv41DSparkForCausalLM if v41 else module.DSv4DSparkForCausalLM
    wrapper = object.__new__(cls)
    nn.Module.__init__(wrapper)
    stage = nn.Module()
    stage.main_proj = nn.Linear(2, 2, bias=False)
    stage.main_norm = nn.RMSNorm(2)
    stage.norm = nn.RMSNorm(2)
    stage.markov_head = None
    stage.confidence_head = DSparkConfidenceHead(hidden_size=2, block_size=2)
    model = nn.Module()
    model.mtp_layers = nn.ModuleList([stage])
    model.config = SimpleNamespace(
        q_lora_rank=1,
        num_attention_heads=1,
        num_hidden_layers=0,
        num_nextn_predict_layers=1,
        kv_lora_rank=1,
    )
    model.model_config = SimpleNamespace(
        mapping=SimpleNamespace(
            tp_rank=0,
            tp_size=1,
            cp_rank=0,
            cp_size=1,
            enable_attention_dp=False,
        )
    )
    model.post_load_weights = Mock()
    model.cache_attn_weights_from_state_dict = Mock()
    wrapper.dspark_model = model
    wrapper.config = model.config
    wrapper.num_stages = 1
    # Remapping has separate checkpoint-layout tests. This test starts with
    # model-named tensors and does not mock the loader or its parameter copies.
    remapper = "remap_dspark_v41_draft_keys" if v41 else "remap_dspark_draft_keys"
    monkeypatch.setattr(module, remapper, lambda weights, *args, **kwargs: weights)
    weights = {name: torch.full_like(param, 2.0) for name, param in model.named_parameters()}
    return wrapper, weights


@pytest.mark.parametrize("v41", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_embedded_confidence_load_validates_and_copies_real_parameters(monkeypatch, v41, bias):
    wrapper, weights = _wrapper(monkeypatch, v41)
    head = wrapper.dspark_model.mtp_layers[-1].confidence_head
    prefix = "mtp_layers.0.confidence_head."
    if bias:
        weights[prefix + "proj.bias"] = torch.tensor([3.0])
    weight_pointer = head.proj.weight.data_ptr()
    rng_before = torch.get_rng_state()
    wrapper.load_weights(weights)
    assert head.proj.weight.data_ptr() == weight_pointer
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(head.proj.weight, weights[prefix + "proj.weight"])
    assert (head.proj.bias is not None) is bias
    if bias:
        assert torch.equal(head.proj.bias, weights[prefix + "proj.bias"])
    wrapper.dspark_model.post_load_weights.assert_called_once()


@pytest.mark.parametrize("v41", [False, True])
@pytest.mark.parametrize("fault", ["unknown", "missing_weight", "broadcast_weight"])
def test_embedded_confidence_rejects_incomplete_or_unknown_tensors(monkeypatch, v41, fault):
    wrapper, weights = _wrapper(monkeypatch, v41)
    prefix = "mtp_layers.0.confidence_head."
    if fault == "unknown":
        weights[prefix + "bias"] = torch.tensor([99.0])
    elif fault == "missing_weight":
        weights.pop(prefix + "proj.weight")
        weights[prefix + "proj.bias"] = torch.tensor([3.0])
    else:
        weights[prefix + "proj.weight"] = torch.tensor([[2.0]])
    with pytest.raises(ValueError, match="confidence head checkpoint"):
        wrapper.load_weights(weights)
    wrapper.dspark_model.post_load_weights.assert_not_called()


def test_v41_entirely_absent_confidence_head_remains_supported(monkeypatch):
    wrapper, weights = _wrapper(monkeypatch, True)
    weights.pop("mtp_layers.0.confidence_head.proj.weight")
    wrapper.load_weights(weights)
    assert wrapper.dspark_model.mtp_layers[-1].confidence_head is None
    wrapper.dspark_model.post_load_weights.assert_called_once()
