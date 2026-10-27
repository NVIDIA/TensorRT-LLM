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
"""Tests for how DWDP setup finds a model's decoder layers and MoE experts."""

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.modules.dwdp.setup import (
    _get_decoder_model,
    _get_gate_module,
    _get_moe_and_experts,
)

pytestmark = pytest.mark.cpu_only


def _decoder():
    decoder = nn.Module()
    decoder.layers = nn.ModuleList()
    return decoder


def test_decoder_of_a_causal_lm():
    causal_lm = nn.Module()
    causal_lm.model = _decoder()
    assert _get_decoder_model(causal_lm) is causal_lm.model


def test_decoder_under_a_multimodal_llm_wrapper():
    # MuseSparkOpenOmniForConditionalGeneration holds the text CausalLM as .llm.
    wrapper = nn.Module()
    wrapper.llm = nn.Module()
    wrapper.llm.model = _decoder()
    assert _get_decoder_model(wrapper) is wrapper.llm.model


def test_moe_under_feed_forward():
    # MuseSparkOpenDecoderLayer.feed_forward (MuseSparkOpenMoE) -> .experts
    # (ConfigurableMoE) -> .backend, which owns the expert weights.
    backend = nn.Module()
    backend.w3_w1_weight = nn.Parameter(torch.empty(0), requires_grad=False)
    experts = nn.Module()
    experts.backend = backend
    feed_forward = nn.Module()
    feed_forward.experts = experts
    feed_forward.expert_bias = nn.Parameter(torch.zeros(4), requires_grad=False)
    layer = nn.Module()
    layer.feed_forward = feed_forward

    moe_module, experts_module = _get_moe_and_experts(layer)

    assert moe_module is feed_forward
    assert experts_module is backend
    # The expert bias is full on the MoE module; there is no gate to allgather.
    assert _get_gate_module(moe_module) is None
