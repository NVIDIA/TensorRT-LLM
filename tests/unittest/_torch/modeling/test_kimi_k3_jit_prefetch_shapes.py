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
"""Kimi K3 names, before its weights load, the FP8 weight-read GEMMs
``_finalize_weight_load`` will create (``jit_prefetch_fp8_weight_read_shapes``)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.cpu_only

_FP8 = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)


def _fp8_linear(n, k, tp_size=1):
    lin = Linear(
        k, n, bias=False, dtype=torch.bfloat16, quant_config=_FP8, skip_create_weights_in_init=True
    )
    with torch.device("meta"):
        lin.create_weights()
    lin.tp_size = tp_size
    return lin


def _bf16_linear(n, k, tp_size=1):
    lin = Linear(k, n, bias=False, dtype=torch.bfloat16, skip_create_weights_in_init=True)
    with torch.device("meta"):
        lin.create_weights()
    lin.tp_size = tp_size
    return lin


def _kda_layer(h, proj, gate_full_rank=True, path="optimized"):
    attn = nn.Module()
    for name in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj"):
        setattr(
            attn,
            name,
            _fp8_linear(proj if name != "o_proj" else h, h if name != "o_proj" else proj),
        )
    attn.use_full_rank_gate = gate_full_rank
    attn._dispatch = SimpleNamespace(decode_kernel_path=path)
    layer = nn.Module()
    layer.is_kda = True
    layer.linear_attn = attn
    return layer


def _moe(layer, h, latent, shared_inter, shared_tp=1):
    moe = nn.Module()
    with torch.device("meta"):
        moe.routed_expert_down_proj = nn.Linear(h, latent, bias=False)
        moe.routed_expert_up_proj = nn.Linear(latent, h, bias=False)
    shared = nn.Module()
    shared.gate_up_proj = _bf16_linear(2 * shared_inter, h, shared_tp)
    shared.down_proj = _bf16_linear(h, shared_inter, shared_tp)
    moe.shared_experts = shared
    layer.block_sparse_moe = moe
    return layer


def _k3(layers, attention_dp=True, mlp_fp8=True):
    model = object.__new__(mkl.KimiLinearForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList(layers)
    model.model_config = SimpleNamespace(mapping=SimpleNamespace(enable_attention_dp=attention_dp))
    model._fp8_weight_read_moe_mlp = mlp_fp8
    return model


@pytest.fixture(autouse=True)
def _blackwell(monkeypatch):
    monkeypatch.setattr(mkl, "is_sm_100f", lambda *a: True)
    monkeypatch.setattr(mkl, "_rubin_mxfp8_is_available", lambda: False)


def test_kda_qkvg_fusion_and_converted_moe_mlps():
    h, proj, latent, inter = 512, 384, 640, 128
    model = _k3([_moe(_kda_layer(h, proj), h, latent, inter)])
    assert model.jit_prefetch_fp8_weight_read_shapes() == {
        (4 * proj, h),  # the fused [q | k | v | g] projection
        (proj, h),  # q / k / v / g, still read individually by prefill
        (h, proj),  # o_proj
        (latent, h),  # routed_expert_down_proj
        (h, latent),  # routed_expert_up_proj
        (2 * inter, h),  # shared gate_up_proj (attention DP)
        (h, inter),  # shared down_proj
    }


def test_rules_follow_the_conversion_conditions():
    h, proj, latent, inter = 512, 384, 640, 128
    # No attention DP: the fused shared gate_up stays BF16. A TP-sharded
    # shared expert is not converted at all.
    tp = _k3([_moe(_kda_layer(h, proj), h, latent, inter)], attention_dp=False)
    assert (2 * inter, h) not in tp.jit_prefetch_fp8_weight_read_shapes()
    sharded = _k3([_moe(_kda_layer(h, proj), h, latent, inter, shared_tp=4)])
    assert (h, inter) not in sharded.jit_prefetch_fp8_weight_read_shapes()
    # Without KIMI_K3_FP8_WEIGHT_READ_MOE_MLP only the attention converts.
    attn_only = _k3([_moe(_kda_layer(h, proj), h, latent, inter)], mlp_fp8=False)
    assert attn_only.jit_prefetch_fp8_weight_read_shapes() == {(4 * proj, h), (proj, h), (h, proj)}
    # The FLA decode path and the low-rank gate keep the projections unfused.
    for layer in (_kda_layer(h, proj, path="fla"), _kda_layer(h, proj, gate_full_rank=False)):
        assert (4 * proj, h) not in _k3([layer]).jit_prefetch_fp8_weight_read_shapes()


def test_off_blackwell_nothing_is_predicted(monkeypatch):
    monkeypatch.setattr(mkl, "is_sm_100f", lambda *a: False)
    assert _k3([_kda_layer(512, 256)]).jit_prefetch_fp8_weight_read_shapes() == set()
