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
"""Tests of the dense_mlp call site in the target's decoder layer.

``GlmDecoderLayer.forward_mlp`` is exercised on a decode step on a minimal layer object
(dense GatedMLP, single rank, no fusion), with and without ``dense_mlp`` selected.
"""

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plan_testlib  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

HIDDEN = 2048
INTER = 4096


@pytest.fixture(scope="module", autouse=True)
def dense_mlp_op():
    import dense_mlp_utils as wu

    wu.ensure_op()


@pytest.fixture(autouse=True)
def decode_step():
    """The owned call site runs on decode steps."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    with dispatch.step(dispatch.Phase.DECODE):
        yield


def _select(monkeypatch, value):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    plan_testlib.select(dispatch, value, monkeypatch)


@pytest.fixture
def spy(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    dispatch.get_impl("dense_mlp", "dense_mlp_forward")
    real = dispatch._REGISTRY["dense_mlp"]["dense_mlp_forward"]

    class Calls(list):
        pass

    calls = Calls()

    def wrapped(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result is not None)
        if result is not None:
            wrapped.last = (args[0], args[1].clone(), args[2], result.clone())
        return result

    wrapped.last = None

    monkeypatch.setitem(dispatch._REGISTRY["dense_mlp"], "dense_mlp_forward", wrapped)
    calls.wrapped = wrapped
    return calls


def _layer():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers import (  # noqa: E501
        GlmDecoderLayer,
    )
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_utils import EagerFusionConfig
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    torch.manual_seed(0)
    mlp = GatedMLP(
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        bias=False,
        dtype=torch.bfloat16,
        config=ModelConfig(),
        reduce_output=False,
    ).cuda()
    with torch.no_grad():
        mlp.gate_up_proj.weight.normal_(0, 0.02)
        mlp.down_proj.weight.normal_(0, 0.02)
    layer = types.SimpleNamespace(
        mlp=mlp,
        fusion_config=EagerFusionConfig(),
        mlp_tp_size=1,
        allreduce=None,
        layer_idx=0,
        post_attention_layernorm=RMSNorm(hidden_size=HIDDEN, eps=1e-5, dtype=torch.bfloat16).cuda(),
        next_layer_layernorm=RMSNorm(hidden_size=HIDDEN, eps=1e-5, dtype=torch.bfloat16).cuda(),
    )
    layer._apply_next_layer_layernorm = types.MethodType(
        GlmDecoderLayer._apply_next_layer_layernorm, layer
    )
    return layer, GlmDecoderLayer.forward_mlp


def _run(layer, forward_mlp, rows, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(rows, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    residual = torch.randn(rows, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    return forward_mlp(layer, hidden, residual)


@pytest.mark.parametrize("rows", [1, 6, 8])
def test_dense_mlp_engages_and_matches_native(monkeypatch, spy, rows):
    layer, forward_mlp = _layer()
    _select(monkeypatch, "")
    _run(layer, forward_mlp, rows)
    assert spy == []
    _select(monkeypatch, "dense_mlp")
    hidden, residual = _run(layer, forward_mlp, rows)
    assert spy == [True]
    # The MLP output handed back to forward_mlp matches the native module on
    # the same input within the BF16 GEMM tolerance.
    mlp, x_in, params, dense_mlp_out = spy.wrapped.last
    torch.testing.assert_close(
        dense_mlp_out, mlp(x_in, final_all_reduce_params=params), rtol=1e-2, atol=5e-3
    )
    # Everything after the MLP stays native: the native forward_mlp fed with
    # the same MLP output reproduces the dense_mlp layer outputs bit for bit.
    _select(monkeypatch, "")
    native_mlp = layer.mlp
    layer.mlp = lambda x, final_all_reduce_params=None: dense_mlp_out.clone()
    ref_hidden, ref_residual = _run(layer, forward_mlp, rows)
    layer.mlp = native_mlp
    assert torch.equal(hidden, ref_hidden)
    assert torch.equal(residual, ref_residual)


def test_ineligible_rows_take_native_path(monkeypatch, spy):
    layer, forward_mlp = _layer()
    _select(monkeypatch, "")
    nat_hidden, nat_residual = _run(layer, forward_mlp, 12)
    _select(monkeypatch, "dense_mlp")
    hidden, residual = _run(layer, forward_mlp, 12)
    assert spy == [False]
    assert torch.equal(hidden, nat_hidden)
    assert torch.equal(residual, nat_residual)


def test_other_features_do_not_engage_dense_mlp(monkeypatch, spy):
    layer, forward_mlp = _layer()
    _select(monkeypatch, "absorbed_query")
    _run(layer, forward_mlp, 6)
    assert spy == []
