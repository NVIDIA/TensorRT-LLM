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
"""Negative-path + dispatch tests for CuteDslB12xFusedMoE.

These checks run without a GPU: they verify the can_implement() gating
matrix, the SM120/SM121 + NVFP4 selection in create_moe.get_moe_cls (the
backend is selected on the `moe_backend=CUTEDSL` path when flashinfer
is importable, never from `moe_backend=CUTLASS`), and the hybrid
CUTLASS-prefill / b12x-decode dispatch predicate. Functional
correctness of the b12x kernel is covered by end-to-end model tests on
SM120/SM121 hardware.
"""

import sys
import types
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.create_moe import get_moe_cls
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl_b12x import CuteDslB12xFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.quantization import (
    NVFP4CuteDslB12xFusedMoEMethod,
    NVFP4CutlassFusedMoEMethod,
)
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

_FUSED_MOE_MODULE = "tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl_b12x"


@pytest.mark.parametrize("sm_version", [80, 89, 90, 100, 103])
def test_can_implement_rejects_unsupported_sm(sm_version):
    """can_implement returns False on every SM outside the supported set."""
    with patch(f"{_FUSED_MOE_MODULE}.get_sm_version", return_value=sm_version):
        ok, reason = CuteDslB12xFusedMoE.can_implement(QuantAlgo.NVFP4)
    assert not ok
    assert reason is not None and f"SM{sm_version}" in reason


@pytest.mark.parametrize("sm_version", sorted(CuteDslB12xFusedMoE._SUPPORTED_SM_VERSIONS))
def test_can_implement_accepts_supported_sm_with_nvfp4(sm_version):
    with patch(f"{_FUSED_MOE_MODULE}.get_sm_version", return_value=sm_version):
        ok, reason = CuteDslB12xFusedMoE.can_implement(QuantAlgo.NVFP4)
    assert ok
    assert reason is None


@pytest.mark.parametrize("sm_version", sorted(CuteDslB12xFusedMoE._SUPPORTED_SM_VERSIONS))
def test_can_implement_accepts_supported_sm_with_w4a16_nvfp4(sm_version):
    with patch(f"{_FUSED_MOE_MODULE}.get_sm_version", return_value=sm_version):
        ok, reason = CuteDslB12xFusedMoE.can_implement(QuantAlgo.W4A16_NVFP4)
    assert ok
    assert reason is None


@pytest.mark.parametrize(
    "quant_algo",
    [
        None,
        QuantAlgo.FP8,
        QuantAlgo.FP8_BLOCK_SCALES,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_FP8,
    ],
)
def test_can_implement_rejects_non_nvfp4(quant_algo):
    """Only NVFP4 is supported; everything else must be turned away."""
    with patch(f"{_FUSED_MOE_MODULE}.get_sm_version", return_value=120):
        ok, reason = CuteDslB12xFusedMoE.can_implement(quant_algo)
    assert not ok
    assert reason is not None and "NVFP4" in reason


def test_can_implement_rejects_swiglu_gptoss_style():
    with patch(f"{_FUSED_MOE_MODULE}.get_sm_version", return_value=120):
        ok, reason = CuteDslB12xFusedMoE.can_implement(QuantAlgo.NVFP4, swiglu_gptoss_style=True)
    assert not ok
    assert reason is not None and "swiglu_gptoss_style" in reason


@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
def test_can_implement_rejects_unsupported_activation_dtype(dtype):
    with patch(f"{_FUSED_MOE_MODULE}.get_sm_version", return_value=120):
        ok, reason = CuteDslB12xFusedMoE.can_implement(QuantAlgo.NVFP4, dtype_activation=dtype)
    assert not ok
    assert reason is not None


def test_get_moe_cls_cutlass_path_never_auto_promotes():
    """Explicit ``moe_backend=CUTLASS`` always returns ``CutlassFusedMoE`` —
    no silent override to the b12x backend even on eligible hardware. b12x
    is opted into via ``moe_backend=CUTEDSL``."""
    cfg = ModelConfig()
    cfg.moe_backend = "CUTLASS"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=120):
        cls = get_moe_cls(cfg)
    assert cls is CutlassFusedMoE


def test_get_moe_cls_cutedsl_falls_back_to_cutlass_on_unsupported_quant():
    """CUTEDSL + non-(fp8_block_scales|nvfp4) → warn + fall back to CutlassFusedMoE."""
    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=120):
        cls = get_moe_cls(cfg)
    assert cls is CutlassFusedMoE


def test_get_moe_cls_cutedsl_falls_back_to_cutlass_on_missing_quant():
    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = None
    with patch("tensorrt_llm._utils.get_sm_version", return_value=120):
        cls = get_moe_cls(cfg)
    assert cls is CutlassFusedMoE


def test_get_moe_cls_cutedsl_returns_plain_cutedsl_on_unsupported_sm():
    """CUTEDSL + NVFP4 + non-SM120/121 → plain CuteDslFusedMoE (the SM100/103
    cuteDSL backend); the b12x branch is bypassed."""
    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=100):
        cls = get_moe_cls(cfg)
    assert cls is CuteDslFusedMoE


def test_get_moe_cls_cutedsl_returns_cutlass_for_w4a16_nvfp4_on_unsupported_sm():
    """CUTEDSL + W4A16_NVFP4 + non-SM120/121 → CutlassFusedMoE."""
    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=100):
        cls = get_moe_cls(cfg)
    assert cls is CutlassFusedMoE


@pytest.mark.parametrize("sm_version", sorted(CuteDslB12xFusedMoE._SUPPORTED_SM_VERSIONS))
def test_get_moe_cls_cutedsl_selects_b12x_on_supported_sm(sm_version):
    """CUTEDSL + NVFP4 + SM120/121 + flashinfer importable → CuteDslB12xFusedMoE."""
    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=sm_version):
        cls = get_moe_cls(cfg)
    assert cls is CuteDslB12xFusedMoE


@pytest.mark.parametrize("sm_version", sorted(CuteDslB12xFusedMoE._SUPPORTED_SM_VERSIONS))
def test_get_moe_cls_cutedsl_selects_b12x_for_w4a16_nvfp4_on_supported_sm(sm_version):
    """CUTEDSL + W4A16_NVFP4 + SM120/121 + flashinfer importable → CuteDslB12xFusedMoE."""
    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=sm_version):
        cls = get_moe_cls(cfg)
    assert cls is CuteDslB12xFusedMoE


@pytest.mark.parametrize("sm_version", sorted(CuteDslB12xFusedMoE._SUPPORTED_SM_VERSIONS))
def test_get_moe_cls_cutedsl_selects_b12x_for_layer_w4a16_nvfp4_on_supported_sm(sm_version):
    """MIXED_PRECISION per-layer W4A16 must select the same backend that the
    layer will use after apply_layerwise_quant_config().
    """
    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = QuantConfig()
    cfg.quant_config_dict = {
        "model.layers.0.mlp.experts": QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16),
    }
    with patch("tensorrt_llm._utils.get_sm_version", return_value=sm_version):
        cls = get_moe_cls(cfg, layer_idx=0)
    assert cls is CuteDslB12xFusedMoE


def test_get_moe_cls_cutedsl_falls_back_to_plain_cutedsl_when_flashinfer_missing(monkeypatch):
    """CUTEDSL + NVFP4 + SM120/121 + flashinfer NOT importable → CuteDslFusedMoE."""
    import builtins

    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)

    real_import = builtins.__import__

    def _raise_on_flashinfer(name, *args, **kwargs):
        if name == "flashinfer":
            raise ImportError("flashinfer not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise_on_flashinfer)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=120):
        cls = get_moe_cls(cfg)
    assert cls is CuteDslFusedMoE


# --------------------------------------------------------------------------
# Hybrid CUTLASS-prefill / b12x-decode dispatch predicate tests
#
# ``_route_to_cutlass`` is a pure shape predicate on its input ``x``; we test
# it via a stub that holds the class constant, sidestepping the full
# CutlassFusedMoE constructor (which needs a routing method, real model
# config, etc.).
# --------------------------------------------------------------------------


class _RoutePredicateStub:
    """Minimal carrier for ``_PREFILL_VIA_CUTLASS_THRESHOLD`` so we can call
    the unbound ``_route_to_cutlass`` without instantiating the whole MoE
    backend."""

    _PREFILL_VIA_CUTLASS_THRESHOLD = CuteDslB12xFusedMoE._PREFILL_VIA_CUTLASS_THRESHOLD

    _route_to_cutlass = CuteDslB12xFusedMoE._route_to_cutlass


def test_dispatch_routes_prefill_shape_via_cutlass():
    stub = _RoutePredicateStub()
    x = torch.empty(_RoutePredicateStub._PREFILL_VIA_CUTLASS_THRESHOLD, 1024)
    assert stub._route_to_cutlass(x) is True


def test_dispatch_just_below_threshold_takes_b12x():
    stub = _RoutePredicateStub()
    x = torch.empty(_RoutePredicateStub._PREFILL_VIA_CUTLASS_THRESHOLD - 1, 1024)
    assert stub._route_to_cutlass(x) is False


def test_dispatch_decode_shape_takes_b12x():
    stub = _RoutePredicateStub()
    x = torch.empty(1, 1024)
    assert stub._route_to_cutlass(x) is False


def test_w4a16_nvfp4_prefill_quantize_input_stays_on_b12x():
    moe = object.__new__(CuteDslB12xFusedMoE)
    moe.quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    x = torch.empty(CuteDslB12xFusedMoE._PREFILL_VIA_CUTLASS_THRESHOLD, 1024)

    with patch.object(
        CutlassFusedMoE,
        "quantize_input",
        side_effect=AssertionError("W4A16_NVFP4 prefill must not route through CUTLASS"),
    ):
        out, out_sf = CuteDslB12xFusedMoE.quantize_input(moe, x)

    assert out is x
    assert out_sf is None


def test_w4a16_nvfp4_post_load_uses_modelopt_scale_contract(monkeypatch):
    class _RoutingMethod:
        experts_per_token = 4

    class _FakeB12xWrapper:
        calls = []

        def __init__(self, **kwargs):
            self._moe_output = None
            self.calls.append(kwargs)

    def _convert_sf_to_mma_layout(scales, *, m, k, num_groups):
        return scales

    flashinfer = types.ModuleType("flashinfer")
    flashinfer.B12xMoEWrapper = _FakeB12xWrapper
    cute_dsl = types.ModuleType("flashinfer.cute_dsl")
    utils = types.ModuleType("flashinfer.cute_dsl.utils")
    utils.convert_sf_to_mma_layout = _convert_sf_to_mma_layout
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.cute_dsl", cute_dsl)
    monkeypatch.setitem(sys.modules, "flashinfer.cute_dsl.utils", utils)

    num_experts = 2
    hidden_size = 128
    logical_intermediate_size = 1856
    padded_intermediate_size = 1920
    module = torch.nn.Module()
    module.num_experts = num_experts
    module.hidden_size = hidden_size
    module.intermediate_size_per_partition = logical_intermediate_size
    module.moe_max_num_tokens = 8
    module.routing_method = _RoutingMethod()
    module.activation_type = ActivationType.Swiglu
    module.quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    module.w3_w1_weight = torch.empty(
        num_experts,
        2 * padded_intermediate_size,
        hidden_size // 16,
        dtype=torch.int64,
    )
    module.w2_weight = torch.empty(
        num_experts,
        hidden_size,
        padded_intermediate_size // 16,
        dtype=torch.int64,
    )
    w3_w1_weight_scale = torch.ones(
        num_experts,
        2 * padded_intermediate_size,
        hidden_size // 16,
        dtype=torch.float8_e4m3fn,
    )
    w2_weight_scale = torch.ones(
        num_experts,
        hidden_size,
        padded_intermediate_size // 16,
        dtype=torch.float8_e4m3fn,
    )
    module.w3_w1_weight_scale = w3_w1_weight_scale.clone()
    module.w2_weight_scale = w2_weight_scale.clone()
    module.fc31_alpha = torch.tensor([0.25, 0.5], dtype=torch.float32)
    module.fc2_alpha = torch.tensor([0.125, 0.25], dtype=torch.float32)
    module.fc31_input_scale = torch.tensor(2.0, dtype=torch.float32)
    module.fc2_input_scale = torch.tensor(4.0, dtype=torch.float32)

    with patch.object(NVFP4CutlassFusedMoEMethod, "post_load_weights", return_value=None):
        NVFP4CuteDslB12xFusedMoEMethod().post_load_weights(module)

    assert _FakeB12xWrapper.calls
    wrapper_kwargs = _FakeB12xWrapper.calls[0]
    assert wrapper_kwargs.get("quant_mode") == "w4a16", wrapper_kwargs
    assert wrapper_kwargs["intermediate_size"] == padded_intermediate_size
    assert module._b12x_weights["fc2_input_scale"] is None
    assert torch.equal(
        module._b12x_weights["w1_weight_sf"].float(),
        w3_w1_weight_scale.float(),
    )
    assert torch.equal(
        module._b12x_weights["w2_weight_sf"].float(),
        w2_weight_scale.float(),
    )
    assert torch.allclose(
        module._b12x_weights["w1_alpha"],
        torch.tensor([0.5, 1.0], dtype=torch.float32),
    )
    assert torch.allclose(
        module._b12x_weights["w2_alpha"],
        torch.tensor([0.5, 1.0], dtype=torch.float32),
    )


def test_dispatch_rejects_non_tensor():
    """Non-tensor inputs (e.g. Fp4QuantizedTensor) stay on the b12x path
    so the existing ValueError surfaces in quantize_input."""
    stub = _RoutePredicateStub()
    assert stub._route_to_cutlass(object()) is False
