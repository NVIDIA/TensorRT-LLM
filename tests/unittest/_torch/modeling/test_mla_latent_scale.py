# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA latent scale: power-of-two rescaling of the latent that a quantized KV cache stores."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_deepseekv3 import _MLA_LATENT_SCALE_ENV, _mla_latent_scales
from tensorrt_llm.quantization.mode import QuantAlgo, QuantMode

pytestmark = pytest.mark.cpu_only

_LATENT_DIM = 512
_PREFIX = "model.layers.{}.self_attn"
_NVFP4_KV = QuantMode.from_quant_algo(kv_cache_quant_algo=QuantAlgo.NVFP4)
_FP8_KV = QuantMode.from_quant_algo(kv_cache_quant_algo=QuantAlgo.FP8)
_UNQUANTIZED_KV = QuantMode.from_quant_algo()


def _checkpoint(layernorm_rms, projection_dtype=torch.bfloat16):
    """Layernorm and projection weights of one attention module per requested RMS."""
    generator = torch.Generator().manual_seed(0)
    weights = {}
    for layer, rms in enumerate(layernorm_rms):
        gamma = torch.rand(_LATENT_DIM, generator=generator) + 0.5
        gamma = gamma * (rms / gamma.pow(2).mean().sqrt())
        weights[f"{_PREFIX.format(layer)}.kv_a_layernorm.weight"] = gamma.to(torch.bfloat16)
        projection = torch.randn(64, _LATENT_DIM, generator=generator)
        weights[f"{_PREFIX.format(layer)}.kv_b_proj.weight"] = projection.to(projection_dtype)
    return weights


def _scales(weights, kv=_NVFP4_KV, dsa=True, dequantized=False):
    """Scales of a DSA model with an NVFP4 KV cache unless a case says otherwise."""
    return _mla_latent_scales(weights, kv, uses_dsa=dsa, dequantizes_kv_b_proj=dequantized)


def test_scales_bring_the_latent_to_unit_rms(monkeypatch):
    monkeypatch.setenv(_MLA_LATENT_SCALE_ENV, "auto")
    weights = _checkpoint([0.004, 0.03, 0.2, 0.8])
    scales = _scales(weights)
    assert scales == {
        _PREFIX.format(0): 256.0,
        _PREFIX.format(1): 32.0,
        _PREFIX.format(2): 4.0,
        _PREFIX.format(3): 1.0,
    }
    for name, scale in scales.items():
        gamma = weights[f"{name}.kv_a_layernorm.weight"].float() * scale
        assert 2**-0.5 <= gamma.pow(2).mean().sqrt() <= 2**0.5


def test_default_scales_nvfp4_kv_caches_of_dsa_models_only(monkeypatch):
    monkeypatch.delenv(_MLA_LATENT_SCALE_ENV, raising=False)
    weights = _checkpoint([0.004])
    assert _scales(weights) == {_PREFIX.format(0): 256.0}
    assert _scales(weights, dsa=False) == {}
    assert _scales(weights, kv=_FP8_KV) == {}
    assert _scales(weights, kv=_UNQUANTIZED_KV) == {}
    monkeypatch.setenv(_MLA_LATENT_SCALE_ENV, "off")
    assert _scales(weights) == {}


def test_explicit_target_scales_every_quantized_kv_cache(monkeypatch):
    monkeypatch.setenv(_MLA_LATENT_SCALE_ENV, "0.25")
    weights = _checkpoint([0.004])
    assert _scales(weights) == {_PREFIX.format(0): 64.0}
    assert _scales(weights, dsa=False) == {_PREFIX.format(0): 64.0}
    assert _scales(weights, kv=_FP8_KV, dsa=False) == {_PREFIX.format(0): 64.0}
    assert _scales(weights, kv=_UNQUANTIZED_KV) == {}


def test_quantized_projection_is_scaled_only_when_the_loader_dequantizes_it(monkeypatch):
    monkeypatch.setenv(_MLA_LATENT_SCALE_ENV, "auto")
    weights = _checkpoint([0.004], projection_dtype=torch.float8_e4m3fn)
    assert _scales(weights) == {}
    assert _scales(weights, dequantized=True) == {_PREFIX.format(0): 256.0}
