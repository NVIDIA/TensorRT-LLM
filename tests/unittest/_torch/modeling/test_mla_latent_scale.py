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


def test_scales_bring_the_latent_to_unit_rms(monkeypatch):
    monkeypatch.setenv(_MLA_LATENT_SCALE_ENV, "auto")
    weights = _checkpoint([0.004, 0.03, 0.2, 0.8])
    scales = _mla_latent_scales(weights, _NVFP4_KV)
    assert scales == {
        _PREFIX.format(0): 256.0,
        _PREFIX.format(1): 32.0,
        _PREFIX.format(2): 4.0,
        _PREFIX.format(3): 1.0,
    }
    for name, scale in scales.items():
        gamma = weights[f"{name}.kv_a_layernorm.weight"].float() * scale
        assert 2**-0.5 <= gamma.pow(2).mean().sqrt() <= 2**0.5


def test_default_scales_nvfp4_kv_caches_only(monkeypatch):
    monkeypatch.delenv(_MLA_LATENT_SCALE_ENV, raising=False)
    weights = _checkpoint([0.004])
    assert _mla_latent_scales(weights, _NVFP4_KV) == {_PREFIX.format(0): 256.0}
    assert _mla_latent_scales(weights, _FP8_KV) == {}
    assert _mla_latent_scales(weights, _UNQUANTIZED_KV) == {}


def test_explicit_target_scales_every_quantized_kv_cache(monkeypatch):
    monkeypatch.setenv(_MLA_LATENT_SCALE_ENV, "0.25")
    weights = _checkpoint([0.004])
    assert _mla_latent_scales(weights, _NVFP4_KV) == {_PREFIX.format(0): 64.0}
    assert _mla_latent_scales(weights, _FP8_KV) == {_PREFIX.format(0): 64.0}
    assert _mla_latent_scales(weights, _UNQUANTIZED_KV) == {}


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "on"])
def test_rejects_invalid_target(monkeypatch, value):
    monkeypatch.setenv(_MLA_LATENT_SCALE_ENV, value)
    with pytest.raises(ValueError, match=_MLA_LATENT_SCALE_ENV):
        _mla_latent_scales(_checkpoint([0.004]), _NVFP4_KV)


def test_off_leaves_weights_alone(monkeypatch):
    monkeypatch.setenv(_MLA_LATENT_SCALE_ENV, "off")
    assert _mla_latent_scales(_checkpoint([0.004]), _NVFP4_KV) == {}


def test_quantized_projection_is_skipped(monkeypatch):
    monkeypatch.setenv(_MLA_LATENT_SCALE_ENV, "auto")
    weights = _checkpoint([0.004], projection_dtype=torch.float8_e4m3fn)
    assert _mla_latent_scales(weights, _NVFP4_KV) == {}


def test_scaled_weights_keep_the_projection_output(monkeypatch):
    """kv_b_proj(layernorm(x)) is bit-identical in BF16 after the power-of-two rescaling."""
    monkeypatch.setenv(_MLA_LATENT_SCALE_ENV, "auto")
    weights = _checkpoint([0.004])
    name = _PREFIX.format(0)
    scale = _mla_latent_scales(weights, _NVFP4_KV)[name]
    gamma = weights[f"{name}.kv_a_layernorm.weight"]
    projection = weights[f"{name}.kv_b_proj.weight"]
    x = torch.randn(8, _LATENT_DIM, generator=torch.Generator().manual_seed(1))
    normed = (x / x.pow(2).mean(-1, keepdim=True).sqrt()).to(torch.bfloat16)

    latent = normed * gamma
    scaled_latent = normed * (gamma * scale)
    assert torch.equal(scaled_latent.float(), latent.float() * scale)
    reference = latent.float() @ projection.float().T
    rescaled = scaled_latent.float() @ (projection / scale).float().T
    assert torch.equal(rescaled, reference)
