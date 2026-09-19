# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only unit tests for the Xing4.0 model and its mHC weight loader.

The full :class:`Xing4_0Model` constructor requires CUDA (it allocates
``torch.cuda.Stream`` objects and heavy DeepSeek-V3 sub-modules), so the
forward test builds the model with a stubbed decoder layer and a mocked
CUDA stream. The weight-loader tests target ``_load_hc`` directly with a
tiny module tree, covering both supported mHC checkpoint layouts.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.models.modeling_xing4_0 as modeling_xing4_0
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.modeling_xing4_0 import Xing4_0Model, Xing4_0WeightLoader
from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC

HIDDEN_SIZE = 4
N_STREAMS = 2
NUM_LAYERS = 2
MIX_HC = (2 + N_STREAMS) * N_STREAMS
HC_DIM = N_STREAMS * HIDDEN_SIZE


class _StubDecoderLayer(nn.Module):
    """Decoder-layer stand-in that records its residual-stream input.

    Returns the input scaled by ``stream_index + 1`` so the model-level
    stream mean reduction is observable in the final output.
    """

    def __init__(self, model_config, layer_idx, aux_stream_dict):
        super().__init__()
        self.layer_idx = layer_idx
        self.seen_shapes = []
        self.seen_inputs = []

    def forward(
        self,
        position_ids,
        hidden_states,
        attn_metadata,
        residual=None,
        spec_metadata=None,
        **kwargs,
    ):
        self.seen_shapes.append(tuple(hidden_states.shape))
        self.seen_inputs.append(hidden_states.detach().clone())
        stream_idx = torch.arange(hidden_states.shape[-2], dtype=hidden_states.dtype).view(1, -1, 1)
        return hidden_states * (stream_idx + 1.0), None


class _HCHost(nn.Module):
    """Tiny module tree exposing the mHC stems a decoder layer owns."""

    def __init__(self):
        super().__init__()
        self.attn_hc = mHC(mult=N_STREAMS, hidden_size=HIDDEN_SIZE, sinkhorn_iters=1)
        self.ffn_hc = mHC(mult=N_STREAMS, hidden_size=HIDDEN_SIZE, sinkhorn_iters=1)


class _FakeModel(nn.Module):
    """Duck-typed model satisfying the DeepSeek-V3 loader constructor."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.model_config = None
        self.layer = _HCHost()


def _make_model_config() -> ModelConfig:
    """Build a minimal CPU-friendly ModelConfig for the Xing4.0 model."""
    pretrained = SimpleNamespace(
        vocab_size=16,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        rms_norm_eps=1e-6,
        torch_dtype=torch.float32,
        hc_mult=N_STREAMS,
    )
    return ModelConfig(pretrained_config=pretrained)


def _build_stubbed_model() -> Xing4_0Model:
    """Build an Xing4_0Model whose decoder layers are recording stubs."""
    model_config = _make_model_config()
    with (
        mock.patch("torch.cuda.Stream", return_value=mock.MagicMock()),
        mock.patch.object(modeling_xing4_0, "Xing4_0DecoderLayer", _StubDecoderLayer),
    ):
        model = Xing4_0Model(model_config)
    model.norm = nn.Identity()
    return model


def test_forward_expands_streams_and_reduces_mean():
    """Forward duplicates embeds into streams and averages them at the end."""
    torch.manual_seed(0)
    tokens = 3
    model = _build_stubbed_model()
    embeds = torch.randn(tokens, HIDDEN_SIZE)

    output = model(attn_metadata=mock.MagicMock(), inputs_embeds=embeds)

    first_layer, second_layer = list(model.layers)

    assert first_layer.seen_shapes == [(tokens, N_STREAMS, HIDDEN_SIZE)]
    expected = embeds.unsqueeze(1).expand(-1, N_STREAMS, -1)
    assert torch.equal(first_layer.seen_inputs[0], expected)

    assert second_layer.seen_shapes == [(tokens, N_STREAMS, HIDDEN_SIZE)]
    assert torch.equal(second_layer.seen_inputs[0][:, 0], embeds)
    assert torch.equal(second_layer.seen_inputs[0][:, 1], 2.0 * embeds)

    assert output.shape == (tokens, HIDDEN_SIZE)
    assert torch.allclose(output, 2.5 * embeds)


def _rand_hc_params(seed: int):
    """Generate synthetic fn/base/scale tensors for one mHC module."""
    gen = torch.Generator().manual_seed(seed)
    fn = torch.randn(MIX_HC, HC_DIM, generator=gen)
    base = torch.randn(MIX_HC, generator=gen)
    scale = torch.randn(3, generator=gen)
    return fn, base, scale


def test_load_hc_fn_base_scale_layout():
    """_load_hc consumes the hc_fn/hc_base/hc_scale checkpoint layout."""
    loader = Xing4_0WeightLoader(_FakeModel())
    model = loader.model
    fn, base, scale = _rand_hc_params(seed=1)
    weights = ConsumableWeightsDict(
        {
            "layer.attn_hc.hc_fn": fn.clone(),
            "layer.attn_hc.hc_base": base.clone(),
            "layer.attn_hc.hc_scale": scale.clone(),
            "layer.ffn_hc.hc_fn": fn.clone(),
            "layer.ffn_hc.hc_base": base.clone(),
            "layer.ffn_hc.hc_scale": scale.clone(),
        }
    )

    loader._load_hc(weights)

    for hc in (model.layer.attn_hc, model.layer.ffn_hc):
        assert torch.equal(hc.fn, fn)
        assert torch.equal(hc.base, base)
        assert torch.equal(hc.scale, scale)
    assert list(weights.keys()) == []


def test_load_hc_mapping_bias_alpha_layout():
    """_load_hc consumes the mapping_weight/bias/alpha_* checkpoint layout.

    Both layouts can coexist in one checkpoint, resolved per module.
    """
    loader = Xing4_0WeightLoader(_FakeModel())
    model = loader.model
    mapping = torch.randn(MIX_HC, HC_DIM)
    bias = torch.randn(MIX_HC)
    alphas = (0.1, 0.2, 0.3)
    fn, base, scale = _rand_hc_params(seed=2)
    weights = ConsumableWeightsDict(
        {
            "layer.attn_hc.mapping_weight": mapping.clone(),
            "layer.attn_hc.bias": bias.clone(),
            "layer.attn_hc.alpha_pre": torch.tensor(alphas[0]),
            "layer.attn_hc.alpha_post": torch.tensor(alphas[1]),
            "layer.attn_hc.alpha_res": torch.tensor(alphas[2]),
            "layer.ffn_hc.hc_fn": fn.clone(),
            "layer.ffn_hc.hc_base": base.clone(),
            "layer.ffn_hc.hc_scale": scale.clone(),
        }
    )

    loader._load_hc(weights)

    attn_hc = model.layer.attn_hc
    assert torch.equal(attn_hc.fn, mapping)
    assert torch.equal(attn_hc.base, bias)
    assert torch.allclose(attn_hc.scale, torch.tensor(alphas))

    ffn_hc = model.layer.ffn_hc
    assert torch.equal(ffn_hc.fn, fn)
    assert torch.equal(ffn_hc.base, base)
    assert torch.equal(ffn_hc.scale, scale)
    assert list(weights.keys()) == []


def test_load_hc_missing_weights_raises():
    """_load_hc raises when a module has neither supported layout."""
    loader = Xing4_0WeightLoader(_FakeModel())
    fn, _, _ = _rand_hc_params(seed=3)
    weights = ConsumableWeightsDict({"layer.attn_hc.hc_fn": fn})

    with pytest.raises(ValueError, match="no loadable weights"):
        loader._load_hc(weights)
