# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_qwen3vl
from tensorrt_llm._torch.models.modeling_qwen3_5 import _Qwen3_5VLModel
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VisionModelBase
from tensorrt_llm._torch.models.modeling_utils import rename_weights_with_regex


class _TinyFusedQkv(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(6, 2))


class _TinyVisualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_heads=1, num_attention_heads=None)
        self.patch_embed = nn.Module()
        self.patch_embed.weight = nn.Parameter(torch.zeros(2, 2))
        self.attn = nn.Module()
        self.attn.qkv_proj = _TinyFusedQkv()
        self.attn.o_proj = nn.Module()
        self.attn.o_proj.weight = nn.Parameter(torch.zeros(2, 2))
        self.projector = nn.Module()
        self.projector.weight = nn.Parameter(torch.zeros(3, 2))


def _vision_encoder() -> Qwen3VisionModelBase:
    encoder = Qwen3VisionModelBase.__new__(Qwen3VisionModelBase)
    nn.Module.__init__(encoder)
    encoder.visual = _TinyVisualModel()
    encoder._vision_weights_loaded = False
    return encoder


def _vision_weights() -> dict[str, torch.Tensor]:
    return {
        "model.visual.patch_embed.weight": torch.arange(4).reshape(2, 2),
        "model.visual.attn.qkv.weight": torch.arange(12).reshape(6, 2),
        "model.visual.attn.proj.weight": torch.arange(4).reshape(2, 2) + 20,
        "model.visual.projector.weight": torch.arange(6).reshape(3, 2) + 30,
    }


def _copy_tiny_visual_weights(model, weights, params_map):
    canonical_weights = rename_weights_with_regex(params_map, weights)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name == "attn.qkv_proj.weight":
                parameter.copy_(
                    torch.cat(
                        [
                            canonical_weights[f"attn.{projection}.weight"]
                            for projection in ("q_proj", "k_proj", "v_proj")
                        ]
                    )
                )
            else:
                parameter.copy_(canonical_weights[name])


def test_partial_text_group_does_not_walk_vision_model(monkeypatch):
    encoder = _vision_encoder()
    loader = mock.Mock()
    monkeypatch.setattr(modeling_qwen3vl, "_load_weights_impl", loader)

    encoder.load_weights(
        {"model.language_model.layers.0.input_layernorm.weight": torch.ones(2)},
        allow_partial_loading=True,
    )

    loader.assert_not_called()
    assert not encoder._vision_weights_loaded


def test_incomplete_partial_vision_group_fails_before_mutation(monkeypatch):
    encoder = _vision_encoder()
    loader = mock.Mock()
    monkeypatch.setattr(modeling_qwen3vl, "_load_weights_impl", loader)
    original_parameters = {
        name: parameter.detach().clone() for name, parameter in encoder.visual.named_parameters()
    }
    incomplete_weights = _vision_weights()
    del incomplete_weights["model.visual.projector.weight"]

    with pytest.raises(ValueError, match="missing 1 parameters"):
        encoder.load_weights(incomplete_weights, allow_partial_loading=True)

    loader.assert_not_called()
    for name, parameter in encoder.visual.named_parameters():
        torch.testing.assert_close(parameter, original_parameters[name])
    assert not encoder._vision_weights_loaded


def test_complete_partial_vision_group_loads_once(monkeypatch):
    encoder = _vision_encoder()
    loader = mock.Mock(side_effect=_copy_tiny_visual_weights)
    monkeypatch.setattr(modeling_qwen3vl, "_load_weights_impl", loader)
    weights = _vision_weights()

    encoder.load_weights(weights, allow_partial_loading=True)

    loader.assert_called_once()
    assert encoder._vision_weights_loaded
    with pytest.raises(RuntimeError, match="already loaded"):
        encoder.load_weights(weights, allow_partial_loading=True)
    loader.assert_called_once()


def test_full_and_partial_vision_loading_produce_the_same_state(monkeypatch):
    monkeypatch.setattr(modeling_qwen3vl, "_load_weights_impl", _copy_tiny_visual_weights)
    weights = _vision_weights()
    full_encoder = _vision_encoder()
    partial_encoder = _vision_encoder()

    full_encoder.load_weights(weights)
    partial_encoder.load_weights(weights, allow_partial_loading=True)

    full_parameters = dict(full_encoder.visual.named_parameters())
    partial_parameters = dict(partial_encoder.visual.named_parameters())
    assert partial_parameters.keys() == full_parameters.keys()
    for name in full_parameters:
        torch.testing.assert_close(partial_parameters[name], full_parameters[name])


def test_full_vision_load_keeps_strict_empty_checkpoint_behavior(monkeypatch):
    encoder = _vision_encoder()
    loader = mock.Mock()
    monkeypatch.setattr(modeling_qwen3vl, "_load_weights_impl", loader)

    encoder.load_weights({}, allow_partial_loading=False)

    loader.assert_called_once_with(
        encoder.visual,
        {},
        params_map=Qwen3VisionModelBase._VISION_PARAMS_MAP,
    )


class _RecordingVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def load_weights(self, weights, allow_partial_loading=False):
        self.calls.append((weights, allow_partial_loading))


class _RecordingLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = object()
        self.calls = []

    def load_weights(
        self,
        weights,
        weight_mapper,
        params_map=None,
        allow_partial_loading=False,
    ):
        self.calls.append((weights, weight_mapper, params_map, allow_partial_loading))


class _RecordingMapper:
    def __init__(self):
        self.init_calls = []
        self._model = None

    @property
    def model(self):
        return self._model

    def init_model_and_config(self, model, model_config):
        self.init_calls.append((model, model_config))
        self._model = model


def test_qwen3_5_vlm_routes_atomic_vision_group_without_text_walk():
    model = _Qwen3_5VLModel.__new__(_Qwen3_5VLModel)
    nn.Module.__init__(model)
    model.mm_encoder = _RecordingVisionEncoder()
    model.llm = _RecordingLanguageModel()
    mapper = _RecordingMapper()
    weights = {"model.visual.patch_embed.weight": torch.ones(2, 2)}

    model.load_weights(weights, weight_mapper=mapper, allow_partial_loading=True)

    assert model.mm_encoder.calls == [(weights, True)]
    assert not mapper.init_calls
    assert not model.llm.calls


def test_qwen3_5_vlm_reuses_session_mapper_across_text_groups():
    model = _Qwen3_5VLModel.__new__(_Qwen3_5VLModel)
    nn.Module.__init__(model)
    model.mm_encoder = _RecordingVisionEncoder()
    model.llm = _RecordingLanguageModel()
    mapper = _RecordingMapper()
    mapper._model = model
    weights = {"model.language_model.norm.weight": torch.ones(2)}

    model.load_weights(weights, weight_mapper=mapper, allow_partial_loading=True)
    second_weights = {"lm_head.weight": torch.ones(2, 2)}
    model.load_weights(second_weights, weight_mapper=mapper, allow_partial_loading=True)

    assert model.mm_encoder.calls == [(weights, True), (second_weights, True)]
    assert mapper.init_calls == [(model.llm, model.llm.model_config)]
    assert len(model.llm.calls) == 2
    loaded_weights, loaded_mapper, _, allow_partial_loading = model.llm.calls[0]
    assert loaded_weights.keys() == weights.keys()
    assert (
        loaded_weights["model.language_model.norm.weight"]
        is weights["model.language_model.norm.weight"]
    )
    assert loaded_mapper is mapper
    assert allow_partial_loading
    assert model.llm.calls[1][1] is mapper
