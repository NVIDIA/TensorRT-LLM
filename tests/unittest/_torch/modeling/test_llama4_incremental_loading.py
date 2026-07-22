# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_llama
from tensorrt_llm._torch.models.modeling_llama import Llama4VisionEncoder


class _TinyVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        del config
        self.weight = nn.Parameter(torch.zeros(2))


class _TinyProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        del config
        self.weight = nn.Parameter(torch.zeros(3))


@pytest.fixture
def vision_encoder(monkeypatch):
    monkeypatch.setattr(modeling_llama, "Llama4VisionModel", _TinyVisionModel)
    monkeypatch.setattr(modeling_llama, "Llama4MultiModalProjector", _TinyProjector)

    encoder = Llama4VisionEncoder.__new__(Llama4VisionEncoder)
    nn.Module.__init__(encoder)
    encoder.pretrained_config = SimpleNamespace(
        vision_config=object(),
        _name_or_path="unused-checkpoint",
    )
    encoder.device = "cpu"
    return encoder


def test_partial_text_batch_does_not_open_vision_checkpoint(vision_encoder, monkeypatch):
    fallback = mock.Mock()
    monkeypatch.setattr(modeling_llama, "_load_checkpoint_into_module", fallback)

    vision_encoder.load_weights(
        {"language_model.model.layers.0.input_layernorm.weight": torch.ones(2)},
        allow_partial_loading=True,
    )

    fallback.assert_not_called()
    assert not hasattr(vision_encoder, "vision_model")
    assert not hasattr(vision_encoder, "mm_projector")


def test_partial_vision_batch_must_be_complete_before_mutation(vision_encoder, monkeypatch):
    fallback = mock.Mock()
    monkeypatch.setattr(modeling_llama, "_load_checkpoint_into_module", fallback)

    with pytest.raises(ValueError, match="missing 1 parameters"):
        vision_encoder.load_weights(
            {"vision_model.weight": torch.ones(2)},
            allow_partial_loading=True,
        )

    fallback.assert_not_called()
    assert not hasattr(vision_encoder, "vision_model")
    assert not hasattr(vision_encoder, "mm_projector")
    assert not getattr(vision_encoder, "_vision_weights_loaded", False)


def test_complete_partial_vision_group_loads_exactly_once(vision_encoder, monkeypatch):
    fallback = mock.Mock()
    monkeypatch.setattr(modeling_llama, "_load_checkpoint_into_module", fallback)
    weights = {
        "vision_model.weight": torch.tensor([1.0, 2.0]),
        "multi_modal_projector.weight": torch.tensor([3.0, 4.0, 5.0]),
    }

    vision_encoder.load_weights(weights, allow_partial_loading=True)

    fallback.assert_not_called()
    torch.testing.assert_close(vision_encoder.vision_model.weight, weights["vision_model.weight"])
    torch.testing.assert_close(
        vision_encoder.mm_projector.weight, weights["multi_modal_projector.weight"]
    )
    assert vision_encoder._vision_weights_loaded

    with pytest.raises(RuntimeError, match="already loaded"):
        vision_encoder.load_weights(weights, allow_partial_loading=True)


def test_full_load_preserves_checkpoint_fallback(vision_encoder, monkeypatch):
    fallback = mock.Mock()
    monkeypatch.setattr(modeling_llama, "_load_checkpoint_into_module", fallback)

    vision_encoder.load_weights({}, allow_partial_loading=False)

    fallback.assert_called_once()
    assert hasattr(vision_encoder, "vision_model")
    assert hasattr(vision_encoder, "mm_projector")
