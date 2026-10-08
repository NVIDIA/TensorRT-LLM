# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``SpecDecOneEngineForCausalLM._build_draft_model``: the default is the mode registry's builder, and a subclass
override is what ``__init__`` builds the drafter with."""

import pytest
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_speculative
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

pytestmark = pytest.mark.cpu_only


def test_default_build_draft_model_is_get_draft_model(monkeypatch):
    built = nn.Module()
    calls = []

    def fake_get_draft_model(model_config, draft_config, lm_head, model):
        calls.append((model_config, draft_config, lm_head, model))
        return built

    monkeypatch.setattr(modeling_speculative, "get_draft_model", fake_get_draft_model)
    shell = object.__new__(SpecDecOneEngineForCausalLM)
    nn.Module.__init__(shell)
    shell.lm_head = nn.Linear(2, 2)
    shell.model = nn.Module()
    model_config, draft_config = object(), object()

    assert shell._build_draft_model(model_config, draft_config) is built
    assert calls == [(model_config, draft_config, shell.lm_head, shell.model)]


def _minimal_decoder_init(self, model, *, config, hidden_size, vocab_size):
    """``DecoderModelForCausalLM.__init__`` reduced to what the one-engine shell reads."""
    nn.Module.__init__(self)
    self.model_config = config
    self.model = model
    self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    self.logits_processor = object()
    self.epilogue = []


def test_init_builds_the_drafter_through_an_override(monkeypatch):
    monkeypatch.setattr(DecoderModelForCausalLM, "__init__", _minimal_decoder_init)
    monkeypatch.setattr(DecoderModelForCausalLM, "__post_init__", lambda self: None)
    monkeypatch.setattr(modeling_speculative, "get_spec_worker", lambda *args, **kwargs: None)

    def registry_builder_not_called(*args, **kwargs):
        raise AssertionError("the override replaces get_draft_model")

    monkeypatch.setattr(modeling_speculative, "get_draft_model", registry_builder_not_called)
    drafter = nn.Linear(1, 1)
    calls = []

    class OwnDrafter(SpecDecOneEngineForCausalLM):
        def _build_draft_model(self, model_config, draft_config):
            calls.append((model_config, draft_config))
            return drafter

    model_config = ModelConfig(
        pretrained_config=PretrainedConfig(hidden_size=8, vocab_size=16, num_hidden_layers=2),
        spec_config=MTPDecodingConfig(max_draft_len=1),
    )
    shell = OwnDrafter(nn.Module(), model_config)

    assert calls == [(model_config, shell.draft_config)]
    assert shell.draft_model is drafter
    assert shell.epilogue == [drafter]
