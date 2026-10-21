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
"""Tests for how `create_input_processor` resolves a model's input processor."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import tensorrt_llm._torch.model_config as model_config_mod
import tensorrt_llm._torch.models as models_mod
from tensorrt_llm.inputs.registry import (
    INPUT_PROCESSOR_REGISTRY,
    DefaultInputProcessor,
    create_input_processor,
)

pytestmark = pytest.mark.cpu_only


class _FakeVisionLanguageModel:
    """Stand-in model class that the fake architecture lookup resolves to."""


class _RecordingInputProcessor:
    """Registered input processor that records its constructor arguments."""

    def __init__(self, model_path, config, tokenizer, **kwargs) -> None:
        self.model_path = model_path
        self.config = config
        self.tokenizer = tokenizer
        self.kwargs = kwargs


@pytest.fixture
def config_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Register a fake model processor and record every Hugging Face config load."""
    loads: list[str] = []
    pretrained_config = SimpleNamespace(architectures=["FakeForConditionalGeneration"])

    def _from_pretrained(_cls, model_path, trust_remote_code=False, **_kwargs):
        loads.append(model_path)
        return SimpleNamespace(pretrained_config=pretrained_config)

    monkeypatch.setattr(
        model_config_mod.ModelConfig, "from_pretrained", classmethod(_from_pretrained)
    )
    monkeypatch.setattr(model_config_mod, "hf_remote_code_lock", nullcontext)
    monkeypatch.setattr(
        models_mod,
        "get_model_architecture",
        lambda config: (_FakeVisionLanguageModel, config.architectures[0]),
    )
    monkeypatch.setitem(
        INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type,
        _FakeVisionLanguageModel,
        _RecordingInputProcessor,
    )
    return loads


@pytest.mark.parametrize("checkpoint_format", ["HF", "MX"])
def test_create_input_processor_uses_registered_processor_for_hf_format_checkpoints(
    config_loads: list[str],
    checkpoint_format: str,
) -> None:
    """HF and MX checkpoints both resolve the model's registered input processor.

    MX loads Hugging Face-format checkpoints, so a vision-language model loaded
    through MX must get the same processor as the plain Hugging Face load.
    """
    tokenizer = object()

    processor = create_input_processor("/ckpt", tokenizer, checkpoint_format)

    assert isinstance(processor, _RecordingInputProcessor)
    assert processor.model_path == "/ckpt"
    assert processor.tokenizer is tokenizer
    assert processor.config.architectures == ["FakeForConditionalGeneration"]
    assert config_loads == ["/ckpt"]


@pytest.mark.parametrize("checkpoint_format", [None, "custom"])
def test_create_input_processor_skips_hf_config_for_other_formats(
    config_loads: list[str],
    checkpoint_format: str | None,
) -> None:
    """Formats without a Hugging Face config fall back to `DefaultInputProcessor`."""
    tokenizer = object()

    processor = create_input_processor("/ckpt", tokenizer, checkpoint_format)

    assert type(processor) is DefaultInputProcessor
    assert processor.tokenizer is tokenizer
    assert config_loads == []
