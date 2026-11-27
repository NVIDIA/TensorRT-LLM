# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which HF frontend files the readers of a Mistral-Large-4 checkpoint use.

A Mistral-Large-4 checkpoint may ship, beside its Mistral-native params.json and tekken.json,
the HF frontend files tokenizer.json, tokenizer_config.json, chat_template.jinja,
processor_config.json and generation_config.json. Under ``checkpoint_format="mistral"``
the input processor and standalone renderer load mistral-common's MistralTokenizer and no HF
processor, whatever HF tokenizer files the directory ships, and the LLM still reads sampling
defaults from generation_config.json.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_common.mistral_tekken import bundled_tekken_file, write_hf_tokenizer

pytest.importorskip("mistral_common")

import torch  # noqa: E402

from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import (  # noqa: E402
    MistralConfigLoader,
)
from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer  # noqa: E402
from tensorrt_llm.inputs.registry import create_input_processor  # noqa: E402
from tensorrt_llm.llmapi.llm import BaseLLM  # noqa: E402
from tensorrt_llm.tokenizer import TransformersTokenizer, load_hf_tokenizer  # noqa: E402

pytestmark = pytest.mark.cpu_only

_HF_FRONTEND = {
    "chat_template.jinja": "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}",
    "processor_config.json": {"processor_class": "PixtralProcessor"},
    "generation_config.json": {"temperature": 1.0, "top_p": 0.95, "eos_token_id": 2},
}


@pytest.fixture
def native_dir(tmp_path) -> Path:
    """Mistral-native files and the HF frontend files, no config.json."""
    shutil.copy(bundled_tekken_file(), tmp_path / "tekken.json")
    (tmp_path / "params.json").write_text(json.dumps({"dim": 8}))
    write_hf_tokenizer(tmp_path)
    for name, content in _HF_FRONTEND.items():
        text = content if isinstance(content, str) else json.dumps(content)
        (tmp_path / name).write_text(text)
    return tmp_path


def _llm(model_dir, checkpoint_format, generation_config="auto"):
    """The attributes BaseLLM's tokenizer and generation-config loaders read."""
    return SimpleNamespace(
        args=SimpleNamespace(
            skip_tokenizer_init=False,
            tokenizer=None,
            checkpoint_format=checkpoint_format,
            model=str(model_dir),
            trust_remote_code=False,
            tokenizer_mode="auto",
            backend="pytorch",
            lora_config=None,
            generation_config=generation_config,
        ),
        _hf_model_dir=None,
    )


def test_the_native_format_tokenizes_with_mistral_common_even_beside_a_config(
    native_dir, monkeypatch
):
    # The LLM takes its tokenizer from the input processor, which the Mistral-native format
    # builds over mistral-common whatever the directory's other readers load.
    (native_dir / "config.json").write_text(json.dumps({"model_type": "mistral"}))
    assert type(load_hf_tokenizer(str(native_dir))) is TransformersTokenizer
    config = SimpleNamespace(model_type="mistral_large_4", torch_dtype=torch.bfloat16)
    monkeypatch.setattr(
        MistralConfigLoader, "load", lambda self, path: SimpleNamespace(pretrained_config=config)
    )
    processor = create_input_processor(str(native_dir), None, checkpoint_format="mistral")
    assert type(processor.tokenizer) is MistralTokenizer


@pytest.mark.parametrize("checkpoint_format", ["mistral", "HF"])
def test_every_format_reads_the_generation_config(native_dir, checkpoint_format):
    llm = _llm(native_dir, checkpoint_format)
    assert BaseLLM._try_load_generation_config(llm).top_p == 0.95
    assert BaseLLM._try_load_generation_config_explicit_values(llm) == {
        "temperature": 1.0,
        "top_p": 0.95,
        "eos_token_id": 2,
    }


def test_the_standalone_renderer_loads_no_hf_processor(native_dir, monkeypatch):
    # trtllm-serve leaves the HF processor out for the native format; a renderer that loaded
    # the checkpoint's PixtralProcessor would fingerprint the processor's chat template too.
    from tensorrt_llm.serve.render.resources import RenderResources

    config = SimpleNamespace(model_type="mistral_large_4", torch_dtype=torch.bfloat16)
    monkeypatch.setattr(
        MistralConfigLoader, "load", lambda self, path: SimpleNamespace(pretrained_config=config)
    )
    rendered = RenderResources.load(str(native_dir), checkpoint_format="mistral")
    assert rendered.processor is None
    assert type(rendered.tokenizer) is MistralTokenizer
