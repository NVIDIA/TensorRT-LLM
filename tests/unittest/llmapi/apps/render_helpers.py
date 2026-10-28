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
"""Shared builders for the prompt-preparation tests: real tokenizers and requests.

The tests use the tokenizers of real models from ``llm_models_root()``; a test whose model is
not there is skipped. ``real_tokenizer`` returns a shared object: do not change it.
``make_tokenizer`` returns a shallow copy, which may get another chat template or attribute
(the encoding stays the real one); ``copy_of`` returns a deep copy for the tests that change
the encoding itself.
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path

import pytest
from utils.llm_data import llm_datasets_root, llm_models_root

from tensorrt_llm.serve.render import RenderResources

QWEN3 = "Qwen3/Qwen3-0.6B"  # byte-level BPE, Jinja template with tools and thinking
LLAMA = "llama-3.1-model/Llama-3.1-8B-Instruct"  # adds BOS by default
KIMI_K3 = "Kimi-K3"  # tiktoken, a Python (native) chat template, control tokens

SERVER_TEMPLATE = "SERVER:{% for m in messages %}{{ m.content }}|{% endfor %}"
REQUEST_TEMPLATE = "REQUEST:{% for m in messages %}{{ m.content }}|{% endfor %}"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}

_KEEP = object()


def model_dir(name: str) -> Path:
    path = llm_models_root() / name
    if not path.is_dir():
        pytest.skip(f"{name} is not under LLM_MODELS_ROOT")
    return path


@functools.lru_cache(maxsize=None)
def real_tokenizer(name: str = QWEN3):
    """The Hugging Face tokenizer of ``name``, loaded the way a worker loads it. Shared."""
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    loaded = ModelLoader.load_hf_tokenizer(str(model_dir(name)), trust_remote_code=name == KIMI_K3)
    return loaded.tokenizer


def make_tokenizer(chat_template=_KEEP, name: str = QWEN3):
    """A shallow copy of the real tokenizer, with ``chat_template`` as its template if given."""
    tokenizer = copy.copy(real_tokenizer(name))
    if chat_template is not _KEEP:
        tokenizer.chat_template = chat_template
    return tokenizer


def copy_of(name: str = QWEN3):
    """A deep copy of the real tokenizer, for tests that change its encoding."""
    return copy.deepcopy(real_tokenizer(name))


@functools.lru_cache(maxsize=None)
def real_processor(name: str = QWEN3):
    """``AutoProcessor`` as a worker loads it (a tokenizer for a text-only model). Shared."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(model_dir(name)), trust_remote_code=name == KIMI_K3)


def fallback_processor(tokenizer):
    """What ``AutoProcessor`` returns for a text-only model: a tokenizer with the same template."""
    processor = copy.copy(real_processor())
    processor.chat_template = tokenizer.chat_template
    return processor


def checkpoint_dir(tmp_path: Path, name: str = QWEN3, *, config: bool = True) -> Path:
    """A directory of symlinks to the tokenizer files (and the config) of ``name``: no weights."""
    source = model_dir(name)
    target = tmp_path / source.name
    target.mkdir()
    for path in source.iterdir():
        is_config = path.name in ("config.json", "generation_config.json")
        is_tokenizer = path.suffix in (".json", ".txt", ".model", ".py") or path.name.startswith(
            "tokenizer"
        )
        if (is_config and not config) or not (is_config or is_tokenizer):
            continue
        if path.name == "model.safetensors.index.json":
            continue
        (target / path.name).symlink_to(path)
    return target


def use_harmony_vocab(monkeypatch) -> None:
    """Point Harmony at the local copy of its vocabulary; skip when it is not there."""
    vocab = Path(llm_datasets_root()) / "tiktoken_vocab"
    if not vocab.is_dir():
        pytest.skip("the Harmony vocabulary is not available")
    monkeypatch.setenv("TIKTOKEN_RS_CACHE_DIR", str(vocab))
    monkeypatch.setenv("TIKTOKEN_ENCODINGS_BASE", str(vocab))


def harmony_resources(model_type: str | None = "gpt_oss") -> RenderResources:
    """Resources of a router that renders for gpt-oss: Harmony, no tokenizer."""
    return RenderResources.from_tokenizer(None, model_type=model_type, use_harmony=True)


def chat_request(**overrides):
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    fields = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "this is a test"},
            {"role": "user", "content": "get the weather"},
        ],
    }
    fields.update(overrides)
    return ChatCompletionRequest(**fields)


def resources(tokenizer, **overrides) -> RenderResources:
    fields = {"tokenizer": tokenizer, "model_type": "render-test-model"}
    fields.update(overrides)
    return RenderResources(**fields)
