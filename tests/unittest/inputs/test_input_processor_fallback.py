# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""create_input_processor uses DefaultInputProcessor for an unregistered architecture."""

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm._errors import ConfigInputError, UnsupportedError
from tensorrt_llm._torch.models import modeling_utils
from tensorrt_llm.inputs.registry import DefaultInputProcessor, create_input_processor

pytestmark = pytest.mark.cpu_only

_LLAMA_CONFIG = {
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "num_hidden_layers": 1,
    "vocab_size": 128,
}


@pytest.mark.parametrize(
    "arch, error",
    [
        ("FooBarForCausalLM", UnsupportedError),
        # Gemma4 has no registered class when transformers is older than it requires.
        ("Gemma4ForCausalLM", ConfigInputError),
    ],
)
def test_unregistered_architecture_uses_default_input_processor(tmp_path, monkeypatch, arch, error):
    monkeypatch.setattr(modeling_utils, "get_registered_model_class", lambda arch: None)
    with pytest.raises(error):
        modeling_utils.get_model_architecture(SimpleNamespace(architectures=[arch]))

    (tmp_path / "config.json").write_text(json.dumps({**_LLAMA_CONFIG, "architectures": [arch]}))
    processor = create_input_processor(str(tmp_path), tokenizer=None)

    assert type(processor) is DefaultInputProcessor
