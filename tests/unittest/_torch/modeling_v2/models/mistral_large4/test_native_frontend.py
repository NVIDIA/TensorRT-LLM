# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The native Mistral-Large-4 format reads no HF frontend file.

A Mistral-Large-4 checkpoint may ship, beside its Mistral-native params.json and tekken.json,
the HF frontend files tokenizer.json, tokenizer_config.json, chat_template.jinja,
processor_config.json and generation_config.json. Under ``checkpoint_format="mistral_large_4"``
the LLM tokenizes and renders chats with mistral-common's MistralTokenizer and takes no
sampling defaults from generation_config.json; a directory without config.json loads
MistralTokenizer for every reader, whatever HF tokenizer files it ships.

The directory tests run on CPU with mistral-common's bundled tekken file. The checkpoint
test needs MISTRAL_LARGE4_CHECKPOINT_DIR.
"""

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

mistral_common = pytest.importorskip("mistral_common")
tokenizers = pytest.importorskip("tokenizers")

from transformers import PreTrainedTokenizerFast  # noqa: E402

from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer  # noqa: E402
from tensorrt_llm.inputs.utils import apply_chat_template  # noqa: E402
from tensorrt_llm.llmapi.llm import BaseLLM  # noqa: E402
from tensorrt_llm.tokenizer import TransformersTokenizer, load_hf_tokenizer  # noqa: E402

pytestmark = pytest.mark.cpu_only

_HF_FRONTEND = {
    "tokenizer_config.json": {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"},
    "chat_template.jinja": "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}",
    "processor_config.json": {"processor_class": "PixtralProcessor"},
    "generation_config.json": {"temperature": 1.0, "top_p": 0.95, "eos_token_id": 2},
}


def _word_level_tokenizer() -> "tokenizers.Tokenizer":
    vocab = {"[UNK]": 0, "the": 1, "capital": 2, "of": 3, "france": 4, "is": 5}
    word_level = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    word_level.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return word_level


def _bundled_tekken_file() -> Path:
    files = sorted((Path(mistral_common.__file__).parent / "data").glob("tekken*.json"))
    if not files:
        pytest.skip("this mistral_common ships no tekken tokenizer file")
    return files[-1]


@pytest.fixture
def native_dir(tmp_path) -> Path:
    """Mistral-native files and the HF frontend files, no config.json."""
    shutil.copy(_bundled_tekken_file(), tmp_path / "tekken.json")
    (tmp_path / "params.json").write_text(json.dumps({"dim": 8}))
    _word_level_tokenizer().save(str(tmp_path / "tokenizer.json"))
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


def test_a_native_directory_loads_mistral_common_for_every_reader(native_dir):
    assert type(load_hf_tokenizer(str(native_dir))) is MistralTokenizer


def test_a_complete_hf_checkpoint_keeps_the_transformers_path(native_dir):
    (native_dir / "config.json").write_text(json.dumps({"model_type": "mistral"}))
    assert type(load_hf_tokenizer(str(native_dir))) is TransformersTokenizer


def test_the_native_format_loads_mistral_common_even_beside_a_config(native_dir):
    (native_dir / "config.json").write_text(json.dumps({"model_type": "mistral"}))
    llm = _llm(native_dir, "mistral_large_4")
    assert type(BaseLLM._try_load_tokenizer(llm)) is MistralTokenizer


def test_the_native_format_reads_no_generation_config(native_dir):
    llm = _llm(native_dir, "mistral_large_4")
    assert BaseLLM._try_load_generation_config(llm) is None
    assert BaseLLM._try_load_generation_config_explicit_values(llm) == {}


def test_other_formats_still_read_the_generation_config(native_dir):
    llm = _llm(native_dir, "HF")
    assert BaseLLM._try_load_generation_config_explicit_values(llm)["top_p"] == 0.95


def test_the_native_renderer_rejects_an_hf_tokenizer():
    hf = TransformersTokenizer(PreTrainedTokenizerFast(tokenizer_object=_word_level_tokenizer()))
    with pytest.raises(ValueError, match="MistralTokenizer"):
        apply_chat_template(
            model_type="mistral_large_4",
            tokenizer=hf,
            processor=None,
            conversation=[{"role": "user", "content": "the capital"}],
            add_generation_prompt=True,
            mm_placeholder_counts=[{}],
        )


def test_the_checkpoint_loads_mistral_common_for_every_reader():
    path = os.environ.get("MISTRAL_LARGE4_CHECKPOINT_DIR")
    if not path:
        pytest.skip("needs MISTRAL_LARGE4_CHECKPOINT_DIR")
    assert type(load_hf_tokenizer(path)) is MistralTokenizer
    llm = _llm(path, "mistral_large_4")
    assert type(BaseLLM._try_load_tokenizer(llm)) is MistralTokenizer
    assert BaseLLM._try_load_generation_config(llm) is None
