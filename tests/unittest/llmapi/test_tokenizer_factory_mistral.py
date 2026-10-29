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
"""A Mistral-native checkpoint directory loads through ``MistralTokenizer``.

Such a directory ships mistral-common's ``tekken.json`` and is not a complete HF
checkpoint (``config.json`` and ``tokenizer.json``). AutoTokenizer converts
``tekken.json`` into a fast tokenizer whose ids lack the special-token offset and
whose ``encode`` adds no BOS, so both ``tokenizer_factory`` and
``load_hf_tokenizer`` (the path the LLM takes when no tokenizer is given) must
read it through the ``mistral_common`` alias instead. A complete HF checkpoint
keeps the AutoTokenizer path, even with a ``tekken.json`` beside it.
"""

import gc
import json
import pickle
import shutil
from pathlib import Path

import pytest

mistral_common = pytest.importorskip("mistral_common")

from mistral_common.tokens.tokenizers.mistral import (  # noqa: E402
    MistralTokenizer as MistralCommonTokenizer,
)

from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer  # noqa: E402
from tensorrt_llm.tokenizer import (  # noqa: E402
    TransformersTokenizer,
    load_hf_tokenizer,
    tokenizer_factory,
)

pytestmark = pytest.mark.cpu_only

PROMPT = "The capital of France is"


def _bundled_tekken_file() -> Path:
    files = sorted((Path(mistral_common.__file__).parent / "data").glob("tekken*.json"))
    if not files:
        pytest.skip("this mistral_common ships no tekken tokenizer file")
    return files[-1]


@pytest.fixture(scope="module")
def tekken_dir(tmp_path_factory) -> Path:
    model_dir = tmp_path_factory.mktemp("tekken_only")
    shutil.copy(_bundled_tekken_file(), model_dir / "tekken.json")
    return model_dir


@pytest.fixture(scope="module")
def mistral_common_ids(tekken_dir) -> list:
    mistral = MistralCommonTokenizer.from_file(str(tekken_dir / "tekken.json"))
    return mistral.instruct_tokenizer.tokenizer.encode(PROMPT, bos=True, eos=False)


def _assert_reads_like_mistral_common(tokenizer, expected_ids):
    assert type(tokenizer) is MistralTokenizer
    assert expected_ids[0] == tokenizer.bos_token_id == 1
    assert tokenizer.encode(PROMPT) == expected_ids
    # The call the LLM's input processor makes.
    assert tokenizer.encode(PROMPT, add_special_tokens=True) == expected_ids
    assert tokenizer.decode(expected_ids, skip_special_tokens=True) == PROMPT


def test_load_hf_tokenizer_reads_tekken_through_mistral_common(tekken_dir, mistral_common_ids):
    _assert_reads_like_mistral_common(load_hf_tokenizer(str(tekken_dir)), mistral_common_ids)


@pytest.mark.parametrize("as_path", [False, True], ids=["str", "Path"])
def test_tokenizer_factory_reads_tekken_through_mistral_common(
    tekken_dir, mistral_common_ids, as_path
):
    model_dir = tekken_dir if as_path else str(tekken_dir)
    _assert_reads_like_mistral_common(tokenizer_factory(model_dir), mistral_common_ids)


def test_pickle_round_trip_keeps_the_class(tekken_dir, mistral_common_ids):
    tokenizer = load_hf_tokenizer(str(tekken_dir))
    restored = pickle.loads(pickle.dumps(tokenizer))
    _assert_reads_like_mistral_common(restored, mistral_common_ids)


def test_cloudpickle_round_trip_keeps_the_class(tekken_dir, mistral_common_ids):
    cloudpickle = pytest.importorskip("cloudpickle")
    tokenizer = load_hf_tokenizer(str(tekken_dir))
    restored = cloudpickle.loads(cloudpickle.dumps(tokenizer))
    _assert_reads_like_mistral_common(restored, mistral_common_ids)


@pytest.fixture
def tokenizer_json_dir(tmp_path) -> Path:
    """A complete HF checkpoint directory (config.json, tokenizer.json) beside a tekken.json."""
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[UNK]": 0, "the": 1, "capital": 2, "of": 3, "france": 4, "is": 5}
    word_level = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    word_level.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    word_level.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"})
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mistral"}))
    shutil.copy(_bundled_tekken_file(), tmp_path / "tekken.json")
    return tmp_path


def test_hf_checkpoint_directory_keeps_the_transformers_path(tokenizer_json_dir):
    expected = TransformersTokenizer.from_pretrained(str(tokenizer_json_dir)).encode(
        "the capital of france is"
    )
    assert expected == [1, 2, 3, 4, 5]
    for tokenizer in (
        load_hf_tokenizer(str(tokenizer_json_dir)),
        tokenizer_factory(str(tokenizer_json_dir)),
    ):
        assert type(tokenizer) is TransformersTokenizer
        assert tokenizer.encode("the capital of france is") == expected


@pytest.fixture(scope="module")
def tekken_v15_dir(tmp_path_factory):
    from mistral_common.tokens.tokenizers.base import SpecialTokens

    metadata = json.loads(_bundled_tekken_file().read_text())
    metadata["config"]["version"] = "v15"
    metadata["special_tokens"] = [
        {"rank": rank, "token_str": token.value, "is_control": True}
        for rank, token in enumerate(SpecialTokens)
    ]
    metadata["model_settings_builder"] = {
        "reasoning_effort": {
            "type": "enum",
            "accepts_none": True,
            "default": "none",
            "values": ["none", "high"],
        }
    }
    model_dir = tmp_path_factory.mktemp("tekken_v15")
    (model_dir / "tekken.json").write_text(json.dumps(metadata))
    return model_dir


@pytest.fixture
def unsupported_schema_dir(tekken_v15_dir, tmp_path, monkeypatch):
    # Exercise the compatibility branch even when the installed mistral-common
    # already understands json_schema. Other validation errors must still escape.
    from pydantic import ValidationError

    from tensorrt_llm._torch.models.checkpoints.mistral import tokenizer as module

    original_init = module.TransformersMistralTokenizer.__init__

    def legacy_init(self, *args, **kwargs):
        path = Path(kwargs.get("tokenizer_path") or args[0])
        settings = json.loads(path.read_text()).get("model_settings_builder", {})
        extra = set(settings) - {"reasoning_effort"}
        if extra:
            raise ValidationError.from_exception_data(
                "ModelSettingsBuilder",
                [
                    {"type": "extra_forbidden", "loc": (key,), "input": settings[key]}
                    for key in sorted(extra)
                ],
            )
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(module.TransformersMistralTokenizer, "__init__", legacy_init)
    metadata = json.loads((tekken_v15_dir / "tekken.json").read_text())
    metadata["model_settings_builder"]["json_schema"] = {
        "type": "json_schema",
        "accepts_none": False,
        "default": None,
    }
    (tmp_path / "tekken.json").write_text(json.dumps(metadata))
    return tmp_path


@pytest.mark.parametrize("as_file", [False, True])
def test_unsupported_schema_preserves_tokenization(unsupported_schema_dir, tekken_v15_dir, as_file):
    source = unsupported_schema_dir / "tekken.json"
    original = source.read_bytes()
    reference = MistralCommonTokenizer.from_file(str(tekken_v15_dir / "tekken.json"))
    tokenizer = MistralTokenizer.from_pretrained(str(source if as_file else unsupported_schema_dir))
    for prompt in (PROMPT, "71432", "Grüße — 你好 👋"):
        expected = reference.instruct_tokenizer.tokenizer.encode(prompt, bos=True, eos=False)
        assert tokenizer.encode(prompt) == expected
        assert tokenizer.encode(prompt, add_special_tokens=True) == expected
        assert tokenizer.decode(expected, skip_special_tokens=True) == prompt
    assert tokenizer.eos_token_id == reference.instruct_tokenizer.tokenizer.eos_id
    assert (
        tokenizer.tokenizer.model_settings_builder
        == reference.instruct_tokenizer.tokenizer.model_settings_builder
    )
    assert source.read_bytes() == original


def test_schema_fallback_pickle_reloads_original_checkpoint(unsupported_schema_dir):
    tokenizer = load_hf_tokenizer(str(unsupported_schema_dir))
    expected = tokenizer.encode(PROMPT)
    serialized = pickle.dumps(tokenizer)
    # Restore only after the original, and any temporary file it owns, is collected.
    del tokenizer
    gc.collect()
    restored = pickle.loads(serialized)
    _assert_reads_like_mistral_common(restored, expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("unknown_setting", {"type": "invalid"}),
        ("reasoning_effort", {"type": "invalid"}),
        # The fixture's json_schema entry with accepts_none flipped: unrecognized metadata.
        ("json_schema", {"type": "json_schema", "accepts_none": True, "default": None}),
    ],
    ids=["unknown_setting", "reasoning_effort", "unrecognized_json_schema"],
)
def test_schema_fallback_preserves_other_validation_errors(unsupported_schema_dir, field, value):
    from pydantic import ValidationError

    source = unsupported_schema_dir / "tekken.json"
    metadata = json.loads(source.read_text())
    metadata["model_settings_builder"][field] = value
    source.write_text(json.dumps(metadata))
    with pytest.raises(ValidationError):
        MistralTokenizer.from_pretrained(str(unsupported_schema_dir))


def test_tokenizer_json_without_a_config_reads_through_mistral_common(
    tokenizer_json_dir, mistral_common_ids
):
    """A Mistral-native directory that also ships HF tokenizer files is still native."""
    (tokenizer_json_dir / "config.json").unlink()
    for tokenizer in (
        load_hf_tokenizer(str(tokenizer_json_dir)),
        tokenizer_factory(str(tokenizer_json_dir)),
    ):
        _assert_reads_like_mistral_common(tokenizer, mistral_common_ids)
