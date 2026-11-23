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
``load_hf_tokenizer`` (the path the LLM, the post-processing workers and the
routers take when no tokenizer is given) must read it through the
``mistral_common`` alias instead. A complete HF checkpoint keeps the
AutoTokenizer path, even with a ``tekken.json`` beside it.
"""

import json
import shutil
from pathlib import Path

import pytest
from test_common.mistral_tekken import (
    HF_TOKENIZER_IDS,
    HF_TOKENIZER_TEXT,
    bundled_tekken_file,
    write_hf_tokenizer,
)

pytest.importorskip("mistral_common")

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


@pytest.fixture(scope="module")
def tekken_dir(tmp_path_factory) -> Path:
    model_dir = tmp_path_factory.mktemp("tekken_only")
    shutil.copy(bundled_tekken_file(), model_dir / "tekken.json")
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


def test_a_tekken_directory_reads_through_mistral_common(tekken_dir, mistral_common_ids):
    for tokenizer in (load_hf_tokenizer(str(tekken_dir)), tokenizer_factory(tekken_dir)):
        _assert_reads_like_mistral_common(tokenizer, mistral_common_ids)


@pytest.fixture
def tokenizer_json_dir(tmp_path) -> Path:
    """A complete HF checkpoint directory (config.json, tokenizer.json) beside a tekken.json."""
    write_hf_tokenizer(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mistral"}))
    shutil.copy(bundled_tekken_file(), tmp_path / "tekken.json")
    return tmp_path


def test_hf_checkpoint_directory_keeps_the_transformers_path(tokenizer_json_dir):
    expected = TransformersTokenizer.from_pretrained(str(tokenizer_json_dir)).encode(
        HF_TOKENIZER_TEXT
    )
    assert expected == HF_TOKENIZER_IDS
    for tokenizer in (
        load_hf_tokenizer(str(tokenizer_json_dir)),
        tokenizer_factory(str(tokenizer_json_dir)),
    ):
        assert type(tokenizer) is TransformersTokenizer
        assert tokenizer.encode(HF_TOKENIZER_TEXT) == expected


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
