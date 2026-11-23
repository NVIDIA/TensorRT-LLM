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
"""Mistral Tekken tokenizers for tests.

``bundled`` is the newest Tekken file mistral-common ships (v3, no reasoning tokens); ``v13``
and ``v15`` are the files of Mistral-Large-3 and Mistral-Small-4 under LLM_MODELS_ROOT, with
``[THINK]``; v15 also declares a model setting, the reasoning effort.
"""

import json
from pathlib import Path

import pytest

from test_common.llm_data import llm_models_root

TEKKEN_CHECKPOINT_FILES = {
    "v13": "Mistral-Large-3-675B/Mistral-Large-3-675B-Instruct-2512/tekken.json",
    "v15": "Mistral-Small-4-119B-2603/Mistral-Small-4-119B-2603/tekken.json",
}


HF_TOKENIZER_TEXT = "the capital of france is"
"""Text the tokenizer of ``write_hf_tokenizer`` encodes as ``HF_TOKENIZER_IDS``."""
HF_TOKENIZER_IDS = [1, 2, 3, 4, 5]


def bundled_tekken_file() -> Path:
    """The newest Tekken file mistral-common ships; skips when it ships none."""
    mistral_common = pytest.importorskip("mistral_common")
    files = sorted((Path(mistral_common.__file__).parent / "data").glob("tekken*.json"))
    if not files:
        pytest.skip("this mistral_common ships no Tekken file")
    return files[-1]


def write_hf_tokenizer(directory: Path) -> None:
    """Write a word-level Hugging Face tokenizer (``tokenizer.json``, ``tokenizer_config.json``).

    Its vocabulary differs from every Tekken one, so a test can tell which of the two files of
    a directory a loader read.
    """
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[UNK]": 0, **{word: i for i, word in enumerate(HF_TOKENIZER_TEXT.split(), 1)}}
    word_level = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    word_level.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    word_level.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"})
    )


def load_tekken(source: str):
    """A ``MistralTokenizer`` over the ``source`` Tekken file; skips when it is not there."""
    pytest.importorskip("mistral_common")
    from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer

    if source == "bundled":
        path = bundled_tekken_file()
    else:
        root = llm_models_root()
        path = None if root is None else root / TEKKEN_CHECKPOINT_FILES[source]
    if path is None or not path.is_file():
        pytest.skip(f"no {source} Tekken file")
    return MistralTokenizer.from_pretrained(str(path))
