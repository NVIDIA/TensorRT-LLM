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

from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, Fuse, Sequence
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils import PreTrainedTokenizer

from tensorrt_llm.tokenizer import TransformersTokenizer


class _ToySlowTokenizer(PreTrainedTokenizer):
    """A minimal genuine *slow* tokenizer (``is_fast`` is False).

    The slow ``convert_ids_to_tokens`` path is the only one that filters
    special tokens in TransformersTokenizer's wrapper; fast tokenizers delegate
    straight to the backend.  This toy tokenizer lets us exercise that path
    deterministically without downloading weights.
    """

    def __init__(self, **kwargs):
        self._vocab = {f"tok{i}": i for i in range(64)}
        self._ids = {v: k for k, v in self._vocab.items()}
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, 0)

    def _convert_id_to_token(self, index):
        return self._ids.get(int(index), "tok0")

    def _tokenize(self, text):
        return text.split()


def test_convert_ids_to_tokens_reflects_special_tokens_registered_after_wrap() -> None:
    # Specials registered after wrapping (before first detok) must be reflected.
    inner = _ToySlowTokenizer()
    inner.add_special_tokens(
        {
            "bos_token": "tok1",
            "eos_token": "tok2",
            "additional_special_tokens": ["tok5"],
        }
    )
    tokenizer = TransformersTokenizer(inner)
    assert inner.is_fast is False

    inner.add_special_tokens({"additional_special_tokens": ["tok5", "tok42"]})

    ids = [10, 42, 11, 5, 12]
    got = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
    assert got == inner.convert_ids_to_tokens(ids, skip_special_tokens=True)
    assert got == ["tok10", "tok11", "tok12"]


def test_convert_tokens_to_string_reflects_special_tokens_registered_after_wrap() -> None:
    # The same must hold for convert_tokens_to_string's skip path.
    inner = _ToySlowTokenizer()
    inner.add_special_tokens(
        {
            "bos_token": "tok1",
            "eos_token": "tok2",
            "additional_special_tokens": ["tok5"],
        }
    )
    tokenizer = TransformersTokenizer(inner)
    assert inner.is_fast is False

    inner.add_special_tokens({"additional_special_tokens": ["tok5", "tok42"]})

    tokens = ["tok10", "tok42", "tok11"]
    got = tokenizer.convert_tokens_to_string(tokens, skip_special_tokens=True)
    assert got == "tok10 tok11"


def test_hf_decode_incrementally_recovers_from_invalid_prefix() -> None:
    backend = Tokenizer(
        WordLevel(
            {
                "<0xC2>": 0,
                "<0xA0>": 1,
                "»": 2,
                "<unk>": 3,
            },
            unk_token="<unk>",
        )
    )
    backend.decoder = Sequence([ByteFallback(), Fuse()])
    tokenizer = TransformersTokenizer(PreTrainedTokenizerFast(tokenizer_object=backend))

    text = ""
    states = None
    for token_id in [0, 1, 0, 2]:
        text, states = tokenizer.hf_decode_incrementally([token_id], text, states)

    assert text == "\N{NO-BREAK SPACE}»"
