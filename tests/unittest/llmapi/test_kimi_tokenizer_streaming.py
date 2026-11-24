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

"""Checkpoint-free coverage of Kimi's byte-preserving streaming contract."""

import pytest
from transformers import PreTrainedTokenizer

from tensorrt_llm.tokenizer import TransformersTokenizer

pytestmark = pytest.mark.cpu_only


class TikTokenTokenizer(PreTrainedTokenizer):
    """Tiny byte vocabulary with the Kimi tokenizer's decoding behavior."""

    def __init__(self) -> None:
        self.byte_decoder = {chr(i): i for i in range(256)}
        self.vocab = {chr(i): i for i in range(256)}
        self.vocab[" " * 16] = 256
        self.tokens = dict(enumerate(self.vocab))
        super().__init__(clean_up_tokenization_spaces=False)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        return [chr(byte) for byte in text.encode("utf-8")]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokens[index]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return bytes(self.byte_decoder[c] for c in "".join(tokens)).decode("utf-8", "replace")


@pytest.mark.parametrize("chunk_size", [1, 8])
@pytest.mark.parametrize(
    "text", ["normal text", "\n \t  ", "中文🙂\n", 'key="   \n\t     ', "a\u00a0b\r\n", "e\u0301"]
)
def test_kimi_streaming_matches_full_decode(text: str, chunk_size: int) -> None:
    tokenizer = TransformersTokenizer(TikTokenTokenizer())
    ids = tokenizer.encode(text, add_special_tokens=False)
    streamed = ""
    states = None
    for start in range(0, len(ids), chunk_size):
        streamed, states = tokenizer.decode_incrementally(
            ids[start : start + chunk_size], streamed, states
        )
    streamed, _ = tokenizer.decode_incrementally([], streamed, states, flush=True)
    assert streamed == tokenizer.decode(ids) == text


def test_kimi_streaming_emits_long_whitespace_runs() -> None:
    tokenizer = TransformersTokenizer(TikTokenTokenizer())
    text = ""
    states = None
    for count in range(8, 20001, 8):
        text, states = tokenizer.decode_incrementally([256] * 8, text, states)
        # Each response must expose all complete whitespace without a flush.
        assert len(text) == count * 16
    assert text == " " * (20000 * 16)


def test_kimi_streaming_buffers_split_utf8() -> None:
    tokenizer = TransformersTokenizer(TikTokenTokenizer())
    text, states = tokenizer.decode_incrementally([32])
    assert text == " "
    for byte in [0xE4, 0xB8]:
        text, states = tokenizer.decode_incrementally([byte], text, states)
        assert text == " "
    text, states = tokenizer.decode_incrementally([0xAD], text, states)
    assert text == " 中"
    text, states = tokenizer.decode_incrementally([0xE4], text, states)
    assert text == " 中"
    text, _ = tokenizer.decode_incrementally([], text, states, flush=True)
    assert text == " 中�"


def test_kimi_streaming_skips_special_tokens() -> None:
    inner = TikTokenTokenizer()
    inner.add_special_tokens({"eos_token": "<eos>"})
    tokenizer = TransformersTokenizer(inner)
    text, states = tokenizer.decode_incrementally([32], skip_special_tokens=True)
    assert text == " "
    text, states = tokenizer.decode_incrementally(
        [inner.eos_token_id], text, states, skip_special_tokens=True
    )
    assert text == " "
    text, _ = tokenizer.decode_incrementally([32], text, states, skip_special_tokens=True)
    assert text == "  "


def test_non_kimi_streaming_defers_whitespace() -> None:
    class OtherTokenizer(TikTokenTokenizer):
        pass

    tokenizer = TransformersTokenizer(OtherTokenizer())
    text, states = tokenizer.decode_incrementally([32])
    assert text == ""
    text, _ = tokenizer.decode_incrementally([ord("a")], text, states)
    assert text == " a"
