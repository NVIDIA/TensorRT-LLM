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

from collections.abc import Callable

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, ByteLevel, Fuse, Sequence, Strip
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from transformers import PreTrainedTokenizerFast

from tensorrt_llm.tokenizer import TransformersTokenizer

# Pure-Python tokenizer checks with no GPU dependency. The l0_cpu stage runs pytest
# with ``-m cpu_only`` and conftest drops any file lacking this marker.
pytestmark = pytest.mark.cpu_only


def _byte_fallback_tokenizer() -> TransformersTokenizer:
    """A tokenizer whose ``<0xC2>`` / ``<0xA0>`` ids are halves of one UTF-8 character.

    ``DecodeStream`` returns nothing for ``<0xC2>`` until ``<0xA0>`` completes the
    character, which is the case a streamed response must not lose at its end.
    """
    backend = Tokenizer(
        WordLevel(
            {
                "<0xC2>": 0,
                "<0xA0>": 1,
                "»": 2,
                "<unk>": 3,
                "</s>": 4,
            },
            unk_token="<unk>",
        )
    )
    backend.decoder = Sequence([ByteFallback(), Fuse()])
    backend.add_special_tokens(["</s>"])
    return TransformersTokenizer(PreTrainedTokenizerFast(tokenizer_object=backend))


def _space_prefixed_byte_level_tokenizer() -> TransformersTokenizer:
    """A byte-level BPE whose ``ĠâĶ`` id is a space plus two bytes of ``─`` (U+2500).

    ``DecodeStream`` holds ``ĠâĶ`` until ``Ģ`` supplies the last byte. The decoder
    strips one leading space of the whole text, so the held piece decodes to a
    different string on its own than after the token before it: only a decode in
    that context reproduces what decoding the whole output gives.
    """
    backend = Tokenizer(
        BPE(
            vocab={"o": 0, "k": 1, "ok": 2, "Ġ": 3, "â": 4, "Ķ": 5, "Ģ": 6, "ĠâĶ": 7, "âĶ": 8},
            merges=[("o", "k"), ("â", "Ķ"), ("Ġ", "âĶ")],
        )
    )
    backend.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    backend.decoder = Sequence([ByteLevel(), Strip(" ", 1, 0)])
    return TransformersTokenizer(PreTrainedTokenizerFast(tokenizer_object=backend))


def _stream(tokenizer: TransformersTokenizer, token_ids: list[int], **kwargs) -> str:
    """Decode one token per call, as a streamed response does, flushing on the last.

    Goes through ``decode_incrementally``, the entry point the streaming result
    path calls, so the test also covers handing ``flush`` down to the HF backend.
    """
    text = ""
    states = None
    for index, token_id in enumerate(token_ids):
        text, states = tokenizer.decode_incrementally(
            [token_id], text, states, flush=index == len(token_ids) - 1, **kwargs
        )
    return text


@pytest.fixture(autouse=True)
def _hf_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route ``decode_incrementally`` to the HF ``DecodeStream`` path under test."""
    from tensorrt_llm.tokenizer import tokenizer as tokenizer_module

    monkeypatch.setattr(tokenizer_module, "TLLM_INCREMENTAL_DETOKENIZATION_BACKEND", "HF")


def test_hf_decode_incrementally_recovers_from_invalid_prefix() -> None:
    tokenizer = _byte_fallback_tokenizer()

    text = ""
    states = None
    for token_id in [0, 1, 0, 2]:
        text, states = tokenizer.hf_decode_incrementally([token_id], text, states)

    assert text == "\N{NO-BREAK SPACE}»"


@pytest.mark.parametrize(
    "token_ids, expected",
    [
        pytest.param([0], "\N{REPLACEMENT CHARACTER}", id="single_partial_token"),
        pytest.param([2, 0], "»\N{REPLACEMENT CHARACTER}", id="text_then_partial_token"),
        pytest.param(
            [0, 1, 0],
            "\N{NO-BREAK SPACE}\N{REPLACEMENT CHARACTER}",
            id="complete_char_then_partial_token",
        ),
        pytest.param([0, 4], "\N{REPLACEMENT CHARACTER}", id="partial_token_then_special_token"),
    ],
)
def test_hf_decode_incrementally_flush_releases_held_tokens(
    token_ids: list[int], expected: str
) -> None:
    """The last response of a stream releases what ``DecodeStream`` still holds.

    Without the flush a response whose final tokens are an incomplete UTF-8
    sequence streams no text for them at all; a one-token response then has no
    content, while the same response decoded in full (non-streamed) does.
    """
    tokenizer = _byte_fallback_tokenizer()

    text = _stream(tokenizer, token_ids, skip_special_tokens=True)

    assert text == expected


@pytest.mark.parametrize(
    "skip_special_tokens, expected",
    [
        pytest.param(True, "", id="skipped"),
        pytest.param(False, "</s>", id="kept"),
    ],
)
def test_hf_decode_incrementally_flush_keeps_special_token_policy(
    skip_special_tokens: bool, expected: str
) -> None:
    tokenizer = _byte_fallback_tokenizer()

    text = _stream(tokenizer, [4], skip_special_tokens=skip_special_tokens)

    assert text == expected


def test_hf_decode_incrementally_without_flush_holds_partial_token() -> None:
    """Intermediate responses keep the streaming behaviour: nothing is emitted early."""
    tokenizer = _byte_fallback_tokenizer()

    text, states = tokenizer.hf_decode_incrementally([2, 0], "", None)
    assert text == "»"

    text, _ = tokenizer.hf_decode_incrementally([1], text, states, flush=True)
    assert text == "»\N{NO-BREAK SPACE}"


@pytest.mark.parametrize(
    "token_ids",
    [
        pytest.param([2, 7], id="ends_on_held_piece"),
        pytest.param([2, 7, 6], id="held_piece_completed"),
        pytest.param([7], id="only_held_piece"),
    ],
)
def test_hf_decode_incrementally_flush_matches_full_decode_in_context(token_ids: list[int]) -> None:
    """The flushed tail is decoded after the previous token, so the stream equals a full decode.

    ``ĠâĶ`` decodes to a space plus U+FFFD after ``ok`` but to U+FFFD alone on its own,
    because the decoder strips a leading space of the text. Decoding the held id without
    the token before it would drop the space that the full decode keeps.
    """
    tokenizer = _space_prefixed_byte_level_tokenizer()
    assert tokenizer.decode([7]) != tokenizer.decode([2, 7])[len(tokenizer.decode([2])) :]

    text = _stream(tokenizer, token_ids, skip_special_tokens=True)

    assert text == tokenizer.decode(token_ids, skip_special_tokens=True)


@pytest.mark.parametrize(
    "make_tokenizer, held_ids",
    [
        pytest.param(_byte_fallback_tokenizer, [0], id="byte_fallback"),
        pytest.param(_space_prefixed_byte_level_tokenizer, [2, 7], id="space_prefixed_byte_level"),
    ],
)
def test_hf_decode_incrementally_flush_with_no_new_tokens(
    make_tokenizer: Callable[[], TransformersTokenizer], held_ids: list[int]
) -> None:
    """A final response that brings no new token ids still releases the held ones."""
    tokenizer = make_tokenizer()

    text, states = tokenizer.decode_incrementally(held_ids, "", None, skip_special_tokens=True)
    text, _ = tokenizer.decode_incrementally([], text, states, flush=True, skip_special_tokens=True)

    assert text == tokenizer.decode(held_ids, skip_special_tokens=True)


def test_num_special_tokens_to_add_delegates_to_the_wrapped_tokenizer() -> None:
    """The wrapper answers from the tokenizer it wraps.

    ``PreTrainedTokenizerBase.num_special_tokens_to_add`` raises NotImplementedError; the MiniMax-M3 input
    processor's tokenization-cache check calls it on the wrapper.
    """
    tokenizer = _byte_fallback_tokenizer()
    assert tokenizer.num_special_tokens_to_add() == tokenizer.tokenizer.num_special_tokens_to_add()
    assert tokenizer.num_special_tokens_to_add(
        pair=True
    ) == tokenizer.tokenizer.num_special_tokens_to_add(True)
