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

"""CPU coverage of the Kimi fast path against Hugging Face's slow tokenizer.

The tiny local vocabulary needs no checkpoint or download. The fixture uses
Kimi's native-encode / HF-with-options contract, including the HF added-token
Trie and token-ID/string round trip. Real model vocabulary is not required to
exercise the compatibility guards.
"""

import random
from unittest.mock import Mock

import pytest
import tiktoken
import torch
from transformers import AddedToken, PreTrainedTokenizer

from tensorrt_llm.tokenizer.tokenizer import TransformersTokenizer


class TikTokenTokenizer(PreTrainedTokenizer):
    def __init__(self, with_added_tokens: bool = True) -> None:
        ranks = {bytes([i]): i for i in range(256)}
        ranks[b"ab"] = 256  # Detect accidental per-character encoding.
        self.model = tiktoken.Encoding(
            name="kimi-test", pat_str=r"\s+|\S+", mergeable_ranks=ranks, special_tokens={}
        )
        self.decoder = {i: piece.hex() for piece, i in ranks.items()}
        self.encoder = {piece: i for i, piece in self.decoder.items()}
        super().__init__(model_max_length=100000)
        if with_added_tokens:
            # Overlap and regex metacharacters exercise longest-first escaping.
            self.add_tokens(["<tag>", "<tag>long", "[BOS]", ".*"], special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        return [self.decoder[i] for i in self.model.encode(text)]

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder[index]

    def encode(self, text, *args, **kwargs) -> list[int]:
        if args or kwargs:
            return super().encode(text, *args, **kwargs)
        return self.model.encode(text)


# Match the checkpoint class identity checked by the production guard.
TikTokenTokenizer.__module__ = "tests.tokenization_kimi"


@pytest.fixture
def tokenizer(monkeypatch):
    monkeypatch.setenv("K3_TOKENIZER_FASTPATH", "1")
    return TikTokenTokenizer()


def _assert_batch_equal(actual, expected) -> None:
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key].dtype == expected[key].dtype
        assert actual[key].shape == expected[key].shape
        assert torch.equal(actual[key], expected[key])


_rng = random.Random(42)
_TEXTS = [
    "",
    "ab ab hello",
    "café 日本語 😀\x00\n\t",
    "[BOS]<tag>long<tag>.*ab",
    "ab" * 15000,
    " " * 30000 + "[BOS]" + "ab" * 15000,
] + [
    "".join(_rng.choices(["ab", "<tag>", "<tag>long", "[BOS]", ".*", " 日本語 ", "\n"], k=40))
    for _ in range(10)
]


@pytest.mark.parametrize("text", _TEXTS, ids=[f"case-{i}" for i in range(len(_TEXTS))])
def test_fast_encode_and_tensor_call_match_hf(tokenizer, text, monkeypatch) -> None:
    expected_ids = tokenizer.encode(text, add_special_tokens=False)
    expected_batch = tokenizer(text, return_tensors="pt")
    # Fail if the candidate silently takes HF's slow tokenization path.
    monkeypatch.setattr(tokenizer, "_tokenize", Mock(side_effect=AssertionError("HF fallback")))
    wrapper = TransformersTokenizer(tokenizer)
    assert wrapper.encode(text, add_special_tokens=False) == expected_ids
    _assert_batch_equal(wrapper(text, return_tensors="pt"), expected_batch)


@pytest.mark.parametrize("text", [[97, 98], (97, 98), ["61", "62"], ("61", "62"), [], ()])
def test_non_string_encode_retains_hf_behavior(tokenizer, text, monkeypatch) -> None:
    wrapper = TransformersTokenizer(tokenizer)
    original = tokenizer.encode
    spy = Mock(wraps=original)
    monkeypatch.setattr(tokenizer, "encode", spy)
    try:
        expected = original(text, add_special_tokens=False)
    except (TypeError, ValueError, AssertionError) as error:
        with pytest.raises(type(error)) as actual:
            wrapper.encode(text, add_special_tokens=False)
        assert str(actual.value) == str(error)
    else:
        assert wrapper.encode(text, add_special_tokens=False) == expected
    spy.assert_called_once_with(text, add_special_tokens=False)


def test_empty_added_vocab_uses_hf(monkeypatch) -> None:
    monkeypatch.setenv("K3_TOKENIZER_FASTPATH", "1")
    tokenizer = TikTokenTokenizer(with_added_tokens=False)
    assert tokenizer.get_added_vocab() == {}
    spy = Mock(wraps=tokenizer.encode)
    monkeypatch.setattr(tokenizer, "encode", spy)
    assert TransformersTokenizer(tokenizer).encode("ab", add_special_tokens=False) == [256]
    spy.assert_called_once_with("ab", add_special_tokens=False)
    _assert_batch_equal(
        TransformersTokenizer(tokenizer)("ab", return_tensors="pt"),
        tokenizer("ab", return_tensors="pt"),
    )


@pytest.mark.parametrize("flag", [None, "0", "true", "1"])
def test_env_is_opt_in_and_read_per_call(tokenizer, monkeypatch, flag) -> None:
    wrapper = TransformersTokenizer(tokenizer)
    # Establish eligibility before changing the environment on this same instance.
    assert wrapper._can_run_kimi_fast_encode("ab", (), {"add_special_tokens": False})
    if flag is None:
        monkeypatch.delenv("K3_TOKENIZER_FASTPATH")
    else:
        monkeypatch.setenv("K3_TOKENIZER_FASTPATH", flag)
    assert wrapper._can_run_kimi_fast_encode("ab", (), {"add_special_tokens": False}) == (
        flag == "1"
    )
    assert wrapper._requires_pt_from_kimi_tokenizer("ab", (), {"return_tensors": "pt"}) == (
        flag == "1"
    )
    assert wrapper.encode("ab", add_special_tokens=False) == [256]
    _assert_batch_equal(wrapper("ab", return_tensors="pt"), tokenizer("ab", return_tensors="pt"))


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((), {}),
        ((), {"add_special_tokens": True}),
        ((), {"add_special_tokens": False, "truncation": True, "max_length": 2}),
        ((None,), {"add_special_tokens": False}),
    ],
)
def test_encode_options_use_hf(tokenizer, monkeypatch, args, kwargs) -> None:
    expected = tokenizer.encode("ab hello", *args, **kwargs)
    spy = Mock(wraps=tokenizer.encode)
    monkeypatch.setattr(tokenizer, "encode", spy)
    assert TransformersTokenizer(tokenizer).encode("ab hello", *args, **kwargs) == expected
    spy.assert_called_once_with("ab hello", *args, **kwargs)


@pytest.mark.parametrize(
    "text,args,kwargs",
    [
        ("ab", (), {"return_tensors": "pt", "add_special_tokens": False}),
        ("ab", (), {"return_tensors": "pt", "return_attention_mask": False}),
        (["ab", "ab"], (), {"return_tensors": "pt"}),
        ("ab", ("ab",), {"return_tensors": "pt"}),
    ],
)
def test_call_options_use_hf(tokenizer, text, args, kwargs) -> None:
    wrapper = TransformersTokenizer(tokenizer)
    assert not wrapper._requires_pt_from_kimi_tokenizer(text, args, kwargs)
    _assert_batch_equal(wrapper(text, *args, **kwargs), tokenizer(text, *args, **kwargs))


@pytest.mark.parametrize(
    "guard", ["do_lower_case", "split_special_tokens", "lstrip", "rstrip", "single_word"]
)
def test_normalization_and_added_token_options_use_hf(tokenizer, monkeypatch, guard) -> None:
    if guard in ("do_lower_case", "split_special_tokens"):
        monkeypatch.setattr(tokenizer, guard, True, raising=False)
    else:
        tokenizer.add_tokens([AddedToken("<extra>", **{guard: True})])
    expected = tokenizer.encode("ab <extra> ab", add_special_tokens=False)
    spy = Mock(wraps=tokenizer.encode)
    monkeypatch.setattr(tokenizer, "encode", spy)
    wrapper = TransformersTokenizer(tokenizer)
    assert wrapper.encode("ab <extra> ab", add_special_tokens=False) == expected
    spy.assert_called_once_with("ab <extra> ab", add_special_tokens=False)


@pytest.mark.parametrize(
    "attribute,value", [("__name__", "OtherTokenizer"), ("__module__", "other.tokenizer")]
)
def test_other_tokenizers_use_hf(tokenizer, monkeypatch, attribute, value) -> None:
    other_type = type(
        "TikTokenTokenizer", (TikTokenTokenizer,), {"__module__": "tests.tokenization_kimi"}
    )
    setattr(other_type, attribute, value)
    tokenizer = other_type()
    wrapper = TransformersTokenizer(tokenizer)
    assert not wrapper._can_run_kimi_fast_encode("ab", (), {"add_special_tokens": False})
    assert not wrapper._requires_pt_from_kimi_tokenizer("ab", (), {"return_tensors": "pt"})
    assert wrapper.encode("ab", add_special_tokens=False) == tokenizer.encode(
        "ab", add_special_tokens=False
    )
    _assert_batch_equal(wrapper("ab", return_tensors="pt"), tokenizer("ab", return_tensors="pt"))
