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
"""The text-only fast tokenization path of the Kimi K2.5/K3 input processor.

The fast path must return exactly the ids of ``tokenizer(text, return_tensors="pt")``.
``_TikTokenLikeTokenizer`` follows the structure of the Kimi ``TikTokenTokenizer``
(a slow tokenizer whose ``encode`` calls tiktoken directly) with a tiny vocabulary,
so these tests need neither a checkpoint nor a GPU.
"""

import os
import random
from typing import Dict, List
from unittest import mock

import pytest
import tiktoken
from tokenizers import AddedToken
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.convert_slow_tokenizer import bytes_to_unicode
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.models import modeling_kimi_k25
from tensorrt_llm._torch.models.modeling_kimi_k25 import KimiK25InputProcessor, _FastTextTokenizer
from tensorrt_llm.tokenizer import TransformersTokenizer

pytestmark = pytest.mark.cpu_only

_PAT_STR = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?\p{L}+""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)
_MERGES = [b"He", b"ll", b"llo", b"Hello", b" w", b" wor", b"ld", b" world", b"  ", b"    "]
_ADDED = ["<|im_end|>", "<|im_user|>", "<|im_assistant|>", "<|im_middle|>", "[EOT]"]
_RESERVED = ["[BOS]", "[EOS]", "<|media_pad|>"]


class _TikTokenLikeTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs):
        ranks = {bytes([i]): i for i in range(256)}
        for merge in _MERGES:
            ranks[merge] = len(ranks)
        special = {tok: len(ranks) + i for i, tok in enumerate(_RESERVED + _ADDED)}
        self.model = tiktoken.Encoding(
            name="kimi-like-test",
            pat_str=_PAT_STR,
            mergeable_ranks=ranks,
            special_tokens=special,
        )
        byte_encoder = bytes_to_unicode()
        self.decoder = {
            i: "".join(
                byte_encoder[ord(c)]
                for c in self.model.decode_single_token_bytes(i).decode("latin-1")
            )
            for i in range(self.model.n_vocab)
        }
        self.encoder = {v: k for k, v in self.decoder.items()}
        additional = [AddedToken(t, special=True) for t in _ADDED]
        super().__init__(
            bos_token="[BOS]",
            eos_token="[EOS]",
            additional_special_tokens=additional,
            **kwargs,
        )

    def encode(self, text: str, allow_special_tokens: bool = True, **kwargs) -> List[int]:
        if kwargs:
            return super().encode(text, **kwargs)
        if allow_special_tokens:
            return self.model.encode(text, allowed_special="all")
        return self.model.encode(text, disallowed_special=())

    @property
    def vocab_size(self) -> int:
        return self.model.n_vocab

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.encoder)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return [self.decoder[t] for t in self.encode(text)]

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, 0)

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.model.decode([self.encoder[t] for t in tokens])


def _reference_ids(tokenizer, text: str) -> List[int]:
    return tokenizer(text, return_tensors="pt").input_ids[0].tolist()


def _texts() -> List[str]:
    added = "".join(_ADDED)
    texts = [
        "",
        "Hello world",
        "Hello, 世界! 🙂 tab\there\r\n  x  y   end",
        added,
        added + added,
        "<|im_user|>user<|im_middle|>Hello world<|im_end|><|im_assistant|>",
        "<|im_end" + "|><|im_end|>|>",
        "<|im_" + "user",
        "[EOT][EOT] [EOT]\n[EOT]",
        "[BOS] literal reserved tokens [EOS] <|media_pad|>",
        "<<|im_end|>>",
        "\n\n\n    \t\t  \u00a0\u3000x",
        "e\u0301 a\u0308 \U0001f469\u200d\U0001f4bb",
        "数" * 50 + " 123456789 " + "x" * 300,
        " " * 2000 + "y",
    ]
    rng = random.Random(0)
    alphabet = list("ab Hl lo wr d\n\t,.!?<|>[]_") + ["世", "🙂", "\u0301"] + _ADDED + _RESERVED
    for _ in range(200):
        texts.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 40))))
    return texts


@pytest.fixture(scope="module")
def tokenizer():
    return _TikTokenLikeTokenizer()


def test_fast_path_ids_match_tokenizer_call(tokenizer):
    fast = _FastTextTokenizer.create(tokenizer)
    assert fast is not None
    for text in _texts():
        assert fast(text) == _reference_ids(tokenizer, text), repr(text)


def test_fast_path_unwraps_transformers_tokenizer(tokenizer):
    fast = _FastTextTokenizer.create(TransformersTokenizer(tokenizer))
    assert fast is not None
    text = "<|im_user|>Hello world<|im_end|>"
    assert fast(text) == _reference_ids(tokenizer, text)


def test_fast_path_ids_match_with_extra_added_token():
    tokenizer = _TikTokenLikeTokenizer()
    tokenizer.add_tokens([AddedToken("<|im_extra|>", special=True)], special_tokens=True)
    fast = _FastTextTokenizer.create(tokenizer)
    assert fast is not None
    for text in ["a<|im_extra|>b", "<|im_extra|><|im_end|><|im_extra", " <|im_extra|> "]:
        assert fast(text) == _reference_ids(tokenizer, text), repr(text)


@pytest.mark.parametrize(
    "added_token_kwargs", [{"lstrip": True}, {"rstrip": True}, {"single_word": True}]
)
def test_fast_path_rejects_stripping_added_tokens(added_token_kwargs):
    tokenizer = _TikTokenLikeTokenizer()
    tokenizer.add_tokens(
        [AddedToken("<|im_extra|>", special=True, **added_token_kwargs)], special_tokens=True
    )
    assert _FastTextTokenizer.create(tokenizer) is None


def test_fast_path_rejects_split_special_tokens():
    assert _FastTextTokenizer.create(_TikTokenLikeTokenizer(split_special_tokens=True)) is None


def test_fast_path_rejects_tokenizer_without_encode_override(tokenizer):
    class _NoEncodeOverride(_TikTokenLikeTokenizer):
        encode = PreTrainedTokenizer.encode

    assert _FastTextTokenizer.create(_NoEncodeOverride()) is None


def test_fast_path_rejects_probe_mismatch(tokenizer):
    with mock.patch.object(_FastTextTokenizer, "__call__", return_value=[0]):
        assert _FastTextTokenizer.create(tokenizer) is None


def _bare_processor(tokenizer) -> KimiK25InputProcessor:
    processor = KimiK25InputProcessor.__new__(KimiK25InputProcessor)
    processor._tokenizer = tokenizer
    processor._fast_text = None
    processor._fast_text_checked = False
    return processor


@pytest.mark.parametrize("enabled", ["1", "0"])
def test_input_processor_text_only_ids(tokenizer, enabled):
    text = "<|im_system|>Hello<|im_end|><|im_user|>Hello, 世界<|im_end|><|im_assistant|>"
    processor = _bare_processor(tokenizer)
    with mock.patch.dict(os.environ, {modeling_kimi_k25._FAST_TEXT_TOKENIZE_ENV: enabled}):
        token_ids, extra = processor.call_with_text_prompt({"prompt": text}, sampling_params=None)
        assert (processor._fast_text_tokenizer() is not None) == (enabled == "1")
    assert token_ids == _reference_ids(tokenizer, text)
    assert extra == {}


_KIMI_K25_MODEL_DIR = os.path.join(llm_models_root(), "Kimi-K2.5-NVFP4")


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(_KIMI_K25_MODEL_DIR, "tiktoken.model")),
    reason=f"Kimi K2.5 tokenizer not found under {_KIMI_K25_MODEL_DIR}",
)
def test_fast_path_ids_match_kimi_k25_tokenizer():
    kimi = AutoTokenizer.from_pretrained(_KIMI_K25_MODEL_DIR, trust_remote_code=True)
    fast = _FastTextTokenizer.create(kimi)
    assert fast is not None
    added = "".join(kimi.added_tokens_encoder)
    for text in _texts() + [added, added[3:-3], "def f(x):\n    return x  # <|im_end|>\n" * 500]:
        assert fast(text) == _reference_ids(kimi, text), repr(text[:200])
