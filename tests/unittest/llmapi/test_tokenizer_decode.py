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

from tensorrt_llm.tokenizer import TransformersTokenizer


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


def _byte_fallback_tokenizer() -> TransformersTokenizer:
    """A tokenizer with one added special token, wrapped the way the LLM API wraps it."""
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
