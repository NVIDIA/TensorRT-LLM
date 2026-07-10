# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List

from tensorrt_llm.sampling_params import SamplingParams


class _PrefixSpaceTokenizer:
    """Mimics a BPE prefix-space tokenizer (GPT-2 / Qwen).

    A word tokenizes to different ids at the start of a sequence than in the
    middle.
    """

    eos_token_id = 0
    pad_token_id = 0
    # "London" -> [23421], " London" -> [3995] (as reported in the issue).
    _table = {"London": [23421], " London": [3995]}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if text in self._table:
            return list(self._table[text])
        return [ord(c) for c in text]


class _ConstantTokenizer:
    """Encodes any text to the same id.

    The unprefixed and space-prefixed forms of a word coincide, exercising the
    dedup path.
    """

    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [7]


def test_bad_words_records_both_position_variants() -> None:
    sp = SamplingParams(bad="London")
    sp._setup(
        _PrefixSpaceTokenizer(),
        hf_model_config=None,
        generation_config=None,
        add_special_tokens=False,
    )
    bad_words = sp._get_bad_words()
    assert [23421] in bad_words  # start-of-sequence form
    assert [3995] in bad_words  # mid-sequence (space-prefixed) form


def test_bad_words_deduplicates_identical_variants() -> None:
    sp = SamplingParams(bad="hi")
    sp._setup(
        _ConstantTokenizer(), hf_model_config=None, generation_config=None, add_special_tokens=False
    )
    bad_words = sp._get_bad_words()
    assert bad_words.count([7]) == 1


def test_bad_words_handles_list_input() -> None:
    sp = SamplingParams(bad=["London"])
    sp._setup(
        _PrefixSpaceTokenizer(),
        hf_model_config=None,
        generation_config=None,
        add_special_tokens=False,
    )
    bad_words = sp._get_bad_words()
    assert [23421] in bad_words
    assert [3995] in bad_words
