# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prompt token ids outside the vocabulary are rejected before submission."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.llmapi.llm import BaseLLM, _check_prompt_token_id_range

pytestmark = pytest.mark.cpu_only

_VOCAB = 1000


def _as(kind, ids):
    if kind == "list":
        return list(ids)
    if kind == "numpy":
        return np.asarray(ids, dtype=np.int32)
    return torch.tensor(ids, dtype=torch.int32)


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_in_range_ids_are_accepted(kind):
    _check_prompt_token_id_range(_as(kind, [0, 5, _VOCAB - 1]), _VOCAB, allow_out_of_vocab=False)


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_id_at_vocab_size_is_rejected(kind):
    with pytest.raises(RequestError, match=f"Prompt token id {_VOCAB} is out of range"):
        _check_prompt_token_id_range(_as(kind, [1, _VOCAB, 2]), _VOCAB, allow_out_of_vocab=False)


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_negative_id_is_rejected(kind):
    with pytest.raises(RequestError, match="must be non-negative, got -1"):
        _check_prompt_token_id_range(_as(kind, [1, -1, 2]), _VOCAB, allow_out_of_vocab=False)


def test_negative_id_is_rejected_even_when_out_of_vocab_is_allowed():
    with pytest.raises(RequestError, match="must be non-negative"):
        _check_prompt_token_id_range([-5, 3], _VOCAB, allow_out_of_vocab=True)


def test_out_of_vocab_placeholders_are_allowed_when_requested():
    _check_prompt_token_id_range([1, _VOCAB, _VOCAB + 7], _VOCAB, allow_out_of_vocab=True)


def test_unknown_vocab_size_only_checks_the_sign():
    _check_prompt_token_id_range([1, 10**9], None, allow_out_of_vocab=False)


def test_empty_prompt_is_left_to_other_checks():
    _check_prompt_token_id_range([], _VOCAB, allow_out_of_vocab=False)


@pytest.mark.parametrize(
    "hf_model_config,expected",
    [
        (None, None),
        (SimpleNamespace(), None),
        (SimpleNamespace(vocab_size=_VOCAB), _VOCAB),
        (SimpleNamespace(vocab_size=0), None),
        # Multimodal configs keep the text embedding size in text_config.
        (SimpleNamespace(vocab_size=5, text_config=SimpleNamespace(vocab_size=_VOCAB)), _VOCAB),
        (SimpleNamespace(vocab_size=_VOCAB, text_config=SimpleNamespace()), _VOCAB),
    ],
)
def test_prompt_vocab_size_reads_the_text_config_first(hf_model_config, expected):
    llm = SimpleNamespace(_hf_model_config=hf_model_config)
    assert BaseLLM._prompt_vocab_size(llm) == expected
