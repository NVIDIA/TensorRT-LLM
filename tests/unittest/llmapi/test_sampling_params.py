# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SamplingParams that do not require GPU or model weights."""

from tensorrt_llm.sampling_params import SamplingParams


class _MockTokenizer:
    """Minimal tokenizer stub for _setup() without real models."""

    def __init__(self, eos_token_id, pad_token_id=None):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id


def test_setup_scalar_eos_token_id():
    """_setup with a scalar eos_token_id sets end_id correctly."""
    tokenizer = _MockTokenizer(eos_token_id=2, pad_token_id=0)
    params = SamplingParams()
    params._setup(tokenizer, hf_model_config=None, generation_config=None)

    assert params.end_id == 2
    assert params.pad_id == 0
    assert not params.stop_token_ids


def test_setup_list_eos_token_id_single_element():
    """_setup with a single-element list behaves like a scalar."""
    tokenizer = _MockTokenizer(eos_token_id=[128001], pad_token_id=0)
    params = SamplingParams()
    params._setup(tokenizer, hf_model_config=None, generation_config=None)

    assert params.end_id == 128001
    assert params.pad_id == 0
    assert not params.stop_token_ids


def test_setup_list_eos_token_id_multiple_elements():
    """Regression for GitHub issue #11625.

    _setup with a multi-element eos_token_id list (e.g. Llama 3.1) must set
    end_id to a scalar int — not a list — to avoid std::bad_cast in the C++
    binding.  Extra EOS tokens go into stop_token_ids so generation stops on
    all of them.
    """
    tokenizer = _MockTokenizer(eos_token_id=[128001, 128009], pad_token_id=0)
    params = SamplingParams()
    params._setup(tokenizer, hf_model_config=None, generation_config=None)

    assert isinstance(params.end_id, int), (
        "end_id must be int, not list, to avoid std::bad_cast in C++ binding"
    )
    assert params.end_id == 128001
    assert params.stop_token_ids is not None
    assert 128009 in params.stop_token_ids
    assert 128001 not in params.stop_token_ids


def test_setup_list_eos_token_id_no_duplicate_stop_tokens():
    """Extra EOS tokens are not added to stop_token_ids twice.

    When the user already specified them explicitly, no duplicate should appear.
    """
    tokenizer = _MockTokenizer(eos_token_id=[128001, 128009], pad_token_id=0)
    params = SamplingParams(stop_token_ids=[128009])
    params._setup(tokenizer, hf_model_config=None, generation_config=None)

    assert params.end_id == 128001
    assert params.stop_token_ids.count(128009) == 1


def test_setup_end_id_already_set_ignores_tokenizer_eos():
    """When end_id is explicitly provided, _setup must not overwrite it."""
    tokenizer = _MockTokenizer(eos_token_id=[128001, 128009], pad_token_id=0)
    params = SamplingParams(end_id=99)
    params._setup(tokenizer, hf_model_config=None, generation_config=None)

    assert params.end_id == 99
    assert not params.stop_token_ids


def test_setup_list_eos_token_id_pad_fallback():
    """When pad_token_id is None, pad_id falls back to end_id (an int)."""
    tokenizer = _MockTokenizer(eos_token_id=[128001, 128009], pad_token_id=None)
    params = SamplingParams()
    params._setup(tokenizer, hf_model_config=None, generation_config=None)

    assert params.end_id == 128001
    assert params.pad_id == 128001
