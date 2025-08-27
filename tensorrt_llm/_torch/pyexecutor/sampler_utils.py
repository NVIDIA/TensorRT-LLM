# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from itertools import pairwise

from tensorrt_llm.bindings.executor import FinishReason

from .llm_request import LlmRequest

BEAM_0 = 0
SINGLE_BEAM_WIDTH = 1


def max_token_criteria_1_beam(request: LlmRequest, max_seq_len: int) -> bool:
    num_tokens = request.get_num_tokens(BEAM_0)
    return (num_tokens - request.py_orig_prompt_len
            >= request.py_max_new_tokens) or (num_tokens >= max_seq_len)


def produce_stop_words(py_stop_words_list: list[list[int]]):
    """inverts `.../llm_request::def convert_wordlist()`"""
    stop_words_list, prefix_sum = py_stop_words_list
    for start, end in pairwise((0, *prefix_sum)):  # first element: prepend 0
        if end == -1:  # -1 is a sentinel value in convert_wordlist
            break
        yield stop_words_list[start:end]


def stop_token_criteria(py_stop_words_list: list[list[int]] | None,
                        tokens: list[int]) -> bool:
    if py_stop_words_list:
        assert isinstance(py_stop_words_list,
                          list), "request.py_stop_words_list should be a list"
        for stop_word in produce_stop_words(py_stop_words_list):
            if len(stop_word) > len(tokens):
                continue
            if tokens[-len(stop_word):] == stop_word:
                return True
    return False


def handle_stop_1_beam(request: LlmRequest, new_token: int, *,
                       max_seq_len: int) -> bool:
    """Handle stop criteria and set appropriate finish reasons and state.
    Returns True if generation should stop."""
    if new_token == request.py_end_id:
        request.finish_by(FinishReason.END_ID, BEAM_0)
        return True

    if max_token_criteria_1_beam(request, max_seq_len):
        request.finish_by(FinishReason.LENGTH, BEAM_0)
        return True

    if stop_token_criteria(request.py_stop_words_list,
                           request.get_tokens(BEAM_0)):
        request.finish_by(FinishReason.STOP_WORDS, BEAM_0)
        return True

    return False
