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

"""Host-side context and generation limits used before scheduling device work."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._torch.pyexecutor.llm_request import LlmRequest


def context_length_exceeded_message(max_context_length: int,
                                    num_prompt_tokens: int) -> str:
    """Client-facing message for a prompt that exceeds the model context.

    Mirrors OpenAI's wording so agent frameworks that pattern-match it (to
    trigger context compaction) work unchanged. Keep in sync with
    ``is_context_length_exceeded_message``.
    """
    return (
        f"This model's maximum context length is {max_context_length} tokens. "
        f"However, your messages resulted in {num_prompt_tokens} tokens. "
        "Please reduce the length of the messages.")


def generation_capacity(prompt_len: int, max_seq_len: int, cp_size: int = 1) -> int:
    """Available output tokens under the worker's per-CP-rank context limit."""
    capacity = max_seq_len - prompt_len // cp_size
    if capacity <= 0:
        raise ValueError(context_length_exceeded_message(max_seq_len * cp_size, prompt_len))
    return capacity


def clamp_generation_request(
    request: LlmRequest, max_seq_len: int, cp_size: int, *, draft_reserve: int = 0
) -> None:
    """Apply the same limit to the native scheduler and the Python sampler."""
    prompt_len = request.total_input_len_cp
    capacity = generation_capacity(prompt_len, max(0, max_seq_len - draft_reserve), cp_size)
    if request.max_new_tokens > capacity:
        request.max_new_tokens = capacity
        request.py_max_new_tokens = capacity

