# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types

from tensorrt_llm._torch.pyexecutor.py_executor import _UNBOUNDED_PAUSE_MAX_INPUT_LEN, PyExecutor


class _StubRequest:
    def __init__(self) -> None:
        self.generated_tokens = 6
        self.max_new_tokens = 20
        self.prompt_len = 4
        self.pause_max_input_len = None

    def pause(self, max_input_len: int) -> None:
        self.pause_max_input_len = max_input_len
        new_prompt_len = min(max_input_len, self.prompt_len + self.generated_tokens)
        self.max_new_tokens -= new_prompt_len - self.prompt_len
        self.prompt_len = new_prompt_len


def test_recompute_pause_does_not_apply_executor_max_input_len() -> None:
    executor = types.SimpleNamespace(max_input_len=5)
    request = _StubRequest()

    PyExecutor._pause_recompute_request(executor, request)

    assert request.pause_max_input_len == _UNBOUNDED_PAUSE_MAX_INPUT_LEN
    assert request.prompt_len == 10
    assert request.py_prompt_len == 10
    assert request.py_orig_prompt_len == 10
    assert request.py_max_new_tokens == 14
    assert request.py_draft_tokens == []
    assert request.draft_tokens == []
    assert request._cached_tokens == 0
    assert request._cached_tokens_set is False
