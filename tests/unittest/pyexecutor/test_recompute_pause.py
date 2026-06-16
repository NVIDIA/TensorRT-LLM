# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types
from unittest.mock import Mock

from tensorrt_llm._torch.pyexecutor.py_executor import _UNBOUNDED_PAUSE_MAX_INPUT_LEN, PyExecutor


class _StubRequest:
    def __init__(self) -> None:
        self.reset_for_recompute = Mock()


def test_recompute_pause_does_not_apply_executor_max_input_len() -> None:
    executor = types.SimpleNamespace(max_input_len=5)
    request = _StubRequest()

    PyExecutor._pause_recompute_request(executor, request)

    request.reset_for_recompute.assert_called_once_with(_UNBOUNDED_PAUSE_MAX_INPUT_LEN)
