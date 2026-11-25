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
"""The executor side of scheduler_config.gen_init_pause_after_release. No GPU required."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


def _request(request_id, state=LlmRequestState.CONTEXT_INIT, first_chunk=True, dummy=False):
    return SimpleNamespace(
        py_request_id=request_id,
        state=state,
        is_first_context_chunk=first_chunk,
        is_dummy_request=dummy,
        is_generation_only_request=state == LlmRequestState.GENERATION_IN_PROGRESS,
    )


def test_a_finished_generation_request_is_noted_for_the_scheduler() -> None:
    ex = object.__new__(PyExecutor)
    ex._free_request_resources = Mock()
    ex._release_multimodal_resources = Mock()
    ex.gather_all_responses = False
    ex.dist = SimpleNamespace(rank=1)
    note = ex._note_generation_release = Mock()

    ex._do_terminate_request(_request(1, state=LlmRequestState.GENERATION_IN_PROGRESS))
    ex._do_terminate_request(_request(2))  # a context request
    ex._do_terminate_request(_request(3, state=LlmRequestState.GENERATION_IN_PROGRESS, dummy=True))

    note.assert_called_once_with()
