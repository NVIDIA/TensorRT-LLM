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
"""The executor side of the batch-wait skip: what counts as context work held for the next step,
and the model opt-in with its kill switch."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

pytestmark = pytest.mark.cpu_only


GEN_ONLY = "generation_only"
CONTEXT = "context_and_generation"


@pytest.fixture(autouse=True)
def request_types(monkeypatch):
    monkeypatch.setattr(
        py_executor, "RequestType", SimpleNamespace(REQUEST_TYPE_GENERATION_ONLY=GEN_ONLY)
    )


def _item(request_type):
    return SimpleNamespace(request=SimpleNamespace(request_type=request_type))


def _executor(waiting, states):
    executor = object.__new__(PyExecutor)
    executor.waiting_queue = [_item(t) for t in waiting]
    executor.active_requests = [SimpleNamespace(is_context_init_state=s) for s in states]
    return executor


@pytest.mark.parametrize(
    "waiting, states, held",
    [
        ([], [], False),
        ([GEN_ONLY, GEN_ONLY], [False], False),
        ([GEN_ONLY, CONTEXT], [], True),
        ([GEN_ONLY], [True], True),
    ],
)
def test_context_work_is_a_waiting_context_request_or_an_active_context_request(
    waiting, states, held
):
    assert _executor(waiting, states)._holds_context_work() is held


def test_the_predicate_reads_the_current_queues():
    executor = _executor([], [])
    predicate = executor._holds_context_work
    assert not predicate()
    executor.waiting_queue.append(_item(CONTEXT))
    assert predicate()
    executor.waiting_queue.clear()
    executor.active_requests = [SimpleNamespace(is_context_init_state=True)]
    assert predicate()


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("killed", [False, True])
def test_the_hook_turns_the_skip_on_unless_killed(monkeypatch, requested, killed):
    monkeypatch.setattr(py_executor, "_CTX_BATCH_WAIT_SKIP_KILLED", killed)
    hooks = RuntimeHooks(ctx_batch_wait_skip=requested)
    assert py_executor._ctx_batch_wait_skip_enabled(hooks) is (requested and not killed)
