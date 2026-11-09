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
"""The executor side of the batch-wait skip: what counts as context work held for the next step, the process-wide
arm and its kill switch, and the predicate the executor installs on its request queue. CPU only."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

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
        ([], [False, False], False),
        ([GEN_ONLY, GEN_ONLY], [False], False),
        ([CONTEXT], [], True),
        ([GEN_ONLY, CONTEXT], [], True),
        ([], [False, True], True),
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


@pytest.mark.parametrize("killed", [False, True])
def test_arm_turns_the_skip_on_unless_killed(monkeypatch, killed):
    monkeypatch.setattr(py_executor, "_CTX_BATCH_WAIT_SKIP", False)
    monkeypatch.setattr(py_executor, "_CTX_BATCH_WAIT_SKIP_KILLED", killed)
    py_executor.arm_ctx_batch_wait_skip()
    assert py_executor._CTX_BATCH_WAIT_SKIP is not killed


def test_the_skip_is_off_at_import():
    """The module-level default (read from source, so tests that arm the flag cannot mask it)."""
    tree = ast.parse(inspect.getsource(py_executor))
    defaults = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and [getattr(t, "id", None) for t in node.targets] == ["_CTX_BATCH_WAIT_SKIP"]
    ]
    assert len(defaults) == 1
    default = defaults[0]
    if isinstance(default, ast.Constant):
        assert default.value is False
        return
    # MSO arms it from the environment (no target arms it yet): the default must be
    # `os.environ.get("TRTLLM_CTX_BATCH_WAIT_SKIP", "0") == "1" and not <kill switch>`,
    # i.e. off when the variable is unset.
    assert isinstance(default, ast.BoolOp) and isinstance(default.op, ast.And)
    env_check = default.values[0]
    assert ast.unparse(env_check) == "os.environ.get('TRTLLM_CTX_BATCH_WAIT_SKIP', '0') == '1'"


def test_the_executor_installs_the_predicate_only_when_armed():
    """PyExecutor.__init__ installs _holds_context_work on the request queue it builds, under the armed flag."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(PyExecutor.__init__)))
    installs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "_CTX_BATCH_WAIT_SKIP"
    ]
    assert len(installs) == 1
    (assign,) = installs[0].body
    assert ast.unparse(assign) == (
        "self.executor_request_queue.skip_batch_wait_when = self._holds_context_work"
    )
