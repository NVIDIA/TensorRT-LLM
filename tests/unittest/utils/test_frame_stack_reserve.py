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
"""run_with_frame_stack_reserve: call semantics, version gate and data-stack churn (no GPU)."""

import os
import subprocess
import sys
import threading

import pytest

from tensorrt_llm._utils import (
    _FRAME_STACK_RESERVE_DISABLE_ENV,
    _FRAME_STACK_RESERVE_SLOTS,
    _frame_stack_reserve_needed,
    _needs_frame_stack_reserve,
    run_with_frame_stack_reserve,
)

pytestmark = pytest.mark.cpu_only

_CALLS_PER_DEPTH = 2000
# Enough depths to cross several ordinary data-stack chunks.
_DEPTHS = 300


def _leaf():
    return None


# A larger leaf frame makes it cross a chunk boundary during the sweep.
_leaf.__code__ = _leaf.__code__.replace(co_stacksize=64)


def _call_leaf_at_depth(depth, calls):
    if depth:
        return _call_leaf_at_depth(depth - 1, calls)
    for _ in range(calls):
        _leaf()


def _sweep_depths():
    for depth in range(_DEPTHS):
        _call_leaf_at_depth(depth, _CALLS_PER_DEPTH)


def _minor_faults_in_new_thread(fn):
    # Each freshly mapped data-stack chunk shows up as a minor page fault.
    resource = pytest.importorskip("resource")
    result = {}

    def target():
        before = resource.getrusage(resource.RUSAGE_THREAD).ru_minflt
        fn()
        result["faults"] = resource.getrusage(resource.RUSAGE_THREAD).ru_minflt - before

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    return result["faults"]


def test_forwards_arguments_and_result():
    def echo(*args, **kwargs):
        return args, kwargs

    # ``fn`` is positional-only, so a keyword argument of the same name reaches the callee.
    assert run_with_frame_stack_reserve(echo, 1, 2, fn="x") == ((1, 2), {"fn": "x"})


def test_exception_propagates_unchanged():
    error = ValueError("boom")

    def crash():
        raise error

    with pytest.raises(ValueError) as excinfo:
        run_with_frame_stack_reserve(crash)
    assert excinfo.value is error
    assert excinfo.value.__context__ is None


@pytest.mark.parametrize(
    "implementation, version, expected",
    [
        ("cpython", (3, 10, 16), False),
        ("cpython", (3, 11, 0), True),
        ("cpython", (3, 12, 3, "final", 0), True),
        ("cpython", (3, 13, 12), True),
        ("cpython", (3, 13, 13), False),
        ("cpython", (3, 14, 3), True),
        ("cpython", (3, 14, 4), False),
        ("cpython", (3, 15, 0), False),
        ("pypy", (3, 12, 3), False),
    ],
)
def test_gate(implementation, version, expected):
    assert _frame_stack_reserve_needed(implementation, version) is expected


def test_reserve_applied_at_import_iff_gate_holds():
    reserved = run_with_frame_stack_reserve.__code__.co_stacksize >= _FRAME_STACK_RESERVE_SLOTS
    assert reserved is _needs_frame_stack_reserve()


def test_env_var_disables_reserve_at_import():
    env = dict(os.environ, **{_FRAME_STACK_RESERVE_DISABLE_ENV: "1"})
    code = (
        "from tensorrt_llm import _utils as u\n"
        "reserved = u.run_with_frame_stack_reserve.__code__.co_stacksize"
        " >= u._FRAME_STACK_RESERVE_SLOTS\n"
        "print('RESULT', int(reserved))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    assert proc.stdout.strip().splitlines()[-1] == "RESULT 0"


@pytest.mark.skipif(
    sys.platform != "linux" or not _needs_frame_stack_reserve(),
    reason="needs an affected CPython and per-thread rusage",
)
def test_reserve_removes_chunk_churn():
    # Without the reserve, the sweep must hit a chunk boundary, or the second check proves nothing.
    assert _minor_faults_in_new_thread(_sweep_depths) > _CALLS_PER_DEPTH
    faults = _minor_faults_in_new_thread(lambda: run_with_frame_stack_reserve(_sweep_depths))
    assert faults < _CALLS_PER_DEPTH
